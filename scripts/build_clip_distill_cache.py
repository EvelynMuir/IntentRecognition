#!/usr/bin/env python3
"""Build CLIP ViT-L/14 distillation caches with crop+full-image fused features."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is required for build_clip_distill_cache.py. "
        "Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP image features into distillation cache format."
    )
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "train.yaml"))
    parser.add_argument("--data", type=str, default="emotic")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--clip-model-name", type=str, default="ViT-L/14")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is not None:
        return Path(output_dir_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_clip_distill_cache"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_npz(path: Path, bundle: Dict[str, Any]) -> None:
    arrays: Dict[str, Any] = {}
    for key, value in bundle.items():
        arrays[key] = np.asarray(value) if not isinstance(value, np.ndarray) else value
    np.savez(path, **arrays)


def _collect_clip_features(
    clip_model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    fused_features_all: List[np.ndarray] = []
    crop_features_all: List[np.ndarray] = []
    full_features_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    soft_all: List[np.ndarray] = []
    image_ids_all: List[str] = []

    collected = 0
    clip_model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            crop_images = batch["image"].to(device, non_blocking=True)
            full_images = batch["image_full"].to(device, non_blocking=True)

            crop_features = F.normalize(clip_model.encode_image(crop_images).float(), dim=-1)
            full_features = F.normalize(clip_model.encode_image(full_images).float(), dim=-1)
            fused_features = torch.cat([crop_features, full_features], dim=-1)

            labels_cpu = batch["labels"].detach().float().cpu()
            soft_cpu = batch["soft_labels"].detach().float().cpu()
            crop_cpu = crop_features.detach().cpu()
            full_cpu = full_features.detach().cpu()
            fused_cpu = fused_features.detach().cpu()

            image_ids = batch["image_id"]
            image_ids_batch = [str(x) for x in image_ids] if not torch.is_tensor(image_ids) else [str(x) for x in image_ids.detach().cpu().tolist()]

            if max_samples is not None:
                remaining = int(max_samples) - collected
                if remaining <= 0:
                    break
                if fused_cpu.shape[0] > remaining:
                    labels_cpu = labels_cpu[:remaining]
                    soft_cpu = soft_cpu[:remaining]
                    crop_cpu = crop_cpu[:remaining]
                    full_cpu = full_cpu[:remaining]
                    fused_cpu = fused_cpu[:remaining]
                    image_ids_batch = image_ids_batch[:remaining]

            labels_all.append(labels_cpu.numpy())
            soft_all.append(soft_cpu.numpy())
            crop_features_all.append(crop_cpu.numpy())
            full_features_all.append(full_cpu.numpy())
            fused_features_all.append(fused_cpu.numpy())
            image_ids_all.extend(image_ids_batch)

            collected += len(image_ids_batch)
            if max_samples is not None and collected >= int(max_samples):
                break

    return {
        "features": np.concatenate(fused_features_all, axis=0),
        "crop_features": np.concatenate(crop_features_all, axis=0),
        "full_features": np.concatenate(full_features_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "soft_labels": np.concatenate(soft_all, axis=0),
        "image_ids": image_ids_all,
    }


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    cfg = OmegaConf.load(args.config)
    with open_dict(cfg):
        cfg.paths = {"root_dir": str(PROJECT_ROOT)}
    cfg.data = OmegaConf.load(PROJECT_ROOT / "configs" / "data" / f"{args.data}.yaml")
    cfg.data.batch_size = int(args.batch_size)
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.pin_memory = bool(args.pin_memory)

    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    device = _resolve_device(args.device)
    clip_model, clip_preprocess = clip.load(str(args.clip_model_name), device=device)
    clip_model = clip_model.eval().to(device)
    for dataset in [datamodule.data_train, datamodule.data_val, datamodule.data_test]:
        dataset.transform = clip_preprocess

    summary: Dict[str, Any] = {
        "device": str(device),
        "clip_model_name": str(args.clip_model_name),
        "splits": {},
    }

    for split, loader in [
        ("train", datamodule.train_dataloader()),
        ("val", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ]:
        bundle = _collect_clip_features(
            clip_model=clip_model,
            dataloader=loader,
            device=device,
            max_samples=args.max_samples,
        )
        _save_npz(cache_dir / f"{split}_clip.npz", bundle)
        summary["splits"][split] = {
            "num_samples": len(bundle["image_ids"]),
            "feature_dim": int(bundle["features"].shape[1]),
        }
        print(
            f"[CLIPCache] split={split} samples={len(bundle['image_ids'])} "
            f"feature_dim={bundle['features'].shape[1]}"
        )

    (output_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"[CLIPCache] saved to {output_dir}")


if __name__ == "__main__":
    main()
