#!/usr/bin/env python3
"""Build ResNet101 caches compatible with analyze_privileged_distillation.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ResNet101 logits/features into distillation cache format."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--normalize-features", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None, run_dir: Path) -> Path:
    if output_dir_arg is not None:
        return Path(output_dir_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_resnet101_distill_cache"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_ckpt_path(run_dir: Path, ckpt_path_arg: str | None) -> Path:
    if ckpt_path_arg is not None:
        return Path(ckpt_path_arg)
    checkpoints_dir = run_dir / "checkpoints"
    epoch_ckpts = sorted(checkpoints_dir.glob("epoch_*.ckpt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    candidate = checkpoints_dir / "last.ckpt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No checkpoint found under {checkpoints_dir}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("ema_model."):
            continue
        new_key = key
        if ".net._orig_mod." in new_key:
            new_key = new_key.replace(".net._orig_mod.", ".net.")
        elif new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod.") :]
        normalized[new_key] = value
    return normalized


def _dataset_and_loader_from_datamodule(datamodule, split: str) -> tuple[Any, DataLoader]:
    dataset_attr = {"train": "data_train", "val": "data_val", "test": "data_test"}[split]
    loader_fn = {"train": datamodule.train_dataloader, "val": datamodule.val_dataloader, "test": datamodule.test_dataloader}[split]
    dataset = getattr(datamodule, dataset_attr)
    if dataset is None:
        raise RuntimeError(f"Datamodule did not initialize split '{split}'.")
    return dataset, loader_fn()


def _get_backbone_net(model: torch.nn.Module) -> torch.nn.Module:
    net = model.net
    return net._orig_mod if hasattr(net, "_orig_mod") else net


def _extract_resnet_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    net = _get_backbone_net(model)
    features = net.backbone(images)
    features = net.avgpool(features)
    features = net.flatten(features)
    return features


def _extract_dual_features(
    model: torch.nn.Module,
    crop_images: torch.Tensor,
    full_images: torch.Tensor | None,
    normalize_features: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    crop_features = _extract_resnet_features(model, crop_images).float()
    full_features = None
    if full_images is not None:
        full_features = _extract_resnet_features(model, full_images).float()

    if normalize_features:
        crop_features = F.normalize(crop_features, dim=-1)
        if full_features is not None:
            full_features = F.normalize(full_features, dim=-1)

    fused_features = (
        torch.cat([crop_features, full_features], dim=-1)
        if full_features is not None
        else crop_features
    )
    return crop_features, full_features, fused_features


def _collect_base_and_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
    normalize_features: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    logits_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    features_all: List[np.ndarray] = []
    crop_features_all: List[np.ndarray] = []
    full_features_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    soft_all: List[np.ndarray] = []
    image_ids_all: List[str] = []

    collected = 0
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            full_images = None
            if "image_full" in batch:
                full_images = batch["image_full"].to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            crop_features, full_features, features = _extract_dual_features(
                model=model,
                crop_images=images,
                full_images=full_images,
                normalize_features=normalize_features,
            )

            logits_cpu = logits.detach().float().cpu()
            scores_cpu = torch.sigmoid(logits_cpu)
            features_cpu = features.detach().cpu()
            crop_features_cpu = crop_features.detach().cpu()
            full_features_cpu = full_features.detach().cpu() if full_features is not None else None
            labels_cpu = batch["labels"].detach().float().cpu()
            soft_cpu = batch["soft_labels"].detach().float().cpu()

            image_ids = batch["image_id"]
            if torch.is_tensor(image_ids):
                image_ids_batch = [str(x) for x in image_ids.detach().cpu().tolist()]
            else:
                image_ids_batch = [str(x) for x in image_ids]

            if max_samples is not None:
                remaining = int(max_samples) - collected
                if remaining <= 0:
                    break
                if logits_cpu.shape[0] > remaining:
                    logits_cpu = logits_cpu[:remaining]
                    scores_cpu = scores_cpu[:remaining]
                    features_cpu = features_cpu[:remaining]
                    crop_features_cpu = crop_features_cpu[:remaining]
                    if full_features_cpu is not None:
                        full_features_cpu = full_features_cpu[:remaining]
                    labels_cpu = labels_cpu[:remaining]
                    soft_cpu = soft_cpu[:remaining]
                    image_ids_batch = image_ids_batch[:remaining]

            logits_all.append(logits_cpu.numpy())
            scores_all.append(scores_cpu.numpy())
            features_all.append(features_cpu.numpy())
            crop_features_all.append(crop_features_cpu.numpy())
            if full_features_cpu is not None:
                full_features_all.append(full_features_cpu.numpy())
            labels_all.append(labels_cpu.numpy())
            soft_all.append(soft_cpu.numpy())
            image_ids_all.extend(image_ids_batch)

            collected += len(image_ids_batch)
            if max_samples is not None and collected >= int(max_samples):
                break

    base_bundle = {
        "logits": np.concatenate(logits_all, axis=0),
        "scores": np.concatenate(scores_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "soft_labels": np.concatenate(soft_all, axis=0),
        "image_ids": image_ids_all,
    }
    feature_bundle = {
        "features": np.concatenate(features_all, axis=0),
        "crop_features": np.concatenate(crop_features_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "soft_labels": np.concatenate(soft_all, axis=0),
        "image_ids": image_ids_all,
    }
    if full_features_all:
        feature_bundle["full_features"] = np.concatenate(full_features_all, axis=0)
    return base_bundle, feature_bundle


def _save_npz(path: Path, bundle: Dict[str, Any]) -> None:
    arrays: Dict[str, Any] = {}
    for key, value in bundle.items():
        arrays[key] = np.asarray(value) if not isinstance(value, np.ndarray) else value
    np.savez(path, **arrays)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)
    output_dir = _resolve_output_dir(args.output_dir, run_dir)
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.pin_memory = bool(args.pin_memory)
    # Distillation expects binarized training labels while preserving original agreement
    # scores in soft_labels, matching the CLIP cache format used in prior experiments.
    cfg.data.binarize_softprob = True

    model = instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_state_dict_keys(checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)

    device = _resolve_device(args.device)
    model = model.eval().to(device)

    datamodule = instantiate(cfg.data)
    if hasattr(datamodule, "val_test_transform") and hasattr(datamodule, "train_transform"):
        datamodule.train_transform = datamodule.val_test_transform
    datamodule.prepare_data()
    datamodule.setup()

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "device": str(device),
        "normalize_features": bool(args.normalize_features),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        dataset, loader = _dataset_and_loader_from_datamodule(datamodule, split)
        base_bundle, feature_bundle = _collect_base_and_features(
            model=model,
            dataloader=loader,
            device=device,
            max_samples=args.max_samples,
            normalize_features=bool(args.normalize_features),
        )
        _save_npz(cache_dir / f"{split}_base.npz", base_bundle)
        _save_npz(cache_dir / f"{split}_clip.npz", feature_bundle)
        summary["splits"][split] = {
            "num_samples": len(base_bundle["image_ids"]),
            "feature_dim": int(feature_bundle["features"].shape[1]),
        }
        print(
            f"[ResNet101Cache] split={split} samples={len(base_bundle['image_ids'])} "
            f"feature_dim={feature_bundle['features'].shape[1]}"
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ResNet101Cache] output_dir={output_dir}")


if __name__ == "__main__":
    main()
