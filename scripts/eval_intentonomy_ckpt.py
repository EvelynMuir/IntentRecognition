#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import SUBSET2IDS, eval_validation_set


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Intentonomy checkpoint with original metrics plus easy/medium/hard."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing .hydra/config.yaml",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Checkpoint path. Default: <run-dir>/checkpoints/epoch_012.ckpt if exists, else last.ckpt",
    )
    parser.add_argument(
        "--use-inference-strategy",
        action="store_true",
        help="Use inference fallback strategy for all-zero predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override dataloader workers for evaluation.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable dataloader pin_memory (default False).",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading state_dict.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for metrics JSON.",
    )
    return parser.parse_args()


def _resolve_ckpt_path(run_dir: Path, ckpt_path: str | None) -> Path:
    if ckpt_path is not None:
        return Path(ckpt_path)
    
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        epoch_ckpts = sorted(checkpoints_dir.glob("epoch_*.ckpt"))
        if epoch_ckpts:
            candidate = epoch_ckpts[-1]
        else:
            candidate = checkpoints_dir / "last.ckpt"
    else:
        candidate = checkpoints_dir / "last.ckpt"
    if candidate.exists():
        return candidate
    return run_dir / "checkpoints" / "last.ckpt"


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for k, v in state_dict.items():
        if k.startswith("net._orig_mod."):
            normalized["net." + k[len("net._orig_mod.") :]] = v
        else:
            normalized[k] = v
    return normalized


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _collect_scores_targets(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits, _ = model(images, return_slots=False)
            preds = torch.sigmoid(logits).cpu().numpy()
            preds_all.append(preds)
            targets_all.append(labels)

    scores = np.concatenate(preds_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    return scores, targets


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd()))
    cfg = OmegaConf.load(cfg_path)

    cfg.data.num_workers = args.num_workers
    cfg.data.pin_memory = bool(args.pin_memory)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)

    device = _resolve_device(args.device)
    model = model.eval().to(device)

    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    val_scores, val_targets = _collect_scores_targets(model, val_loader, device)

    val_metric_dict = eval_validation_set(
        val_scores, val_targets, use_inference_strategy=args.use_inference_strategy
    )
    val_per_class = val_metric_dict["val_none"]

    val_easy = float(np.mean(val_per_class[SUBSET2IDS["easy"]]))
    val_medium = float(np.mean(val_per_class[SUBSET2IDS["medium"]]))
    val_hard = float(np.mean(val_per_class[SUBSET2IDS["hard"]]))

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_scores, test_targets = _collect_scores_targets(model, test_loader, device)

    test_no_strategy = eval_validation_set(
        test_scores, test_targets, use_inference_strategy=False
    )
    test_use_strategy = eval_validation_set(
        test_scores, test_targets, use_inference_strategy=True
    )

    test_no_mean_f1 = float(
        (test_no_strategy["val_macro"] + test_no_strategy["val_micro"] + test_no_strategy["val_samples"]) / 3.0
    )
    test_use_mean_f1 = float(
        (test_use_strategy["val_macro"] + test_use_strategy["val_micro"] + test_use_strategy["val_samples"]) / 3.0
    )

    result = {
        "ckpt_path": str(ckpt_path),
        "device": str(device),
        "use_inference_strategy": bool(args.use_inference_strategy),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "val_micro": float(val_metric_dict["val_micro"]),
        "val_macro": float(val_metric_dict["val_macro"]),
        "val_samples": float(val_metric_dict["val_samples"]),
        "val_mAP": float(val_metric_dict["val_mAP"]),
        "threshold": float(val_metric_dict["threshold"]),
        "val_easy": val_easy,
        "val_medium": val_medium,
        "val_hard": val_hard,
        "val_easy_pct": val_easy * 100.0,
        "val_medium_pct": val_medium * 100.0,
        "val_hard_pct": val_hard * 100.0,
        "test_no_inference_strategy": {
            "micro": float(test_no_strategy["val_micro"]),
            "macro": float(test_no_strategy["val_macro"]),
            "samples": float(test_no_strategy["val_samples"]),
            "mAP": float(test_no_strategy["val_mAP"]),
            "mean_f1": test_no_mean_f1,
        },
        "test_use_inference_strategy": {
            "micro": float(test_use_strategy["val_micro"]),
            "macro": float(test_use_strategy["val_macro"]),
            "samples": float(test_use_strategy["val_samples"]),
            "mAP": float(test_use_strategy["val_mAP"]),
            "mean_f1": test_use_mean_f1,
        },
    }

    print(json.dumps(result, ensure_ascii=True, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
