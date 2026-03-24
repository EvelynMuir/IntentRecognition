#!/usr/bin/env python3
"""Analyze cls_mean_patch confidence groups against soft-label ambiguity.

Definitions used in this script:

1. Global threshold:
   Search a single probability threshold on the validation split using the
   repo's standard macro-F1 criterion.
2. Confidence groups:
   Use positive train labels whose soft label equals one of {1/3, 2/3, 1}.
3. Precision / recall / F1 per group:
   Evaluate a binary one-vs-zero task on label entries where:
   - positive: soft label == group value
   - negative: soft label == 0
   Other positive confidence levels are excluded from the subset so the metric
   answers "how separable is this confidence bucket from true negatives?"
4. Logit margin:
   For positive entries in the group, margin = raw_logit - logit(global_thr).
   Positive margin means the entry stays above the decision boundary.
5. Threshold sensitivity:
   Sweep thresholds on the group's one-vs-zero subset and report:
   - group-optimal threshold
   - F1 gain over the global validation threshold
   - local F1 variation inside global_thr +/- delta
   - positive flip rate when the threshold moves by +/- delta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.intentonomy_datamodule import IntentonomyDataset
from src.utils.metrics import eval_validation_set


CONFIDENCE_GROUPS = [
    ("0.333", 1.0 / 3.0),
    ("0.666", 2.0 / 3.0),
    ("1.000", 1.0),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cls_mean_patch confidence groups on Intentonomy soft labels."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing .hydra/config.yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional checkpoint path override.",
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
        default=4,
        help="Dataloader workers used during analysis.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable dataloader pin_memory.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading the checkpoint state_dict.",
    )
    parser.add_argument(
        "--threshold-delta",
        type=float,
        default=0.05,
        help="Local threshold perturbation used for sensitivity stats.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for JSON results.",
    )
    return parser.parse_args()


def _resolve_ckpt_path(run_dir: Path, ckpt_path: str | None) -> Path:
    if ckpt_path is not None:
        return Path(ckpt_path)

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        cfg_ckpt_path = cfg.get("ckpt_path")
        if cfg_ckpt_path:
            candidate = Path(str(cfg_ckpt_path))
            if candidate.exists():
                return candidate

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        epoch_ckpts = sorted(checkpoints_dir.glob("epoch_*.ckpt"))
        if epoch_ckpts:
            return epoch_ckpts[-1]
        candidate = checkpoints_dir / "last.ckpt"
        if candidate.exists():
            return candidate

    return run_dir / "checkpoints" / "last.ckpt"


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
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


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    error_rate = (fp + fn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "error_rate": float(error_rate),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _prob_to_logit(prob: float) -> float:
    clipped = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def _sweep_f1(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    thresholds = np.arange(0.05, 0.951, 0.01, dtype=np.float32)
    f1_values: List[float] = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(np.int32)
        f1_values.append(_binary_metrics(y_true, y_pred)["f1"])
    return {
        "thresholds": thresholds,
        "f1_values": np.asarray(f1_values, dtype=np.float32),
    }


def _collect_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    logits_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    soft_all: List[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]

            logits_cpu = logits.detach().float().cpu()
            scores_cpu = torch.sigmoid(logits_cpu)

            logits_all.append(logits_cpu.numpy())
            scores_all.append(scores_cpu.numpy())
            labels_all.append(batch["labels"].detach().float().cpu().numpy())
            soft_all.append(batch["soft_labels"].detach().float().cpu().numpy())

    return {
        "logits": np.concatenate(logits_all, axis=0),
        "scores": np.concatenate(scores_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "soft_labels": np.concatenate(soft_all, axis=0),
    }


def _make_train_eval_loader(cfg, datamodule, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    use_fixed_random_slot_perm = bool(
        OmegaConf.select(cfg, "data.use_fixed_random_slot_perm", default=False)
    )
    fixed_random_slot_perm_tokens = OmegaConf.select(
        cfg, "data.fixed_random_slot_perm_tokens", default=None
    )
    fixed_random_slot_perm_seed = int(
        OmegaConf.select(cfg, "data.fixed_random_slot_perm_seed", default=42)
    )

    dataset = IntentonomyDataset(
        annotation_file=os.path.join(cfg.data.annotation_dir, cfg.data.train_annotation),
        image_dir=cfg.data.image_dir,
        transform=datamodule.val_test_transform,
        binarize_softprob=cfg.data.binarize_softprob,
        use_fixed_random_slot_perm=use_fixed_random_slot_perm,
        fixed_random_slot_perm_tokens=fixed_random_slot_perm_tokens,
        fixed_random_slot_perm_seed=fixed_random_slot_perm_seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )


def _group_analysis(
    *,
    train_scores: np.ndarray,
    train_logits: np.ndarray,
    train_soft_labels: np.ndarray,
    global_threshold: float,
    threshold_delta: float,
    group_value: float,
) -> Dict[str, float | int | Dict[str, float]]:
    pos_mask = np.isclose(train_soft_labels, group_value, atol=1e-6)
    zero_mask = np.isclose(train_soft_labels, 0.0, atol=1e-8)
    subset_mask = pos_mask | zero_mask

    subset_scores = train_scores[subset_mask]
    subset_y_true = pos_mask[subset_mask].astype(np.int32)
    subset_y_pred = (subset_scores >= global_threshold).astype(np.int32)
    metrics = _binary_metrics(subset_y_true, subset_y_pred)

    positive_scores = train_scores[pos_mask]
    positive_logits = train_logits[pos_mask]

    global_logit_threshold = _prob_to_logit(global_threshold)
    positive_margin = positive_logits - global_logit_threshold

    sweep = _sweep_f1(subset_y_true, subset_scores)
    thresholds = sweep["thresholds"]
    f1_values = sweep["f1_values"]
    best_idx = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_values[best_idx])

    local_lower = max(0.05, global_threshold - threshold_delta)
    local_upper = min(0.95, global_threshold + threshold_delta)
    local_mask = (thresholds >= local_lower - 1e-8) & (thresholds <= local_upper + 1e-8)
    local_f1 = f1_values[local_mask]

    thr_up = min(0.95, global_threshold + threshold_delta)
    thr_down = max(0.05, global_threshold - threshold_delta)
    pred_at_global = positive_scores >= global_threshold
    pred_at_up = positive_scores >= thr_up
    pred_at_down = positive_scores >= thr_down

    return {
        "support_positive": int(pos_mask.sum()),
        "support_negative": int(zero_mask.sum()),
        "subset_size": int(subset_mask.sum()),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "error_rate": metrics["error_rate"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "miss_rate": float(1.0 - metrics["recall"]),
        "logit_margin": {
            "mean": float(np.mean(positive_margin)),
            "median": float(np.median(positive_margin)),
            "std": float(np.std(positive_margin)),
            "q25": float(np.quantile(positive_margin, 0.25)),
            "q75": float(np.quantile(positive_margin, 0.75)),
            "positive_ratio": float(np.mean(positive_margin > 0.0)),
        },
        "threshold_sensitivity": {
            "global_threshold": float(global_threshold),
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "f1_gain_to_best": float(best_f1 - metrics["f1"]),
            "local_f1_min": float(np.min(local_f1)),
            "local_f1_max": float(np.max(local_f1)),
            "local_f1_std": float(np.std(local_f1)),
            "positive_flip_rate_plus_delta": float(np.mean(pred_at_global != pred_at_up)),
            "positive_flip_rate_minus_delta": float(np.mean(pred_at_global != pred_at_down)),
        },
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    cfg = OmegaConf.load(cfg_path)
    cfg.data.num_workers = int(args.num_workers)
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
    train_eval_loader = _make_train_eval_loader(
        cfg=cfg,
        datamodule=datamodule,
        batch_size=datamodule.batch_size_per_device,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )

    val_outputs = _collect_outputs(model, val_loader, device)
    val_metrics = eval_validation_set(
        val_outputs["scores"],
        val_outputs["labels"],
        use_inference_strategy=False,
    )
    global_threshold = float(val_metrics["threshold"])

    train_outputs = _collect_outputs(model, train_eval_loader, device)

    unique_soft, unique_counts = np.unique(np.round(train_outputs["soft_labels"], 6), return_counts=True)
    soft_label_hist = {
        f"{float(value):.6f}": int(count)
        for value, count in zip(unique_soft, unique_counts, strict=True)
    }

    groups = {}
    for group_name, group_value in CONFIDENCE_GROUPS:
        groups[group_name] = _group_analysis(
            train_scores=train_outputs["scores"],
            train_logits=train_outputs["logits"],
            train_soft_labels=train_outputs["soft_labels"],
            global_threshold=global_threshold,
            threshold_delta=float(args.threshold_delta),
            group_value=group_value,
        )

    result = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "device": str(device),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "definitions": {
            "metric_subset": "Per-group metrics use soft==group as positives and soft==0 as negatives; other positive soft labels are excluded.",
            "logit_margin": "For positive entries only: raw_logit - logit(global_validation_threshold).",
            "threshold_sensitivity": "Threshold sweep over the one-vs-zero subset plus positive flip rates under +/- threshold_delta perturbations.",
            "train_split_note": "Soft labels exist only on the train split in this repo; val/test annotations are hard labels.",
        },
        "val_metrics": {
            "macro_f1": float(val_metrics["val_macro"]),
            "micro_f1": float(val_metrics["val_micro"]),
            "samples_f1": float(val_metrics["val_samples"]),
            "mAP": float(val_metrics["val_mAP"]),
            "threshold": global_threshold,
        },
        "train_soft_label_histogram": soft_label_hist,
        "groups": groups,
    }

    print(json.dumps(result, ensure_ascii=True, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
