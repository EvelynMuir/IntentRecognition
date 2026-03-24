#!/usr/bin/env python3
"""Train a CLIP-feature baseline and report per-class AP / mAP."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_privileged_distillation import (
    StudentDataset,
    StudentMLP,
    _evaluate_score_bundle,
    _json_ready,
    _load_cache_bundle,
    _write_csv,
    _predict_student,
    _set_component_seed,
    _set_seed,
    _train_student,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP feature baseline on cached features.")
    parser.add_argument("--reuse-cache-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--description-file", type=str, default=str(PROJECT_ROOT.parent / "Emotic" / "emotion_description_gemini.json"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--student-hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-proj-dim", type=int, default=256)
    parser.add_argument("--feature-distill-mode", type=str, default="none")
    parser.add_argument("--feature-distill-weight", type=float, default=0.0)
    parser.add_argument("--feature-distill-temperature", type=float, default=0.1)
    parser.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    parser.add_argument("--standard-kd-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--dynamic-kd-variant", type=str, default="sample_inverse")
    parser.add_argument("--dynamic-gate-alpha", type=float, default=0.3)
    parser.add_argument("--dynamic-gate-beta", type=float, default=0.7)
    parser.add_argument("--entropy-gate-lambda", type=float, default=1.0)
    return parser.parse_args()


def _resolve_output_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_emotic_clip_baseline"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))

    cache_dir = Path(args.reuse_cache_dir)
    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")

    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    train_soft_labels = np.asarray(train_clip["soft_labels"], dtype=np.float32)
    dataset = StudentDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        labels=train_labels,
        agreement=np.ones(train_labels.shape[0], dtype=np.float32),
        soft_labels=train_soft_labels,
        teacher_probs=np.zeros_like(train_labels, dtype=np.float32),
    )

    _set_component_seed(int(args.seed), offset=100)
    model = StudentMLP(
        image_dim=int(dataset.image_features.shape[1]),
        hidden_dim=int(args.student_hidden_dim),
        num_classes=int(dataset.labels.shape[1]),
        dropout=float(args.dropout),
        feature_proj_dim=int(args.feature_proj_dim),
    ).to(device)

    result = _train_student(
        mode="baseline",
        model=model,
        train_dataset=dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_targets=np.asarray(val_clip["labels"], dtype=np.float32),
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_targets=np.asarray(test_clip["labels"], dtype=np.float32),
        device=device,
        args=args,
    )

    per_class_ap_rows = []
    class_names = []
    description_file = Path(args.description_file)
    if description_file.exists():
        payload = json.loads(description_file.read_text(encoding="utf-8"))
        class_names = [str(item["emotion_name"]) for item in payload.get("emotions", [])]
    per_class_ap = np.asarray(result["bundle"]["classwise"]["test"]["per_class_ap"], dtype=np.float32)
    for idx, ap in enumerate(per_class_ap.tolist()):
        per_class_ap_rows.append(
            {"class_id": idx, "class_name": class_names[idx] if idx < len(class_names) else f"class_{idx}", "AP": float(ap)}
        )
    _write_csv(output_dir / "per_class_ap.csv", per_class_ap_rows)

    train_scores = _predict_student(
        model=model,
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )
    val_scores = _predict_student(
        model=model,
        image_features=np.asarray(val_clip["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )
    test_scores = _predict_student(
        model=model,
        image_features=np.asarray(test_clip["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )

    train_eval = _evaluate_score_bundle(
        val_scores=val_scores,
        val_targets=np.asarray(val_clip["labels"], dtype=np.float32),
        test_scores=train_scores,
        test_targets=np.asarray(train_clip["labels"], dtype=np.float32),
    )

    torch.save(result["state_dict"], output_dir / "baseline_best.pt")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "baseline": _json_ready({k: v for k, v in result.items() if k != "state_dict"}),
                "train_as_test_eval": _json_ready(train_eval),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[CLIPBaseline] "
        f"test_mAP={result['bundle']['classwise']['test']['mAP']:.2f} "
        f"test_macro={result['bundle']['classwise']['test']['macro']*100.0:.2f}"
    )


if __name__ == "__main__":
    main()
