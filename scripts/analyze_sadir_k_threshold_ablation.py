#!/usr/bin/env python3
"""Evaluate full-SADIR runs under different thresholding settings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import clip  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_distillation_slrc import (
    DEFAULT_PROMPT_TEMPLATE,
    INTENTONOMY_LEXICAL_PHRASES,
    INTENTONOMY_DESCRIPTIONS,
    ResidualStudent,
    _apply_slr,
    _build_text_pools,
    _encode_text_pool,
    _load_cache_bundle,
    _load_class_names,
    _normalize_scores_per_sample,
    _predict_baseline_logits,
    _predict_residual_student,
    _resolve_device,
    _text_logits_from_features,
)
from scripts.analyze_privileged_distillation import (
    StudentMLP,
    compute_difficulty_scores,
    compute_mAP,
    compute_f1,
    eval_validation_set,
    search_classwise_thresholds,
)


DEFAULT_CACHE_DIR = PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
DEFAULT_UTD_RUN_DIR = PROJECT_ROOT / "logs" / "analysis" / "privileged_distillation_text_teacher_seedfix_20260316"
DEFAULT_SADIR_RUN_DIR = PROJECT_ROOT / "logs" / "analysis" / "distillation_slrc_lcs_rebuild_20260327"
DEFAULT_ANNOTATION_FILE = PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
DEFAULT_GEMINI_FILE = PROJECT_ROOT.parent / "Intentonomy" / "data" / "intent_description_gemini.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a full SADIR run with validation-based threshold search.")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--utd-run-dir", type=str, default=str(DEFAULT_UTD_RUN_DIR))
    parser.add_argument("--sadir-run-dir", type=str, default=str(DEFAULT_SADIR_RUN_DIR))
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION_FILE))
    parser.add_argument("--gemini-file", type=str, default=str(DEFAULT_GEMINI_FILE))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--slr-alpha", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def _build_student_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    image_dim: int,
    num_classes: int,
) -> StudentMLP:
    model = StudentMLP(
        image_dim=image_dim,
        hidden_dim=768,
        num_classes=num_classes,
        dropout=0.1,
        feature_proj_dim=256,
    )
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def _evaluate_split(
    scores: np.ndarray,
    targets: np.ndarray,
    thresholding: str,
    val_scores: np.ndarray,
    val_targets: np.ndarray,
) -> Dict[str, float]:
    if thresholding == "global":
        val_metrics = eval_validation_set(val_scores, val_targets, use_inference_strategy=False)
        threshold = float(val_metrics["threshold"])
        micro, samples, macro, per_class = compute_f1(
            targets,
            scores,
            threshold=threshold,
            use_inference_strategy=False,
        )
    elif thresholding == "classwise":
        class_thresholds = search_classwise_thresholds(val_scores, val_targets)
        micro, samples, macro, per_class = compute_f1(
            targets,
            scores,
            threshold=0.5,
            use_inference_strategy=False,
            class_thresholds=class_thresholds,
        )
    else:
        raise ValueError(f"Unsupported thresholding: {thresholding}")

    difficulty = compute_difficulty_scores(per_class)
    return {
        "macro": float(macro) * 100.0,
        "micro": float(micro) * 100.0,
        "samples": float(samples) * 100.0,
        "avg": float((macro + micro + samples) / 3.0) * 100.0,
        "hard": float(difficulty["hard"]) * 100.0,
        "mAP": float(compute_mAP(scores, targets)),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    train_clip = _load_cache_bundle(Path(args.cache_dir) / "train_clip.npz")
    val_clip = _load_cache_bundle(Path(args.cache_dir) / "val_clip.npz")
    test_clip = _load_cache_bundle(Path(args.cache_dir) / "test_clip.npz")
    class_names = _load_class_names(Path(args.annotation_file))
    num_classes = int(np.asarray(train_clip["labels"], dtype=np.float32).shape[1])
    image_dim = int(np.asarray(train_clip["features"], dtype=np.float32).shape[1])
    sadir_summary = json.loads((Path(args.sadir_run_dir) / "summary.json").read_text(encoding="utf-8"))
    topk_value = int(sadir_summary["config"]["topk"])

    baseline_state = torch.load(Path(args.utd_run_dir) / "baseline_best.pt", map_location="cpu", weights_only=True)
    sadir_state = torch.load(Path(args.sadir_run_dir) / "slr_c_residual_dynamic_kd_best.pt", map_location="cpu", weights_only=True)

    baseline_model = _build_student_from_state_dict(baseline_state, image_dim=image_dim, num_classes=num_classes).to(device)
    residual_model = ResidualStudent(
        image_dim=image_dim,
        hidden_dim=int(sadir_state["net.0.weight"].shape[0]),
        num_classes=num_classes,
        dropout=0.1,
    )
    residual_model.load_state_dict(sadir_state, strict=False)
    residual_model = residual_model.to(device).eval()

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    text_pools = _build_text_pools(class_names, Path(args.gemini_file))
    lexical_embeddings = _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True)
    canonical_embeddings = _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True)
    scenario_embeddings = _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False)

    train_features = np.asarray(train_clip["features"], dtype=np.float32)
    val_features = np.asarray(val_clip["features"], dtype=np.float32)
    test_features = np.asarray(test_clip["features"], dtype=np.float32)
    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    val_labels = np.asarray(val_clip["labels"], dtype=np.float32)
    test_labels = np.asarray(test_clip["labels"], dtype=np.float32)

    train_base_logits = _predict_baseline_logits(baseline_model, train_features, device, int(args.batch_size))
    val_base_logits = _predict_baseline_logits(baseline_model, val_features, device, int(args.batch_size))
    test_base_logits = _predict_baseline_logits(baseline_model, test_features, device, int(args.batch_size))

    train_prior = (
        _normalize_scores_per_sample(_text_logits_from_features(train_features, lexical_embeddings, logit_scale))
        + _normalize_scores_per_sample(_text_logits_from_features(train_features, canonical_embeddings, logit_scale))
        + _normalize_scores_per_sample(_text_logits_from_features(train_features, scenario_embeddings, logit_scale))
    ) / 3.0
    val_prior = (
        _normalize_scores_per_sample(_text_logits_from_features(val_features, lexical_embeddings, logit_scale))
        + _normalize_scores_per_sample(_text_logits_from_features(val_features, canonical_embeddings, logit_scale))
        + _normalize_scores_per_sample(_text_logits_from_features(val_features, scenario_embeddings, logit_scale))
    ) / 3.0
    test_prior = (
        _normalize_scores_per_sample(_text_logits_from_features(test_features, lexical_embeddings, logit_scale))
        + _normalize_scores_per_sample(_text_logits_from_features(test_features, canonical_embeddings, logit_scale))
        + _normalize_scores_per_sample(_text_logits_from_features(test_features, scenario_embeddings, logit_scale))
    ) / 3.0

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    train_slr_logits = _apply_slr(train_base_logits, train_prior, topk=topk_value, alpha=float(args.slr_alpha))
    val_slr_logits = _apply_slr(val_base_logits, val_prior, topk=topk_value, alpha=float(args.slr_alpha))
    test_slr_logits = _apply_slr(test_base_logits, test_prior, topk=topk_value, alpha=float(args.slr_alpha))

    train_scores = _predict_residual_student(
        residual_model,
        image_features=train_features,
        slr_logits=train_slr_logits,
        device=device,
        batch_size=int(args.batch_size),
    )
    val_scores = _predict_residual_student(
        residual_model,
        image_features=val_features,
        slr_logits=val_slr_logits,
        device=device,
        batch_size=int(args.batch_size),
    )
    test_scores = _predict_residual_student(
        residual_model,
        image_features=test_features,
        slr_logits=test_slr_logits,
        device=device,
        batch_size=int(args.batch_size),
    )

    for thresholding in ["global", "classwise"]:
        val_metrics = _evaluate_split(
            scores=val_scores,
            targets=val_labels,
            thresholding=thresholding,
            val_scores=val_scores,
            val_targets=val_labels,
        )
        test_metrics = _evaluate_split(
            scores=test_scores,
            targets=test_labels,
            thresholding=thresholding,
            val_scores=val_scores,
            val_targets=val_labels,
        )
        for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
            row = {
                "split": split_name,
                "topk": "all" if int(topk_value) == num_classes else int(topk_value),
                "thresholding": thresholding,
                **{key: round(float(value), 4) for key, value in metrics.items()},
            }
            rows.append(row)
        summary_rows.append(
            {
                "topk": "all" if int(topk_value) == num_classes else int(topk_value),
                "thresholding": thresholding,
                "val_macro": round(val_metrics["macro"], 2),
                "val_micro": round(val_metrics["micro"], 2),
                "val_samples": round(val_metrics["samples"], 2),
                "val_avg": round(val_metrics["avg"], 2),
                "val_hard": round(val_metrics["hard"], 2),
                "test_macro": round(test_metrics["macro"], 2),
                "test_micro": round(test_metrics["micro"], 2),
                "test_samples": round(test_metrics["samples"], 2),
                "test_avg": round(test_metrics["avg"], 2),
                "test_hard": round(test_metrics["hard"], 2),
            }
        )

    _write_csv(output_dir / "k_threshold_ablation_detailed.csv", rows)
    _write_csv(output_dir / "k_threshold_ablation_summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "cache_dir": str(args.cache_dir),
                    "utd_run_dir": str(args.utd_run_dir),
                    "sadir_run_dir": str(args.sadir_run_dir),
                    "topk": "all" if int(topk_value) == num_classes else int(topk_value),
                    "slr_alpha": float(args.slr_alpha),
                },
                "rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SADIR-K-Threshold] saved to {output_dir}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
