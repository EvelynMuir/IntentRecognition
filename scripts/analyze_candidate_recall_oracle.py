#!/usr/bin/env python3
"""Analyze top-k candidate recall and oracle bounds for baseline and SLR."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_text_prior_boundary import (
    DEFAULT_BASELINE_CKPT,
    _normalize_source_name,
    _build_dataset,
    _build_text_pools,
    _collect_clip_features,
    _collect_model_outputs,
    _encode_text_pool,
    _load_class_names,
    _normalize_state_dict_keys,
    _resolve_ckpt_path,
    _resolve_device,
    _resolve_gemini_file,
    _resolve_run_dir,
    _text_logits_from_features,
)
from src.utils.metrics import eval_validation_set
from src.utils.text_prior_analysis import apply_topk_rerank_fusion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze candidate recall / oracle upper bounds for baseline and SLR."
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=str(DEFAULT_BASELINE_CKPT) if DEFAULT_BASELINE_CKPT.exists() else None,
    )
    parser.add_argument("--gemini-file", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--topk-list", type=str, default="1,3,5,10")
    parser.add_argument(
        "--slr-source",
        type=str,
        default="scenario",
        choices=["lexical", "canonical", "scenario", "discriminative"],
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON artifacts.",
    )
    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _build_topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    num_classes = scores.shape[1]
    k = max(1, min(int(k), num_classes))
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(scores, dtype=np.int32)
    row_idx = np.arange(scores.shape[0])[:, None]
    mask[row_idx, topk_idx] = 1
    return mask


def _candidate_recall_stats(
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: List[int],
) -> Dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int32)
    positive_counts = np.maximum(targets.sum(axis=1), 1)
    results: Dict[str, Any] = {}

    for k in k_values:
        mask = _build_topk_mask(scores, k)
        covered_targets = targets * mask
        label_recall = float(covered_targets.sum() / max(targets.sum(), 1))
        sample_any = float(np.mean((covered_targets.sum(axis=1) > 0).astype(np.float32)))
        sample_all = float(np.mean((covered_targets.sum(axis=1) == targets.sum(axis=1)).astype(np.float32)))
        sample_coverage = float(np.mean(covered_targets.sum(axis=1) / positive_counts))
        results[f"top{k}"] = {
            "label_recall": label_recall,
            "sample_any_recall": sample_any,
            "sample_all_recall": sample_all,
            "mean_positive_coverage": sample_coverage,
        }
    return results


def _oracle_multilabel_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    mask = _build_topk_mask(scores, k)
    oracle_pred = (targets * mask).astype(np.float32)

    from sklearn.metrics import f1_score

    macro = float(f1_score(targets, oracle_pred, average="macro", zero_division=0))
    micro = float(f1_score(targets, oracle_pred, average="micro", zero_division=0))
    samples = float(f1_score(targets, oracle_pred, average="samples", zero_division=0))
    return {
        "macro": macro,
        "micro": micro,
        "samples": samples,
        "coverage_label_recall": float(oracle_pred.sum() / max(targets.sum(), 1)),
    }


def _fn_recoverability(
    scores: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    mask = _build_topk_mask(scores, k)
    fn_mask = (targets == 1) & (predictions == 0)
    recoverable_fn = fn_mask & (mask == 1)
    unrecoverable_fn = fn_mask & (mask == 0)
    total_fn = int(fn_mask.sum())
    return {
        "total_false_negatives": total_fn,
        "recoverable_false_negatives": int(recoverable_fn.sum()),
        "unrecoverable_false_negatives": int(unrecoverable_fn.sum()),
        "recoverable_ratio": float(recoverable_fn.sum() / total_fn) if total_fn > 0 else 0.0,
    }


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir, args.ckpt_path)
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)
    cfg_path = run_dir / ".hydra" / "config.yaml"
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
    state_dict = _normalize_state_dict_keys(checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)

    device = _resolve_device(args.device)
    model = model.eval().to(device)

    clip_model = getattr(getattr(model, "net", model), "clip_model", None)
    clip_preprocess = getattr(getattr(model, "net", model), "clip_preprocess", None)
    if clip_model is None or clip_preprocess is None:
        import clip

        clip_model_name = OmegaConf.select(cfg, "model.net.clip_model_name", default="ViT-L/14")
        clip_model, clip_preprocess = clip.load(str(clip_model_name), device=device)
    else:
        clip_model = clip_model.eval().to(device)

    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    gemini_file = _resolve_gemini_file(cfg, args.gemini_file)
    class_names = _load_class_names(Path(str(cfg.data.annotation_dir)) / str(cfg.data.val_annotation))

    batch_size = int(getattr(cfg.data, "batch_size", datamodule.batch_size_per_device))
    _, val_loader_base, _ = _build_dataset(
        cfg, datamodule, "val", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, test_loader_base, _ = _build_dataset(
        cfg, datamodule, "test", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, val_loader_clip, _ = _build_dataset(
        cfg, datamodule, "val", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )
    _, test_loader_clip, _ = _build_dataset(
        cfg, datamodule, "test", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )

    val_base = _collect_model_outputs(model, val_loader_base, device, max_samples=args.max_samples)
    test_base = _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples)
    val_clip = _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples)
    test_clip = _collect_clip_features(clip_model, test_loader_clip, device, max_samples=args.max_samples)

    text_pools = _build_text_pools(class_names, gemini_file)
    text_embeddings = {
        "lexical": _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True),
        "canonical": _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True),
        "scenario": _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False),
        "discriminative": _encode_text_pool(clip_model, text_pools["discriminative"], wrap_prompt=False),
    }
    slr_source = _normalize_source_name(args.slr_source)
    val_text_logits = _text_logits_from_features(
        val_clip["features"],
        text_embeddings[slr_source],
        clip_logit_scale,
    )
    test_text_logits = _text_logits_from_features(
        test_clip["features"],
        text_embeddings[slr_source],
        clip_logit_scale,
    )
    slr_val_logits = apply_topk_rerank_fusion(
        val_base["logits"],
        val_text_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_test_logits = apply_topk_rerank_fusion(
        test_base["logits"],
        test_text_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_val_scores = 1.0 / (1.0 + np.exp(-slr_val_logits))
    slr_test_scores = 1.0 / (1.0 + np.exp(-slr_test_logits))

    baseline_val_metrics = eval_validation_set(
        val_base["scores"],
        val_base["labels"],
        use_inference_strategy=False,
    )
    slr_val_metrics = eval_validation_set(
        slr_val_scores,
        val_base["labels"],
        use_inference_strategy=False,
    )

    baseline_predictions = (test_base["scores"] > float(baseline_val_metrics["threshold"])).astype(np.int32)
    slr_predictions = (slr_test_scores > float(slr_val_metrics["threshold"])).astype(np.int32)
    slr_classwise_thresholds = np.asarray(
        eval_validation_set(
            slr_val_scores,
            val_base["labels"],
            use_inference_strategy=False,
        )["threshold"],
        dtype=np.float32,
    )
    # true class-wise thresholds
    from src.utils.decision_rule_calibration import search_classwise_thresholds
    slr_classwise_thresholds = search_classwise_thresholds(slr_val_scores, val_base["labels"])
    from src.utils.metrics import compute_f1

    slr_classwise_predictions = (
        slr_test_scores > slr_classwise_thresholds[None, :]
    ).astype(np.int32)

    k_values = _parse_int_list(args.topk_list)
    baseline_candidate_recall = _candidate_recall_stats(test_base["logits"], test_base["labels"], k_values)
    oracle_stats = {
        f"top{k}": _oracle_multilabel_metrics(test_base["logits"], test_base["labels"], k)
        for k in k_values
    }
    fn_recoverability = {
        "baseline_global": _fn_recoverability(
            test_base["logits"], test_base["labels"], baseline_predictions, int(args.topk)
        ),
        "slr_global": _fn_recoverability(
            test_base["logits"], test_base["labels"], slr_predictions, int(args.topk)
        ),
        "slr_classwise": _fn_recoverability(
            test_base["logits"], test_base["labels"], slr_classwise_predictions, int(args.topk)
        ),
    }

    improved_by_slr = ((slr_predictions == 1) & (baseline_predictions == 0) & (test_base["labels"] == 1)).sum()
    improved_by_slr_classwise = (
        (slr_classwise_predictions == 1) & (baseline_predictions == 0) & (test_base["labels"] == 1)
    ).sum()

    result = {
        "metadata": {
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "device": str(device),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "slr_source": slr_source,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "candidate_recall": baseline_candidate_recall,
        "oracle_upper_bound": oracle_stats,
        "false_negative_recoverability_at_topk": fn_recoverability,
        "recovered_positive_labels": {
            "slr_global_over_baseline": int(improved_by_slr),
            "slr_classwise_over_baseline": int(improved_by_slr_classwise),
        },
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_candidate_recall_oracle"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
