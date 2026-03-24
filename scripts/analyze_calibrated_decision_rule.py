#!/usr/bin/env python3
"""Analyze calibrated decision rules for baseline and SLR-v0."""

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
    _metrics_for_json,
    _normalize_state_dict_keys,
    _resolve_ckpt_path,
    _resolve_device,
    _resolve_gemini_file,
    _resolve_run_dir,
    _text_logits_from_features,
)
from src.utils.decision_rule_calibration import (
    build_head_medium_tail_groups,
    build_prior_benefit_groups,
    search_classwise_thresholds,
    search_groupwise_thresholds,
)
from src.utils.metrics import eval_validation_set
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    compute_class_gains,
    evaluate_fixed_threshold,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated decision rule experiments for baseline and SLR-v0."
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
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument(
        "--slr-source",
        type=str,
        default="scenario",
        choices=["lexical", "canonical", "scenario", "discriminative"],
    )
    parser.add_argument(
        "--source-ensemble",
        type=str,
        default="lexical,canonical,lexical_plus_canonical",
        help="Comma-separated sources for step-3 ensemble experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _evaluate_with_class_thresholds(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_scores: np.ndarray,
    test_targets: np.ndarray,
    class_thresholds: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    val_metrics = eval_validation_set(
        val_scores,
        val_targets,
        use_inference_strategy=False,
        class_thresholds=class_thresholds,
    )
    test_metrics = evaluate_fixed_threshold(
        test_scores,
        test_targets,
        threshold=0.5,
        use_inference_strategy=False,
    )
    # Recompute test with explicit class thresholds.
    from src.utils.metrics import compute_f1, compute_mAP
    from src.utils.metrics import compute_difficulty_scores

    micro, samples, macro, per_class = compute_f1(
        test_targets,
        test_scores,
        threshold=0.5,
        use_inference_strategy=False,
        class_thresholds=class_thresholds,
    )
    difficulty = compute_difficulty_scores(per_class)
    test_metrics = {
        "micro": float(micro),
        "samples": float(samples),
        "macro": float(macro),
        "per_class_f1": per_class.astype(np.float32),
        "mAP": float(compute_mAP(test_scores, test_targets)),
        "threshold": float(np.mean(class_thresholds)),
        "easy": difficulty["easy"],
        "medium": difficulty["medium"],
        "hard": difficulty["hard"],
    }
    return {
        "val": {
            "micro": float(val_metrics["val_micro"]),
            "samples": float(val_metrics["val_samples"]),
            "macro": float(val_metrics["val_macro"]),
            "per_class_f1": val_metrics["val_none"].astype(np.float32),
            "mAP": float(val_metrics["val_mAP"]),
            "threshold": float(val_metrics["threshold"]),
            "easy": float(val_metrics["val_easy"]),
            "medium": float(val_metrics["val_medium"]),
            "hard": float(val_metrics["val_hard"]),
            "class_thresholds": class_thresholds.astype(np.float32),
        },
        "test": test_metrics,
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
    _, train_loader_base, _ = _build_dataset(
        cfg, datamodule, "train", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, val_loader_base, _ = _build_dataset(
        cfg, datamodule, "val", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, test_loader_base, _ = _build_dataset(
        cfg, datamodule, "test", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, train_loader_clip, _ = _build_dataset(
        cfg, datamodule, "train", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )
    _, val_loader_clip, _ = _build_dataset(
        cfg, datamodule, "val", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )
    _, test_loader_clip, _ = _build_dataset(
        cfg, datamodule, "test", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )

    train_base = _collect_model_outputs(model, train_loader_base, device, max_samples=args.max_samples)
    val_base = _collect_model_outputs(model, val_loader_base, device, max_samples=args.max_samples)
    test_base = _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples)
    train_clip = _collect_clip_features(clip_model, train_loader_clip, device, max_samples=args.max_samples)
    val_clip = _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples)
    test_clip = _collect_clip_features(clip_model, test_loader_clip, device, max_samples=args.max_samples)

    text_pools = _build_text_pools(class_names, gemini_file)
    text_embeddings = {
        "lexical": _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True),
        "canonical": _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True),
        "scenario": _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False),
        "discriminative": _encode_text_pool(clip_model, text_pools["discriminative"], wrap_prompt=False),
        "lexical_plus_canonical": _encode_text_pool(
            clip_model, text_pools["lexical_plus_canonical"], wrap_prompt=False
        ),
    }

    text_logits = {
        source_name: {
            "val": _text_logits_from_features(val_clip["features"], embeddings, clip_logit_scale),
            "test": _text_logits_from_features(test_clip["features"], embeddings, clip_logit_scale),
        }
        for source_name, embeddings in text_embeddings.items()
    }
    slr_source = _normalize_source_name(args.slr_source)

    baseline_global = eval_validation_set(
        val_base["scores"],
        val_base["labels"],
        use_inference_strategy=False,
    )
    baseline_global_metrics = {
        "val": {
            "micro": float(baseline_global["val_micro"]),
            "samples": float(baseline_global["val_samples"]),
            "macro": float(baseline_global["val_macro"]),
            "per_class_f1": baseline_global["val_none"].astype(np.float32),
            "mAP": float(baseline_global["val_mAP"]),
            "threshold": float(baseline_global["threshold"]),
            "easy": float(baseline_global["val_easy"]),
            "medium": float(baseline_global["val_medium"]),
            "hard": float(baseline_global["val_hard"]),
        },
        "test": evaluate_fixed_threshold(
            test_base["scores"],
            test_base["labels"],
            threshold=float(baseline_global["threshold"]),
            use_inference_strategy=False,
        ),
    }

    baseline_class_thresholds = search_classwise_thresholds(
        val_base["scores"],
        val_base["labels"],
    )
    baseline_classwise_metrics = _evaluate_with_class_thresholds(
        val_base["scores"],
        val_base["labels"],
        test_base["scores"],
        test_base["labels"],
        baseline_class_thresholds,
    )

    train_positive_counts = train_base["labels"].sum(axis=0)
    frequency_groups = build_head_medium_tail_groups(train_positive_counts)
    baseline_frequency_thresholds, baseline_frequency_group_thresholds = search_groupwise_thresholds(
        val_base["scores"],
        val_base["labels"],
        frequency_groups,
    )
    baseline_frequency_metrics = _evaluate_with_class_thresholds(
        val_base["scores"],
        val_base["labels"],
        test_base["scores"],
        test_base["labels"],
        baseline_frequency_thresholds,
    )

    slr_val_logits = apply_topk_rerank_fusion(
        val_base["logits"],
        text_logits[slr_source]["val"],
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_test_logits = apply_topk_rerank_fusion(
        test_base["logits"],
        text_logits[slr_source]["test"],
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_val_scores = 1.0 / (1.0 + np.exp(-slr_val_logits))
    slr_test_scores = 1.0 / (1.0 + np.exp(-slr_test_logits))

    slr_original_threshold_metrics = {
        "val": evaluate_fixed_threshold(
            slr_val_scores,
            val_base["labels"],
            threshold=float(baseline_global["threshold"]),
            use_inference_strategy=False,
        ),
        "test": evaluate_fixed_threshold(
            slr_test_scores,
            test_base["labels"],
            threshold=float(baseline_global["threshold"]),
            use_inference_strategy=False,
        ),
    }
    slr_retuned_global = eval_validation_set(
        slr_val_scores,
        val_base["labels"],
        use_inference_strategy=False,
    )
    slr_retuned_global_metrics = {
        "val": {
            "micro": float(slr_retuned_global["val_micro"]),
            "samples": float(slr_retuned_global["val_samples"]),
            "macro": float(slr_retuned_global["val_macro"]),
            "per_class_f1": slr_retuned_global["val_none"].astype(np.float32),
            "mAP": float(slr_retuned_global["val_mAP"]),
            "threshold": float(slr_retuned_global["threshold"]),
            "easy": float(slr_retuned_global["val_easy"]),
            "medium": float(slr_retuned_global["val_medium"]),
            "hard": float(slr_retuned_global["val_hard"]),
        },
        "test": evaluate_fixed_threshold(
            slr_test_scores,
            test_base["labels"],
            threshold=float(slr_retuned_global["threshold"]),
            use_inference_strategy=False,
        ),
    }

    slr_class_thresholds = search_classwise_thresholds(
        slr_val_scores,
        val_base["labels"],
    )
    slr_classwise_metrics = _evaluate_with_class_thresholds(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        slr_class_thresholds,
    )

    slr_val_class_gains = compute_class_gains(
        baseline_global_metrics["val"]["per_class_f1"],
        slr_retuned_global_metrics["val"]["per_class_f1"],
    )
    semantic_groups = build_prior_benefit_groups(slr_val_class_gains)
    slr_semantic_thresholds, slr_semantic_group_thresholds = search_groupwise_thresholds(
        slr_val_scores,
        val_base["labels"],
        semantic_groups,
    )
    slr_semantic_metrics = _evaluate_with_class_thresholds(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        slr_semantic_thresholds,
    )

    slr_frequency_thresholds, slr_frequency_group_thresholds = search_groupwise_thresholds(
        slr_val_scores,
        val_base["labels"],
        frequency_groups,
    )
    slr_frequency_metrics = _evaluate_with_class_thresholds(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        slr_frequency_thresholds,
    )

    source_names = [_normalize_source_name(x) for x in _parse_str_list(args.source_ensemble)]
    source_ensemble_results = {}
    for source_name in source_names:
        val_source_logits = apply_topk_rerank_fusion(
            val_base["logits"],
            text_logits[source_name]["val"],
            topk=int(args.topk),
            alpha=float(args.alpha),
            mode="add_norm",
        )
        test_source_logits = apply_topk_rerank_fusion(
            test_base["logits"],
            text_logits[source_name]["test"],
            topk=int(args.topk),
            alpha=float(args.alpha),
            mode="add_norm",
        )
        val_source_scores = 1.0 / (1.0 + np.exp(-val_source_logits))
        test_source_scores = 1.0 / (1.0 + np.exp(-test_source_logits))
        global_metrics = eval_validation_set(
            val_source_scores,
            val_base["labels"],
            use_inference_strategy=False,
        )
        class_thresholds = search_classwise_thresholds(val_source_scores, val_base["labels"])
        class_metrics = _evaluate_with_class_thresholds(
            val_source_scores,
            val_base["labels"],
            test_source_scores,
            test_base["labels"],
            class_thresholds,
        )
        source_ensemble_results[source_name] = {
            "global": {
                "val": {
                    "micro": float(global_metrics["val_micro"]),
                    "samples": float(global_metrics["val_samples"]),
                    "macro": float(global_metrics["val_macro"]),
                    "per_class_f1": global_metrics["val_none"].astype(np.float32),
                    "mAP": float(global_metrics["val_mAP"]),
                    "threshold": float(global_metrics["threshold"]),
                    "easy": float(global_metrics["val_easy"]),
                    "medium": float(global_metrics["val_medium"]),
                    "hard": float(global_metrics["val_hard"]),
                },
                "test": evaluate_fixed_threshold(
                    test_source_scores,
                    test_base["labels"],
                    threshold=float(global_metrics["threshold"]),
                    use_inference_strategy=False,
                ),
            },
            "classwise": class_metrics,
        }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_calibrated_decision_rule"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "gemini_file": str(gemini_file),
            "device": str(device),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "slr_source": slr_source,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
            "source_ensemble": source_names,
            "output_dir": str(output_dir),
        },
        "groups": {
            "semantic_prior_gain_groups": semantic_groups,
            "frequency_groups": frequency_groups,
            "baseline_frequency_group_thresholds": baseline_frequency_group_thresholds,
            "slr_semantic_group_thresholds": slr_semantic_group_thresholds,
            "slr_frequency_group_thresholds": slr_frequency_group_thresholds,
        },
        "baseline": {
            "global": {
                "val": _metrics_for_json(baseline_global_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(baseline_global_metrics["test"], include_per_class=True),
            },
            "classwise": {
                "val": _metrics_for_json(baseline_classwise_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(baseline_classwise_metrics["test"], include_per_class=True),
            },
            "group_frequency": {
                "val": _metrics_for_json(baseline_frequency_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(baseline_frequency_metrics["test"], include_per_class=True),
            },
        },
        "slr_v0": {
            "original_threshold": {
                "val": _metrics_for_json(slr_original_threshold_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(slr_original_threshold_metrics["test"], include_per_class=True),
            },
            "retuned_global": {
                "val": _metrics_for_json(slr_retuned_global_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(slr_retuned_global_metrics["test"], include_per_class=True),
            },
            "classwise": {
                "val": _metrics_for_json(slr_classwise_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(slr_classwise_metrics["test"], include_per_class=True),
            },
            "group_semantic_prior_gain": {
                "val": _metrics_for_json(slr_semantic_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(slr_semantic_metrics["test"], include_per_class=True),
            },
            "group_frequency": {
                "val": _metrics_for_json(slr_frequency_metrics["val"], include_per_class=True),
                "test": _metrics_for_json(slr_frequency_metrics["test"], include_per_class=True),
            },
        },
        "source_ensemble": {
            source_name: {
                "global": {
                    "val": _metrics_for_json(item["global"]["val"], include_per_class=True),
                    "test": _metrics_for_json(item["global"]["test"], include_per_class=True),
                },
                "classwise": {
                    "val": _metrics_for_json(item["classwise"]["val"], include_per_class=True),
                    "test": _metrics_for_json(item["classwise"]["test"], include_per_class=True),
                },
            }
            for source_name, item in source_ensemble_results.items()
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
