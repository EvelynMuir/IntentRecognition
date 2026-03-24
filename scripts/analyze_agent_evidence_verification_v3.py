#!/usr/bin/env python3
"""Analyze confusion-aware multi-agent ambiguity resolution v3 on top of scenario SLR-C."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_agent_evidence_verification_v2 import (
    _get_or_compute_bundle,
    _load_cache_bundle,
    _resolve_relation_config,
)
from scripts.analyze_data_driven_agent_evidence_verification import (
    _bundle_summary,
    _candidate_recall_stats,
    _evaluate_with_class_thresholds,
    _json_ready,
    _oracle_multilabel_metrics,
    _parse_float_list,
    _parse_int_list,
    _parse_profile_topn_list,
    _parse_str_list,
    _pearson_correlation,
    _sigmoid,
    _verification_gap,
)
from scripts.analyze_text_prior_boundary import (
    DEFAULT_BASELINE_CKPT,
    _build_dataset,
    _build_text_pools,
    _collect_clip_features,
    _collect_model_outputs,
    _encode_text_pool,
    _load_class_names,
    _normalize_source_name,
    _normalize_state_dict_keys,
    _resolve_ckpt_path,
    _resolve_device,
    _resolve_gemini_file,
    _resolve_run_dir,
    _text_logits_from_features,
    _write_csv,
)
from src.utils.decision_rule_calibration import search_classwise_thresholds
from src.utils.evidence_verification import (
    CONFUSION_AWARE_ROUTER_TRIGGER_MODES,
    DEFAULT_EXPERT_PROMPTS,
    EXPERT_NAMES,
    build_benchmark_expert_phrase_banks,
    build_confusion_aware_router,
    build_confusion_neighborhoods,
    build_expert_bank_statistics,
    build_margin_aware_gate,
    build_pairwise_relation_profiles,
    compute_expert_phrase_scores,
    compute_pairwise_comparative_scores,
    compute_specialist_pairwise_evidence,
    encode_expert_phrase_banks,
    learn_data_driven_relations,
    resolve_routed_specialist_evidence,
)
from src.utils.metrics import compute_difficulty_scores, compute_f1, compute_mAP
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    class_gain_rows,
    evaluate_with_validation_threshold,
)


DEFAULT_REUSE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal confusion-aware multi-agent v3 MVP experiments."
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
        choices=["scenario", "lexical", "canonical", "short_plus_detailed"],
    )
    parser.add_argument(
        "--reuse-cache-dir",
        type=str,
        default=str(DEFAULT_REUSE_CACHE_DIR) if DEFAULT_REUSE_CACHE_DIR.exists() else None,
        help="Optional existing cache directory with train/val/test *_base.npz and *_clip.npz bundles.",
    )
    parser.add_argument(
        "--relation-variant",
        type=str,
        default="hard_negative_diff",
        choices=["hard_negative_diff", "confusion_hard_negative_diff", "support_contradiction"],
    )
    parser.add_argument("--profile-topn", type=str, default="20")
    parser.add_argument("--pair-profile-topn", type=int, default=5)
    parser.add_argument("--activation-topm", type=int, default=5)
    parser.add_argument("--contradiction-lambda", type=float, default=1.0)
    parser.add_argument("--confusion-neighbor-topn", type=int, default=3)
    parser.add_argument("--confusion-topk", type=int, default=10)
    parser.add_argument("--bank-encode-batch-size", type=int, default=64)
    parser.add_argument("--v2-gate-gamma", type=float, default=2.0)
    parser.add_argument("--v2-beta", type=float, default=0.01)
    parser.add_argument(
        "--v2-fusion-mode",
        type=str,
        default="add",
        choices=["add", "add_norm"],
    )
    parser.add_argument(
        "--dispatch-modes",
        type=str,
        default="all,routed",
        help="Comma-separated dispatch modes. Supported: all,routed.",
    )
    parser.add_argument(
        "--trigger-modes",
        type=str,
        default="margin_confusion,always,margin_only,confusion_only",
        help=(
            "Comma-separated trigger modes. "
            "Supported: always,margin_only,confusion_only,margin_confusion,"
            "confusion_top2_only,confusion_top3_only,"
            "margin_and_confusion_top2,margin_or_confusion_top2,"
            "margin_and_confusion_top3,margin_or_confusion_top3."
        ),
    )
    parser.add_argument(
        "--margin-tau-list",
        type=str,
        default="0.5,1.0",
        help="Comma-separated top1-top2 margin thresholds for the rule-based router.",
    )
    parser.add_argument(
        "--v3-beta-list",
        type=str,
        default="0.005,0.01,0.02",
        help="Comma-separated residual fusion strengths for v3.",
    )
    parser.add_argument(
        "--v3-fusion-mode",
        type=str,
        default="add",
        choices=["add", "add_norm"],
    )
    parser.add_argument("--router-anchor-topn", type=int, default=3)
    parser.add_argument("--margin-fallback-topn", type=int, default=2)
    parser.add_argument("--max-routed-experts", type=int, default=2)
    parser.add_argument(
        "--main-trigger-mode",
        type=str,
        default="margin_confusion",
        help="Preferred trigger mode used for the legacy per-dispatch summary when present.",
    )
    parser.add_argument(
        "--subset-margin-tau",
        type=float,
        default=1.0,
        help="Low-margin subset threshold used in the report diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _required_cache_files(cache_dir: Path) -> List[Path]:
    return [
        cache_dir / "train_base.npz",
        cache_dir / "train_clip.npz",
        cache_dir / "val_base.npz",
        cache_dir / "val_clip.npz",
        cache_dir / "test_base.npz",
        cache_dir / "test_clip.npz",
    ]


def _cache_is_ready(cache_dir: Path) -> bool:
    return all(path.exists() for path in _required_cache_files(cache_dir))


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_agent_evidence_verification_v3"
    return Path(output_dir_arg)


def _ordered_topk_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    topk = max(1, min(int(topk), values.shape[1]))
    topk_idx = np.argpartition(-values, kth=topk - 1, axis=1)[:, :topk]
    topk_scores = np.take_along_axis(values, topk_idx, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    return np.take_along_axis(topk_idx, order, axis=1)


def _evaluate_score_bundle(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_scores: np.ndarray,
    test_targets: np.ndarray,
) -> Dict[str, Any]:
    global_metrics = evaluate_with_validation_threshold(
        val_scores,
        val_targets,
        test_scores,
        test_targets,
        use_inference_strategy=False,
    )
    class_thresholds = search_classwise_thresholds(val_scores, val_targets)
    classwise_metrics = _evaluate_with_class_thresholds(
        val_scores,
        val_targets,
        test_scores,
        test_targets,
        class_thresholds,
    )
    return {
        "global": global_metrics,
        "classwise": classwise_metrics,
    }


def _comparison_row(method: str, bundle: Mapping[str, Any]) -> Dict[str, Any]:
    test_metrics = bundle["classwise"]["test"]
    return {
        "method": str(method),
        "macro": float(test_metrics["macro"]) * 100.0,
        "micro": float(test_metrics["micro"]) * 100.0,
        "samples": float(test_metrics["samples"]) * 100.0,
        "mAP": float(test_metrics["mAP"]),
        "hard": float(test_metrics["hard"]) * 100.0,
    }


def _subset_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    class_thresholds: np.ndarray,
    sample_mask: np.ndarray,
) -> Dict[str, Any]:
    mask = np.asarray(sample_mask, dtype=np.bool_)
    subset_scores = np.asarray(scores, dtype=np.float32)[mask]
    subset_targets = np.asarray(targets, dtype=np.int32)[mask]
    if subset_scores.shape[0] == 0:
        return {
            "num_samples": 0,
            "macro": 0.0,
            "micro": 0.0,
            "samples": 0.0,
            "mAP": 0.0,
            "easy": 0.0,
            "medium": 0.0,
            "hard": 0.0,
        }

    micro, samples, macro, per_class = compute_f1(
        subset_targets,
        subset_scores,
        threshold=0.5,
        use_inference_strategy=False,
        class_thresholds=np.asarray(class_thresholds, dtype=np.float32),
    )
    difficulty = compute_difficulty_scores(np.asarray(per_class, dtype=np.float32))
    return {
        "num_samples": int(subset_scores.shape[0]),
        "macro": float(macro),
        "micro": float(micro),
        "samples": float(samples),
        "mAP": float(compute_mAP(subset_scores, subset_targets)),
        "easy": float(difficulty["easy"]),
        "medium": float(difficulty["medium"]),
        "hard": float(difficulty["hard"]),
    }


def _top2_disambiguation_accuracy(
    proposal_logits: np.ndarray,
    evaluated_logits: np.ndarray,
    targets: np.ndarray,
    sample_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    proposal_top2 = _ordered_topk_indices(proposal_logits, topk=2)
    mask = None if sample_mask is None else np.asarray(sample_mask, dtype=np.bool_)
    correct = 0
    total = 0

    for sample_idx in range(proposal_top2.shape[0]):
        if mask is not None and not bool(mask[sample_idx]):
            continue
        class_i = int(proposal_top2[sample_idx, 0])
        class_j = int(proposal_top2[sample_idx, 1])
        target_i = bool(targets[sample_idx, class_i] > 0)
        target_j = bool(targets[sample_idx, class_j] > 0)
        if target_i == target_j:
            continue
        total += 1
        if target_i and float(evaluated_logits[sample_idx, class_i]) > float(evaluated_logits[sample_idx, class_j]):
            correct += 1
        elif target_j and float(evaluated_logits[sample_idx, class_j]) > float(evaluated_logits[sample_idx, class_i]):
            correct += 1

    return {
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "num_samples": int(total),
    }


def _pairwise_ranking_accuracy(
    evaluated_logits: np.ndarray,
    targets: np.ndarray,
    selected_pairs: Sequence[Sequence[Sequence[int] | tuple[int, int]]],
    sample_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    mask = None if sample_mask is None else np.asarray(sample_mask, dtype=np.bool_)
    correct = 0
    total = 0

    for sample_idx, sample_pairs in enumerate(selected_pairs):
        if mask is not None and not bool(mask[sample_idx]):
            continue
        for class_i, class_j in sample_pairs:
            class_i = int(class_i)
            class_j = int(class_j)
            target_i = bool(targets[sample_idx, class_i] > 0)
            target_j = bool(targets[sample_idx, class_j] > 0)
            if target_i == target_j:
                continue
            total += 1
            if target_i and float(evaluated_logits[sample_idx, class_i]) > float(evaluated_logits[sample_idx, class_j]):
                correct += 1
            elif target_j and float(evaluated_logits[sample_idx, class_j]) > float(evaluated_logits[sample_idx, class_i]):
                correct += 1

    return {
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "num_pairs": int(total),
    }


def _router_stats(router_outputs: Mapping[str, Any], experts: Sequence[str]) -> Dict[str, Any]:
    trigger_mask = np.asarray(router_outputs["trigger_mask"], dtype=np.bool_)
    margin_trigger_mask = np.asarray(router_outputs["margin_trigger_mask"], dtype=np.bool_)
    confusion_trigger_mask = np.asarray(router_outputs["confusion_trigger_mask"], dtype=np.bool_)
    neighborhoods = list(router_outputs["selected_neighborhoods"])
    pairs = list(router_outputs["selected_pairs"])
    selected_experts = list(router_outputs["selected_experts"])
    expert_hist = {str(expert): 0 for expert in experts}

    for sample_experts in selected_experts:
        for expert in sample_experts:
            expert_hist[str(expert)] = int(expert_hist.get(str(expert), 0) + 1)

    triggered = np.where(trigger_mask)[0]
    neighborhood_sizes_all = [len(neighborhood) for neighborhood in neighborhoods]
    pair_counts_all = [len(sample_pairs) for sample_pairs in pairs]
    specialist_counts_all = [len(sample_experts) for sample_experts in selected_experts]
    neighborhood_sizes = [len(neighborhoods[idx]) for idx in triggered.tolist()]
    pair_counts = [len(pairs[idx]) for idx in triggered.tolist()]
    specialist_counts = [len(selected_experts[idx]) for idx in triggered.tolist()]

    avg_neighborhood_size = float(np.mean(neighborhood_sizes_all)) if neighborhood_sizes_all else 0.0
    avg_pairs_resolved = float(np.mean(pair_counts_all)) if pair_counts_all else 0.0
    avg_specialists_called = float(np.mean(specialist_counts_all)) if specialist_counts_all else 0.0
    avg_neighborhood_size_triggered = float(np.mean(neighborhood_sizes)) if neighborhood_sizes else 0.0
    avg_pairs_resolved_triggered = float(np.mean(pair_counts)) if pair_counts else 0.0
    avg_specialists_called_triggered = float(np.mean(specialist_counts)) if specialist_counts else 0.0

    return {
        "trigger_rate": float(trigger_mask.mean()),
        "margin_trigger_rate": float(margin_trigger_mask.mean()),
        "confusion_trigger_rate": float(confusion_trigger_mask.mean()),
        "avg_specialists_called": avg_specialists_called,
        "avg_neighborhood_size": avg_neighborhood_size,
        "avg_pairs_resolved": avg_pairs_resolved,
        "avg_neighborhood_size_triggered": avg_neighborhood_size_triggered,
        "avg_pairs_resolved_triggered": avg_pairs_resolved_triggered,
        "avg_specialists_called_triggered": avg_specialists_called_triggered,
        "avg_pair_count_triggered": avg_pairs_resolved_triggered,
        "avg_specialists_triggered": avg_specialists_called_triggered,
        "avg_specialists_all_samples": avg_specialists_called,
        "expert_call_histogram": expert_hist,
    }


def _best_key(record: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(record["classwise"]["val"]["macro"]),
        float(record["classwise"]["test"]["macro"]),
        float(record["classwise"]["test"]["hard"]),
    )


def _update_best_record(
    store: Dict[str, Dict[str, Any]],
    key: str,
    record: Dict[str, Any],
) -> None:
    current = store.get(str(key))
    if current is None or _best_key(record) > _best_key(current):
        store[str(key)] = record


def _router_best_row(key: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    config = record["config"]
    test_metrics = record["classwise"]["test"]
    router_stats = record["router_test_stats"]
    return {
        "key": str(key),
        "dispatch_mode": str(config["dispatch_mode"]),
        "trigger_mode": str(config["trigger_mode"]),
        "confusion_scope": str(record["router_test"]["confusion_scope"]),
        "trigger_logic": str(record["router_test"]["trigger_logic"]),
        "margin_tau": float(config["margin_tau"]),
        "beta": float(config["beta"]),
        "macro": float(test_metrics["macro"]) * 100.0,
        "micro": float(test_metrics["micro"]) * 100.0,
        "samples": float(test_metrics["samples"]) * 100.0,
        "mAP": float(test_metrics["mAP"]),
        "hard": float(test_metrics["hard"]) * 100.0,
        "trigger_rate": float(router_stats["trigger_rate"]),
        "margin_trigger_rate": float(router_stats["margin_trigger_rate"]),
        "confusion_trigger_rate": float(router_stats["confusion_trigger_rate"]),
        "avg_specialists_called": float(router_stats["avg_specialists_called"]),
        "avg_neighborhood_size": float(router_stats["avg_neighborhood_size"]),
        "avg_pairs_resolved": float(router_stats["avg_pairs_resolved"]),
        "avg_specialists_called_triggered": float(router_stats["avg_specialists_called_triggered"]),
        "avg_neighborhood_size_triggered": float(router_stats["avg_neighborhood_size_triggered"]),
        "avg_pairs_resolved_triggered": float(router_stats["avg_pairs_resolved_triggered"]),
    }


def main() -> None:
    args = _parse_args()
    dispatch_modes = _parse_str_list(args.dispatch_modes)
    trigger_modes = [str(mode).strip().lower().replace(" ", "_") for mode in _parse_str_list(args.trigger_modes)]
    main_trigger_mode = str(args.main_trigger_mode).strip().lower().replace(" ", "_")
    margin_tau_list = _parse_float_list(args.margin_tau_list)
    v3_beta_list = _parse_float_list(args.v3_beta_list)
    profile_topn_list = _parse_profile_topn_list(args.profile_topn)
    if len(profile_topn_list) != 1:
        raise ValueError("This minimal v3 script expects a single --profile-topn value.")
    profile_topn = profile_topn_list[0]

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _resolve_run_dir(args.run_dir, args.ckpt_path)
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    cfg = OmegaConf.load(cfg_path)
    gemini_file = _resolve_gemini_file(cfg, args.gemini_file)
    class_names = _load_class_names(Path(str(cfg.data.annotation_dir)) / str(cfg.data.val_annotation))

    cache_dir = Path(args.reuse_cache_dir) if args.reuse_cache_dir is not None else output_dir / "_cache"
    cache_ready = _cache_is_ready(cache_dir)
    if not cache_ready and cache_dir == Path(args.reuse_cache_dir or ""):
        cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _cache_is_ready(cache_dir):
        import clip

        device = torch.device("cpu")
        clip_model_name = OmegaConf.select(cfg, "model.net.clip_model_name", default="ViT-L/14")
        clip_model, _ = clip.load(str(clip_model_name), device=device)
        clip_model = clip_model.eval().to(device)
        train_base = _load_cache_bundle(cache_dir / "train_base.npz")
        train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
        val_base = _load_cache_bundle(cache_dir / "val_base.npz")
        val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
        test_base = _load_cache_bundle(cache_dir / "test_base.npz")
        test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")
    else:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.pin_memory = bool(args.pin_memory)

        datamodule = instantiate(cfg.data)
        model = instantiate(cfg.model)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _normalize_state_dict_keys(checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=args.strict_load)

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

        batch_size = int(getattr(cfg.data, "batch_size", datamodule.batch_size_per_device))
        _, train_loader_base, _ = _build_dataset(
            cfg,
            datamodule,
            "train",
            datamodule.val_test_transform,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )
        _, train_loader_clip, _ = _build_dataset(
            cfg,
            datamodule,
            "train",
            clip_preprocess,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )
        _, val_loader_base, _ = _build_dataset(
            cfg,
            datamodule,
            "val",
            datamodule.val_test_transform,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )
        _, val_loader_clip, _ = _build_dataset(
            cfg,
            datamodule,
            "val",
            clip_preprocess,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )
        _, test_loader_base, _ = _build_dataset(
            cfg,
            datamodule,
            "test",
            datamodule.val_test_transform,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )
        _, test_loader_clip, _ = _build_dataset(
            cfg,
            datamodule,
            "test",
            clip_preprocess,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )

        train_base = _get_or_compute_bundle(
            cache_dir / "train_base.npz",
            lambda: _collect_model_outputs(model, train_loader_base, device, max_samples=args.max_samples),
        )
        train_clip = _get_or_compute_bundle(
            cache_dir / "train_clip.npz",
            lambda: _collect_clip_features(clip_model, train_loader_clip, device, max_samples=args.max_samples),
        )
        val_base = _get_or_compute_bundle(
            cache_dir / "val_base.npz",
            lambda: _collect_model_outputs(model, val_loader_base, device, max_samples=args.max_samples),
        )
        val_clip = _get_or_compute_bundle(
            cache_dir / "val_clip.npz",
            lambda: _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples),
        )
        test_base = _get_or_compute_bundle(
            cache_dir / "test_base.npz",
            lambda: _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples),
        )
        test_clip = _get_or_compute_bundle(
            cache_dir / "test_clip.npz",
            lambda: _collect_clip_features(clip_model, test_loader_clip, device, max_samples=args.max_samples),
        )

    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())

    if train_base["image_ids"] != train_clip["image_ids"]:
        raise RuntimeError("Train image order mismatch between baseline and CLIP features.")
    if val_base["image_ids"] != val_clip["image_ids"]:
        raise RuntimeError("Validation image order mismatch between baseline and CLIP features.")
    if test_base["image_ids"] != test_clip["image_ids"]:
        raise RuntimeError("Test image order mismatch between baseline and CLIP features.")

    slr_source = _normalize_source_name(args.slr_source)
    text_pools = _build_text_pools(class_names, gemini_file)
    if slr_source == "short_plus_detailed":
        lexical_embeddings = _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True)
        canonical_embeddings = _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True)
        train_prior_logits = 0.5 * (
            _text_logits_from_features(train_clip["features"], lexical_embeddings, clip_logit_scale)
            + _text_logits_from_features(train_clip["features"], canonical_embeddings, clip_logit_scale)
        )
        val_prior_logits = 0.5 * (
            _text_logits_from_features(val_clip["features"], lexical_embeddings, clip_logit_scale)
            + _text_logits_from_features(val_clip["features"], canonical_embeddings, clip_logit_scale)
        )
        test_prior_logits = 0.5 * (
            _text_logits_from_features(test_clip["features"], lexical_embeddings, clip_logit_scale)
            + _text_logits_from_features(test_clip["features"], canonical_embeddings, clip_logit_scale)
        )
    else:
        wrap_prompt = slr_source in {"lexical", "canonical"}
        slr_text_embeddings = _encode_text_pool(
            clip_model,
            text_pools[slr_source],
            wrap_prompt=wrap_prompt,
        )
        train_prior_logits = _text_logits_from_features(train_clip["features"], slr_text_embeddings, clip_logit_scale)
        val_prior_logits = _text_logits_from_features(val_clip["features"], slr_text_embeddings, clip_logit_scale)
        test_prior_logits = _text_logits_from_features(test_clip["features"], slr_text_embeddings, clip_logit_scale)

    baseline_val_scores = _sigmoid(val_base["logits"])
    baseline_test_scores = _sigmoid(test_base["logits"])
    baseline_eval = _evaluate_score_bundle(
        baseline_val_scores,
        val_base["labels"],
        baseline_test_scores,
        test_base["labels"],
    )

    slr_train_logits = apply_topk_rerank_fusion(
        train_base["logits"],
        train_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_val_logits = apply_topk_rerank_fusion(
        val_base["logits"],
        val_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_test_logits = apply_topk_rerank_fusion(
        test_base["logits"],
        test_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_val_scores = _sigmoid(slr_val_logits)
    slr_test_scores = _sigmoid(slr_test_logits)
    slr_eval = _evaluate_score_bundle(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
    )

    phrase_banks = build_benchmark_expert_phrase_banks(include_activity=True)
    bank_embeddings = encode_expert_phrase_banks(
        clip_model=clip_model,
        phrase_banks=phrase_banks,
        prompt_templates=DEFAULT_EXPERT_PROMPTS,
        batch_size=int(args.bank_encode_batch_size),
    )
    train_expert_phrase_scores = compute_expert_phrase_scores(train_clip["features"], bank_embeddings, clip_logit_scale)
    val_expert_phrase_scores = compute_expert_phrase_scores(val_clip["features"], bank_embeddings, clip_logit_scale)
    test_expert_phrase_scores = compute_expert_phrase_scores(test_clip["features"], bank_embeddings, clip_logit_scale)
    train_binary_labels = (np.asarray(train_clip["soft_labels"], dtype=np.float32) > 0.0).astype(np.float32)

    confusion_hard_negative_ids = build_confusion_neighborhoods(
        slr_train_logits,
        train_binary_labels,
        topk=int(args.confusion_topk),
        top_n=int(args.confusion_neighbor_topn),
    )
    relation_cfg = _resolve_relation_config(
        args.relation_variant,
        confusion_hard_negative_ids=confusion_hard_negative_ids,
    )
    contradiction_lambda = (
        float(args.contradiction_lambda)
        if relation_cfg["contradiction_lambda_values"] is None
        else float(relation_cfg["contradiction_lambda_values"][0])
    )
    selected_experts = list(EXPERT_NAMES)
    relation_bundle = learn_data_driven_relations(
        train_expert_phrase_scores,
        train_binary_labels,
        selected_experts=selected_experts,
        relation_mode=str(relation_cfg["relation_mode"]),
        profile_topn=profile_topn,
        hard_negative_topn=int(args.confusion_neighbor_topn),
        hard_negative_ids=relation_cfg["hard_negative_ids"],
        positive_only_scores=True,
    )
    pairwise_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=selected_experts,
        pair_profile_topn=int(args.pair_profile_topn),
        contradiction_lambda=contradiction_lambda,
    )

    v2_pairwise_val = compute_pairwise_comparative_scores(
        val_expert_phrase_scores,
        pairwise_profiles,
        slr_val_logits,
        selected_experts=selected_experts,
        candidate_topk=int(args.topk),
        activation_topm=int(args.activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )
    v2_pairwise_test = compute_pairwise_comparative_scores(
        test_expert_phrase_scores,
        pairwise_profiles,
        slr_test_logits,
        selected_experts=selected_experts,
        candidate_topk=int(args.topk),
        activation_topm=int(args.activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )
    v2_gate_val = build_margin_aware_gate(
        slr_val_logits,
        mode="exp" if float(args.v2_gate_gamma) > 0.0 else "none",
        gamma=float(args.v2_gate_gamma),
    )
    v2_gate_test = build_margin_aware_gate(
        slr_test_logits,
        mode="exp" if float(args.v2_gate_gamma) > 0.0 else "none",
        gamma=float(args.v2_gate_gamma),
    )
    v2_val_residual = v2_pairwise_val * v2_gate_val[:, None]
    v2_test_residual = v2_pairwise_test * v2_gate_test[:, None]
    v2_val_logits = apply_topk_rerank_fusion(
        slr_val_logits,
        v2_val_residual,
        topk=int(args.topk),
        alpha=float(args.v2_beta),
        mode=args.v2_fusion_mode,
    )
    v2_test_logits = apply_topk_rerank_fusion(
        slr_test_logits,
        v2_test_residual,
        topk=int(args.topk),
        alpha=float(args.v2_beta),
        mode=args.v2_fusion_mode,
    )
    v2_val_scores = _sigmoid(v2_val_logits)
    v2_test_scores = _sigmoid(v2_test_logits)
    v2_eval = _evaluate_score_bundle(
        v2_val_scores,
        val_base["labels"],
        v2_test_scores,
        test_base["labels"],
    )

    search_rows: List[Dict[str, Any]] = []
    best_by_dispatch_trigger: Dict[str, Dict[str, Any]] = {}
    best_overall_by_dispatch: Dict[str, Dict[str, Any]] = {}
    best_main_by_dispatch: Dict[str, Dict[str, Any]] = {}

    for dispatch_mode in dispatch_modes:
        if dispatch_mode not in {"all", "routed"}:
            raise ValueError(f"Unsupported dispatch mode: {dispatch_mode}")
        for trigger_mode in trigger_modes:
            if trigger_mode not in CONFUSION_AWARE_ROUTER_TRIGGER_MODES:
                raise ValueError(f"Unsupported trigger mode: {trigger_mode}")
            for margin_tau in margin_tau_list:
                router_val = build_confusion_aware_router(
                    slr_val_logits,
                    confusion_neighborhoods=confusion_hard_negative_ids,
                    pairwise_profiles=pairwise_profiles,
                    selected_experts=selected_experts,
                    topk=int(args.topk),
                    margin_tau=float(margin_tau),
                    trigger_mode=trigger_mode,
                    dispatch_mode=dispatch_mode,
                    router_anchor_topn=int(args.router_anchor_topn),
                    margin_fallback_topn=int(args.margin_fallback_topn),
                    max_routed_experts=int(args.max_routed_experts),
                )
                router_test = build_confusion_aware_router(
                    slr_test_logits,
                    confusion_neighborhoods=confusion_hard_negative_ids,
                    pairwise_profiles=pairwise_profiles,
                    selected_experts=selected_experts,
                    topk=int(args.topk),
                    margin_tau=float(margin_tau),
                    trigger_mode=trigger_mode,
                    dispatch_mode=dispatch_mode,
                    router_anchor_topn=int(args.router_anchor_topn),
                    margin_fallback_topn=int(args.margin_fallback_topn),
                    max_routed_experts=int(args.max_routed_experts),
                )
                specialist_val = compute_specialist_pairwise_evidence(
                    val_expert_phrase_scores,
                    pairwise_profiles,
                    router_val,
                    selected_experts=selected_experts,
                    activation_topm=int(args.activation_topm),
                    activation_positive_only=True,
                )
                specialist_test = compute_specialist_pairwise_evidence(
                    test_expert_phrase_scores,
                    pairwise_profiles,
                    router_test,
                    selected_experts=selected_experts,
                    activation_topm=int(args.activation_topm),
                    activation_positive_only=True,
                )
                resolved_val = resolve_routed_specialist_evidence(
                    specialist_val,
                    router_val,
                    selected_experts=selected_experts,
                    aggregate_mode="mean",
                )
                resolved_test = resolve_routed_specialist_evidence(
                    specialist_test,
                    router_test,
                    selected_experts=selected_experts,
                    aggregate_mode="mean",
                )
                router_val_stats = _router_stats(router_val, selected_experts)
                router_test_stats = _router_stats(router_test, selected_experts)

                for beta in v3_beta_list:
                    v3_val_logits = apply_topk_rerank_fusion(
                        slr_val_logits,
                        resolved_val["resolved_scores"],
                        topk=int(args.topk),
                        alpha=float(beta),
                        mode=args.v3_fusion_mode,
                    )
                    v3_test_logits = apply_topk_rerank_fusion(
                        slr_test_logits,
                        resolved_test["resolved_scores"],
                        topk=int(args.topk),
                        alpha=float(beta),
                        mode=args.v3_fusion_mode,
                    )
                    v3_val_scores = _sigmoid(v3_val_logits)
                    v3_test_scores = _sigmoid(v3_test_logits)
                    v3_eval = _evaluate_score_bundle(
                        v3_val_scores,
                        val_base["labels"],
                        v3_test_scores,
                        test_base["labels"],
                    )

                    row = {
                        "method": "confusion_aware_v3_mvp",
                        "dispatch_mode": dispatch_mode,
                        "trigger_mode": trigger_mode,
                        "confusion_scope": str(router_test["confusion_scope"]),
                        "trigger_logic": str(router_test["trigger_logic"]),
                        "margin_tau": float(margin_tau),
                        "beta": float(beta),
                        "relation_variant": str(args.relation_variant),
                        "profile_topn": "all" if profile_topn is None else int(profile_topn),
                        "pair_profile_topn": int(args.pair_profile_topn),
                        "activation_topm": int(args.activation_topm),
                        "val_macro_global": float(v3_eval["global"]["val"]["macro"]),
                        "test_macro_global": float(v3_eval["global"]["test"]["macro"]),
                        "val_macro_classwise": float(v3_eval["classwise"]["val"]["macro"]),
                        "test_macro_classwise": float(v3_eval["classwise"]["test"]["macro"]),
                        "val_micro_classwise": float(v3_eval["classwise"]["val"]["micro"]),
                        "test_micro_classwise": float(v3_eval["classwise"]["test"]["micro"]),
                        "val_samples_classwise": float(v3_eval["classwise"]["val"]["samples"]),
                        "test_samples_classwise": float(v3_eval["classwise"]["test"]["samples"]),
                        "val_map_classwise": float(v3_eval["classwise"]["val"]["mAP"]),
                        "test_map_classwise": float(v3_eval["classwise"]["test"]["mAP"]),
                        "val_hard_classwise": float(v3_eval["classwise"]["val"]["hard"]),
                        "test_hard_classwise": float(v3_eval["classwise"]["test"]["hard"]),
                        "val_trigger_rate": float(router_val_stats["trigger_rate"]),
                        "val_margin_trigger_rate": float(router_val_stats["margin_trigger_rate"]),
                        "val_confusion_trigger_rate": float(router_val_stats["confusion_trigger_rate"]),
                        "test_trigger_rate": float(router_test_stats["trigger_rate"]),
                        "test_margin_trigger_rate": float(router_test_stats["margin_trigger_rate"]),
                        "test_confusion_trigger_rate": float(router_test_stats["confusion_trigger_rate"]),
                        "test_avg_specialists_called": float(router_test_stats["avg_specialists_called"]),
                        "test_avg_neighborhood_size": float(router_test_stats["avg_neighborhood_size"]),
                        "test_avg_pairs_resolved": float(router_test_stats["avg_pairs_resolved"]),
                        "test_avg_specialists_triggered": float(router_test_stats["avg_specialists_called_triggered"]),
                        "test_avg_neighborhood_size_triggered": float(router_test_stats["avg_neighborhood_size_triggered"]),
                        "test_avg_pairs_resolved_triggered": float(router_test_stats["avg_pairs_resolved_triggered"]),
                    }
                    search_rows.append(row)

                    record = {
                        "config": row,
                        "global": v3_eval["global"],
                        "classwise": v3_eval["classwise"],
                        "router_val": router_val,
                        "router_test": router_test,
                        "router_val_stats": router_val_stats,
                        "router_test_stats": router_test_stats,
                        "resolved_val": resolved_val,
                        "resolved_test": resolved_test,
                        "val_logits": v3_val_logits.astype(np.float32),
                        "test_logits": v3_test_logits.astype(np.float32),
                        "val_scores": v3_val_scores.astype(np.float32),
                        "test_scores": v3_test_scores.astype(np.float32),
                    }
                    _update_best_record(
                        best_by_dispatch_trigger,
                        key=f"{dispatch_mode}::{trigger_mode}",
                        record=record,
                    )
                    _update_best_record(best_overall_by_dispatch, key=dispatch_mode, record=record)
                    if trigger_mode == main_trigger_mode:
                        _update_best_record(best_main_by_dispatch, key=dispatch_mode, record=record)

    candidate_recall = _candidate_recall_stats(slr_test_logits, test_base["labels"], int(args.topk))
    oracle_topk = _oracle_multilabel_metrics(slr_test_logits, test_base["labels"], int(args.topk))

    router_probe_test = {
        "broad": build_confusion_aware_router(
            slr_test_logits,
            confusion_neighborhoods=confusion_hard_negative_ids,
            topk=int(args.topk),
            margin_tau=0.0,
            trigger_mode="confusion_only",
            dispatch_mode="all",
            router_anchor_topn=int(args.router_anchor_topn),
            margin_fallback_topn=int(args.margin_fallback_topn),
        ),
        "top2": build_confusion_aware_router(
            slr_test_logits,
            confusion_neighborhoods=confusion_hard_negative_ids,
            topk=int(args.topk),
            margin_tau=0.0,
            trigger_mode="confusion_top2_only",
            dispatch_mode="all",
            router_anchor_topn=int(args.router_anchor_topn),
            margin_fallback_topn=int(args.margin_fallback_topn),
        ),
        "top3": build_confusion_aware_router(
            slr_test_logits,
            confusion_neighborhoods=confusion_hard_negative_ids,
            topk=int(args.topk),
            margin_tau=0.0,
            trigger_mode="confusion_top3_only",
            dispatch_mode="all",
            router_anchor_topn=int(args.router_anchor_topn),
            margin_fallback_topn=int(args.margin_fallback_topn),
        ),
    }
    low_margin_mask = np.asarray(router_probe_test["broad"]["top1_margin"], dtype=np.float32) < float(args.subset_margin_tau)
    subset_masks = {
        "low_margin": low_margin_mask,
        "broad_confusion": np.asarray(router_probe_test["broad"]["confusion_trigger_mask"], dtype=np.bool_),
        "top2_confusion": np.asarray(router_probe_test["top2"]["confusion_trigger_mask"], dtype=np.bool_),
        "top3_confusion": np.asarray(router_probe_test["top3"]["confusion_trigger_mask"], dtype=np.bool_),
    }

    method_logits = {
        "slr_c": slr_test_logits.astype(np.float32),
        "v2_best": v2_test_logits.astype(np.float32),
    }
    method_scores = {
        "slr_c": slr_test_scores.astype(np.float32),
        "v2_best": v2_test_scores.astype(np.float32),
    }
    method_thresholds = {
        "slr_c": np.asarray(slr_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32),
        "v2_best": np.asarray(v2_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32),
    }
    for key, record in best_by_dispatch_trigger.items():
        method_logits[key] = np.asarray(record["test_logits"], dtype=np.float32)
        method_scores[key] = np.asarray(record["test_scores"], dtype=np.float32)
        method_thresholds[key] = np.asarray(record["classwise"]["val"]["class_thresholds"], dtype=np.float32)

    hard_subset_eval = {
        subset_name: {
            method_name: _subset_metrics(
                method_scores[method_name],
                test_base["labels"],
                method_thresholds[method_name],
                subset_mask,
            )
            for method_name in method_scores
        }
        for subset_name, subset_mask in subset_masks.items()
    }

    disambiguation_eval = {}
    for key, record in best_by_dispatch_trigger.items():
        trigger_mask = np.asarray(record["router_test"]["trigger_mask"], dtype=np.bool_)
        reference_pairs = record["router_test"]["selected_pairs"]
        disambiguation_eval[key] = {
            method_name: {
                "top2_accuracy_triggered": _top2_disambiguation_accuracy(
                    slr_test_logits,
                    method_logits[method_name],
                    test_base["labels"],
                    sample_mask=trigger_mask,
                ),
                "pairwise_ranking_triggered_pairs": _pairwise_ranking_accuracy(
                    method_logits[method_name],
                    test_base["labels"],
                    reference_pairs,
                    sample_mask=trigger_mask,
                ),
            }
            for method_name in ["slr_c", "v2_best", key]
        }

    per_class_gains_vs_slr = {}
    gain_source = best_main_by_dispatch if best_main_by_dispatch else best_overall_by_dispatch
    for dispatch_mode, record in gain_source.items():
        per_class_gains_vs_slr[dispatch_mode] = class_gain_rows(
            np.asarray(slr_eval["classwise"]["test"]["per_class_f1"], dtype=np.float32),
            np.asarray(record["classwise"]["test"]["per_class_f1"], dtype=np.float32),
            class_names,
            top_n=min(10, len(class_names)),
        )

    comparison_rows = [
        _comparison_row("baseline", baseline_eval),
        _comparison_row("scenario_slr_c", slr_eval),
        _comparison_row("v2_best_reference", v2_eval),
    ]
    for dispatch_mode in dispatch_modes:
        record = best_main_by_dispatch.get(dispatch_mode) or best_overall_by_dispatch.get(dispatch_mode)
        if record is None:
            continue
        trigger_mode = str(record["config"]["trigger_mode"])
        if trigger_mode == "margin_confusion" and dispatch_mode in {"all", "routed"}:
            label = f"v3_{dispatch_mode}_specialists"
        else:
            label = f"{dispatch_mode}_{trigger_mode}"
        comparison_rows.append(_comparison_row(label, record))

    router_best_rows = []
    for dispatch_mode in dispatch_modes:
        for trigger_mode in trigger_modes:
            key = f"{dispatch_mode}::{trigger_mode}"
            record = best_by_dispatch_trigger.get(key)
            if record is None:
                continue
            router_best_rows.append(_router_best_row(key, record))

    _write_csv(output_dir / "phase1_comparison.csv", comparison_rows)
    _write_csv(output_dir / "v3_search_results.csv", search_rows)
    _write_csv(output_dir / "router_best_by_config.csv", router_best_rows)

    diagnostics = {
        "candidate_recall_topk": candidate_recall,
        "oracle_topk_upper_bound": oracle_topk,
        "v2_reference": {
            "verification_gap": _verification_gap(
                slr_test_logits,
                v2_test_residual,
                test_base["labels"],
                int(args.topk),
            ),
            "correlation": _pearson_correlation(
                slr_test_logits,
                v2_test_residual,
                topk_mask=None,
            ),
        },
        "best_overall_by_dispatch": {
            dispatch_mode: {
                "config": record["config"],
                "verification_gap": _verification_gap(
                    slr_test_logits,
                    np.asarray(record["resolved_test"]["resolved_scores"], dtype=np.float32),
                    test_base["labels"],
                    int(args.topk),
                ),
                "correlation": _pearson_correlation(
                    slr_test_logits,
                    np.asarray(record["resolved_test"]["resolved_scores"], dtype=np.float32),
                    topk_mask=None,
                ),
            }
            for dispatch_mode, record in best_overall_by_dispatch.items()
        },
    }
    if best_main_by_dispatch:
        diagnostics["best_main_by_dispatch"] = {
            dispatch_mode: {
                "config": record["config"],
                "verification_gap": _verification_gap(
                    slr_test_logits,
                    np.asarray(record["resolved_test"]["resolved_scores"], dtype=np.float32),
                    test_base["labels"],
                    int(args.topk),
                ),
                "correlation": _pearson_correlation(
                    slr_test_logits,
                    np.asarray(record["resolved_test"]["resolved_scores"], dtype=np.float32),
                    topk_mask=None,
                ),
            }
            for dispatch_mode, record in best_main_by_dispatch.items()
        }

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "slr": {
            "source": slr_source,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "backend": {
            "relation_variant": str(args.relation_variant),
            "profile_topn": "all" if profile_topn is None else int(profile_topn),
            "pair_profile_topn": int(args.pair_profile_topn),
            "activation_topm": int(args.activation_topm),
            "contradiction_lambda": float(contradiction_lambda),
            "confusion_neighbor_topn": int(args.confusion_neighbor_topn),
            "confusion_topk": int(args.confusion_topk),
        },
        "router_search": {
            "dispatch_modes": dispatch_modes,
            "trigger_modes": trigger_modes,
            "margin_tau_list": margin_tau_list,
            "v3_beta_list": v3_beta_list,
            "main_trigger_mode": main_trigger_mode,
            "router_anchor_topn": int(args.router_anchor_topn),
            "margin_fallback_topn": int(args.margin_fallback_topn),
            "max_routed_experts": int(args.max_routed_experts),
        },
        "bank_stats": build_expert_bank_statistics(phrase_banks),
        "baseline": _bundle_summary(baseline_eval["classwise"], include_per_class=True),
        "slr_c": _bundle_summary(slr_eval["classwise"], include_per_class=True),
        "v2_reference": {
            "config": {
                "gate_gamma": float(args.v2_gate_gamma),
                "beta": float(args.v2_beta),
                "fusion_mode": str(args.v2_fusion_mode),
            },
            "global": _bundle_summary(v2_eval["global"], include_per_class=True),
            "classwise": _bundle_summary(v2_eval["classwise"], include_per_class=True),
        },
        "v3_main": {
            dispatch_mode: {
                "config": record["config"],
                "global": _bundle_summary(record["global"], include_per_class=True),
                "classwise": _bundle_summary(record["classwise"], include_per_class=True),
                "router_test_stats": record["router_test_stats"],
            }
            for dispatch_mode, record in best_main_by_dispatch.items()
        },
        "best_overall_by_dispatch": {
            dispatch_mode: {
                "config": record["config"],
                "global": _bundle_summary(record["global"], include_per_class=True),
                "classwise": _bundle_summary(record["classwise"], include_per_class=True),
                "router_test_stats": record["router_test_stats"],
            }
            for dispatch_mode, record in best_overall_by_dispatch.items()
        },
        "best_by_dispatch_trigger": {
            key: {
                "config": record["config"],
                "classwise": _bundle_summary(record["classwise"], include_per_class=False),
                "router_test_stats": record["router_test_stats"],
                "router_test": {
                    "trigger_logic": record["router_test"]["trigger_logic"],
                    "confusion_scope": record["router_test"]["confusion_scope"],
                    "trigger_mode": record["router_test"]["trigger_mode"],
                },
            }
            for key, record in best_by_dispatch_trigger.items()
        },
        "router_best_by_config": router_best_rows,
        "phase1_comparison": comparison_rows,
        "hard_subset_eval": hard_subset_eval,
        "disambiguation_eval": disambiguation_eval,
        "diagnostics": diagnostics,
        "subset_masks": {
            subset_name: {
                "num_samples": int(np.asarray(subset_mask, dtype=np.bool_).sum()),
                "ratio": float(np.asarray(subset_mask, dtype=np.bool_).mean()),
            }
            for subset_name, subset_mask in subset_masks.items()
        },
        "per_class_gains_vs_slr": per_class_gains_vs_slr,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
