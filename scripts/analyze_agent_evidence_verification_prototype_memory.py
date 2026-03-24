#!/usr/bin/env python3
"""Analyze prototype evidence memory on top of the v2 comparative verifier."""

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
from scripts.analyze_agent_evidence_verification_v3 import (
    _comparison_row,
    _evaluate_score_bundle,
    _ordered_topk_indices,
    _pairwise_ranking_accuracy,
    _subset_metrics,
    _top2_disambiguation_accuracy,
)
from scripts.analyze_data_driven_agent_evidence_verification import (
    _bundle_summary,
    _candidate_recall_stats,
    _json_ready,
    _oracle_multilabel_metrics,
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
from src.utils.evidence_verification import (
    DEFAULT_EXPERT_PROMPTS,
    EXPERT_NAMES,
    build_benchmark_expert_phrase_banks,
    build_confusion_neighborhoods,
    build_expert_bank_statistics,
    build_margin_aware_gate,
    build_pairwise_relation_profiles,
    compute_expert_phrase_scores,
    compute_pairwise_comparative_scores,
    encode_expert_phrase_banks,
    learn_data_driven_relations,
    learn_prototype_memory_relations,
    select_prototype_profile_ids,
)
from src.utils.text_prior_analysis import apply_topk_rerank_fusion


DEFAULT_REUSE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prototype evidence memory experiments on top of fixed v2 comparative verification."
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
    )
    parser.add_argument(
        "--relation-variant",
        type=str,
        default="hard_negative_diff",
        choices=["hard_negative_diff", "confusion_hard_negative_diff", "support_contradiction"],
    )
    parser.add_argument("--profile-topn", type=int, default=20)
    parser.add_argument("--pair-profile-topn", type=int, default=5)
    parser.add_argument("--activation-topm", type=int, default=5)
    parser.add_argument("--contradiction-lambda", type=float, default=0.0)
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
    parser.add_argument("--prototype-k-list", type=str, default="2,3")
    parser.add_argument(
        "--prototype-source",
        type=str,
        default="focused",
        choices=["focused", "full"],
    )
    parser.add_argument(
        "--prototype-source-topm",
        type=int,
        default=None,
        help="If omitted, defaults to --activation-topm for focused source vectors.",
    )
    parser.add_argument("--prototype-min-positive-samples", type=int, default=64)
    parser.add_argument("--prototype-min-cluster-size", type=int, default=16)
    parser.add_argument("--prototype-random-state", type=int, default=0)
    parser.add_argument("--subset-margin-tau", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _parse_int_list(raw_value: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


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
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_agent_evidence_verification_prototype_memory"
    return Path(output_dir_arg)


def _build_topk_candidate_pairs(candidate_logits: np.ndarray, topk: int) -> List[List[tuple[int, int]]]:
    ordered = _ordered_topk_indices(candidate_logits, topk=topk)
    pairs: List[List[tuple[int, int]]] = []
    for row in ordered.tolist():
        sample_pairs: List[tuple[int, int]] = []
        for left_idx in range(len(row)):
            for right_idx in range(left_idx + 1, len(row)):
                sample_pairs.append((int(row[left_idx]), int(row[right_idx])))
        pairs.append(sample_pairs)
    return pairs


def _build_top2_confusion_mask(
    candidate_logits: np.ndarray,
    confusion_neighborhoods: Sequence[Sequence[int]],
) -> np.ndarray:
    ordered_top2 = _ordered_topk_indices(candidate_logits, topk=2)
    mask = np.zeros(ordered_top2.shape[0], dtype=np.bool_)
    for sample_idx in range(ordered_top2.shape[0]):
        class_i = int(ordered_top2[sample_idx, 0])
        class_j = int(ordered_top2[sample_idx, 1])
        neighbors_i = {int(x) for x in confusion_neighborhoods[class_i]}
        neighbors_j = {int(x) for x in confusion_neighborhoods[class_j]}
        if class_j in neighbors_i or class_i in neighbors_j:
            mask[sample_idx] = True
    return mask


def _build_prototype_class_rows(
    class_names: Sequence[str],
    prototype_bundle: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    prototype_counts = np.asarray(prototype_bundle["prototype_counts"], dtype=np.int64)
    class_positive_counts = np.asarray(prototype_bundle["class_positive_counts"], dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    for class_idx, class_name in enumerate(class_names):
        rows.append(
            {
                "class_id": int(class_idx),
                "class_name": str(class_name),
                "num_positive_samples": int(class_positive_counts[class_idx]),
                "actual_num_prototypes": int(prototype_counts[class_idx]),
                "fallback_reason": str(prototype_bundle["fallback_reasons"][class_idx]),
                "cluster_sizes": "|".join(
                    str(int(size)) for size in prototype_bundle["class_cluster_sizes"][class_idx]
                ),
            }
        )
    return rows


def _build_prototype_usage_rows(
    class_names: Sequence[str],
    prototype_bundle: Mapping[str, Any],
    candidate_profile_ids: np.ndarray,
    candidate_logits: np.ndarray,
    topk: int,
) -> List[Dict[str, Any]]:
    prototype_row_ids = np.asarray(prototype_bundle["prototype_row_ids"], dtype=np.int64)
    prototype_counts = np.asarray(prototype_bundle["prototype_counts"], dtype=np.int64)
    cluster_sizes = np.asarray(prototype_bundle["prototype_cluster_sizes"], dtype=np.int64)
    ordered_topk = _ordered_topk_indices(candidate_logits, topk=topk)
    topk_mask = np.zeros(candidate_logits.shape, dtype=np.bool_)
    row_ids = np.arange(candidate_logits.shape[0])[:, None]
    topk_mask[row_ids, ordered_topk] = True

    rows: List[Dict[str, Any]] = []
    for class_idx, class_name in enumerate(class_names):
        count = int(prototype_counts[class_idx])
        for local_idx in range(count):
            profile_row = int(prototype_row_ids[class_idx, local_idx])
            total_selected = int(np.sum(candidate_profile_ids[:, class_idx] == profile_row))
            topk_selected = int(
                np.sum((candidate_profile_ids[:, class_idx] == profile_row) & topk_mask[:, class_idx])
            )
            rows.append(
                {
                    "class_id": int(class_idx),
                    "class_name": str(class_name),
                    "prototype_local_id": int(local_idx),
                    "profile_row_id": profile_row,
                    "train_cluster_size": int(cluster_sizes[profile_row]),
                    "test_selection_count_all": total_selected,
                    "test_selection_count_when_in_topk": topk_selected,
                }
            )
    return rows


def main() -> None:
    args = _parse_args()
    prototype_k_list = _parse_int_list(args.prototype_k_list)
    if any(k <= 1 for k in prototype_k_list):
        raise ValueError("--prototype-k-list must contain values greater than 1 for the multi-prototype runs.")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.reuse_cache_dir) if args.reuse_cache_dir is not None else output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_ready = _cache_is_ready(cache_dir)

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

    if cache_ready:
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
        _, test_loader_base, _ = _build_dataset(
            cfg,
            datamodule,
            "test",
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
        test_base = _get_or_compute_bundle(
            cache_dir / "test_base.npz",
            lambda: _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples),
        )
        val_clip = _get_or_compute_bundle(
            cache_dir / "val_clip.npz",
            lambda: _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples),
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
        if slr_source not in text_pools:
            raise ValueError(f"Unsupported SLR source: {slr_source}")
        wrap_prompt = slr_source in {"lexical", "canonical"}
        slr_text_embeddings = _encode_text_pool(
            clip_model,
            text_pools[slr_source],
            wrap_prompt=wrap_prompt,
        )
        train_prior_logits = _text_logits_from_features(
            train_clip["features"], slr_text_embeddings, clip_logit_scale
        )
        val_prior_logits = _text_logits_from_features(
            val_clip["features"], slr_text_embeddings, clip_logit_scale
        )
        test_prior_logits = _text_logits_from_features(
            test_clip["features"], slr_text_embeddings, clip_logit_scale
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
    slr_thresholds = np.asarray(slr_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32)

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

    selected_experts = list(EXPERT_NAMES)
    v2_relation_bundle = learn_data_driven_relations(
        train_expert_phrase_scores,
        train_binary_labels,
        selected_experts=selected_experts,
        relation_mode=str(relation_cfg["relation_mode"]),
        profile_topn=int(args.profile_topn),
        hard_negative_topn=int(args.confusion_neighbor_topn),
        hard_negative_ids=relation_cfg["hard_negative_ids"],
        positive_only_scores=True,
    )
    v2_pairwise_profiles = build_pairwise_relation_profiles(
        v2_relation_bundle,
        selected_experts=selected_experts,
        pair_profile_topn=int(args.pair_profile_topn),
        contradiction_lambda=float(args.contradiction_lambda),
    )
    v2_pairwise_val = compute_pairwise_comparative_scores(
        val_expert_phrase_scores,
        v2_pairwise_profiles,
        slr_val_logits,
        selected_experts=selected_experts,
        candidate_topk=int(args.topk),
        activation_topm=int(args.activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )
    v2_pairwise_test = compute_pairwise_comparative_scores(
        test_expert_phrase_scores,
        v2_pairwise_profiles,
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
    v2_residual_val = v2_pairwise_val * v2_gate_val[:, None]
    v2_residual_test = v2_pairwise_test * v2_gate_test[:, None]
    v2_val_logits = apply_topk_rerank_fusion(
        slr_val_logits,
        v2_residual_val,
        topk=int(args.topk),
        alpha=float(args.v2_beta),
        mode=str(args.v2_fusion_mode),
    )
    v2_test_logits = apply_topk_rerank_fusion(
        slr_test_logits,
        v2_residual_test,
        topk=int(args.topk),
        alpha=float(args.v2_beta),
        mode=str(args.v2_fusion_mode),
    )
    v2_val_scores = _sigmoid(v2_val_logits)
    v2_test_scores = _sigmoid(v2_test_logits)
    v2_eval = _evaluate_score_bundle(
        v2_val_scores,
        val_base["labels"],
        v2_test_scores,
        test_base["labels"],
    )
    v2_thresholds = np.asarray(v2_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32)

    prototype_source_topm = (
        int(args.prototype_source_topm)
        if args.prototype_source_topm is not None
        else (int(args.activation_topm) if args.prototype_source == "focused" else None)
    )

    prototype_results: Dict[int, Dict[str, Any]] = {}
    main_comparison_rows = [
        _comparison_row("scenario_slr_c", {"classwise": slr_eval["classwise"]}),
        _comparison_row("v2_reference", {"classwise": v2_eval["classwise"]}),
    ]
    prototype_candidate_pairs = _build_topk_candidate_pairs(slr_test_logits, topk=int(args.topk))

    for prototype_k in prototype_k_list:
        bundle = learn_prototype_memory_relations(
            train_expert_phrase_scores,
            train_binary_labels,
            selected_experts=selected_experts,
            relation_mode=str(relation_cfg["relation_mode"]),
            profile_topn=int(args.profile_topn),
            hard_negative_topn=int(args.confusion_neighbor_topn),
            hard_negative_ids=relation_cfg["hard_negative_ids"],
            positive_only_scores=True,
            prototype_k=int(prototype_k),
            prototype_source=str(args.prototype_source),
            prototype_source_topm=prototype_source_topm,
            min_positive_samples=int(args.prototype_min_positive_samples),
            min_cluster_size=int(args.prototype_min_cluster_size),
            random_state=int(args.prototype_random_state),
        )
        val_profile_ids = select_prototype_profile_ids(
            val_expert_phrase_scores,
            bundle,
            selected_experts=selected_experts,
            positive_only_scores=True,
        )
        test_profile_ids = select_prototype_profile_ids(
            test_expert_phrase_scores,
            bundle,
            selected_experts=selected_experts,
            positive_only_scores=True,
        )
        prototype_pairwise_profiles = build_pairwise_relation_profiles(
            bundle,
            selected_experts=selected_experts,
            pair_profile_topn=int(args.pair_profile_topn),
            contradiction_lambda=float(args.contradiction_lambda),
        )
        pairwise_val = compute_pairwise_comparative_scores(
            val_expert_phrase_scores,
            prototype_pairwise_profiles,
            slr_val_logits,
            selected_experts=selected_experts,
            candidate_profile_ids=val_profile_ids,
            candidate_topk=int(args.topk),
            activation_topm=int(args.activation_topm),
            activation_positive_only=True,
            aggregate_mode="mean",
        )
        pairwise_test = compute_pairwise_comparative_scores(
            test_expert_phrase_scores,
            prototype_pairwise_profiles,
            slr_test_logits,
            selected_experts=selected_experts,
            candidate_profile_ids=test_profile_ids,
            candidate_topk=int(args.topk),
            activation_topm=int(args.activation_topm),
            activation_positive_only=True,
            aggregate_mode="mean",
        )
        residual_val = pairwise_val * v2_gate_val[:, None]
        residual_test = pairwise_test * v2_gate_test[:, None]
        val_logits = apply_topk_rerank_fusion(
            slr_val_logits,
            residual_val,
            topk=int(args.topk),
            alpha=float(args.v2_beta),
            mode=str(args.v2_fusion_mode),
        )
        test_logits = apply_topk_rerank_fusion(
            slr_test_logits,
            residual_test,
            topk=int(args.topk),
            alpha=float(args.v2_beta),
            mode=str(args.v2_fusion_mode),
        )
        val_scores = _sigmoid(val_logits)
        test_scores = _sigmoid(test_logits)
        eval_bundle = _evaluate_score_bundle(
            val_scores,
            val_base["labels"],
            test_scores,
            test_base["labels"],
        )
        thresholds = np.asarray(eval_bundle["classwise"]["val"]["class_thresholds"], dtype=np.float32)

        class_rows = _build_prototype_class_rows(class_names, bundle)
        usage_rows = _build_prototype_usage_rows(
            class_names,
            bundle,
            test_profile_ids,
            slr_test_logits,
            topk=int(args.topk),
        )
        _write_csv(output_dir / f"prototype_class_stats_k{prototype_k}.csv", class_rows)
        _write_csv(output_dir / f"prototype_usage_test_k{prototype_k}.csv", usage_rows)

        prototype_results[int(prototype_k)] = {
            "bundle": bundle,
            "eval": eval_bundle,
            "val_scores": val_scores.astype(np.float32),
            "test_scores": test_scores.astype(np.float32),
            "val_logits": val_logits.astype(np.float32),
            "test_logits": test_logits.astype(np.float32),
            "val_residual": residual_val.astype(np.float32),
            "test_residual": residual_test.astype(np.float32),
            "val_profile_ids": val_profile_ids.astype(np.int64),
            "test_profile_ids": test_profile_ids.astype(np.int64),
            "thresholds": thresholds.astype(np.float32),
            "class_rows": class_rows,
            "usage_rows": usage_rows,
        }
        main_comparison_rows.append(
            _comparison_row(f"v2_prototype_memory_k{prototype_k}", {"classwise": eval_bundle["classwise"]})
        )

    _write_csv(output_dir / "main_comparison.csv", main_comparison_rows)

    ordered_top2 = _ordered_topk_indices(slr_test_logits, topk=2)
    top1_margin = slr_test_logits[np.arange(slr_test_logits.shape[0]), ordered_top2[:, 0]] - slr_test_logits[
        np.arange(slr_test_logits.shape[0]),
        ordered_top2[:, 1],
    ]
    subset_masks = {
        "low_margin": np.asarray(top1_margin < float(args.subset_margin_tau), dtype=np.bool_),
        "top2_confusion_pair": _build_top2_confusion_mask(slr_test_logits, confusion_hard_negative_ids),
    }

    method_scores = {
        "slr_c": slr_test_scores.astype(np.float32),
        "v2_reference": v2_test_scores.astype(np.float32),
    }
    method_thresholds = {
        "slr_c": slr_thresholds,
        "v2_reference": v2_thresholds,
    }
    method_logits = {
        "slr_c": slr_test_logits.astype(np.float32),
        "v2_reference": v2_test_logits.astype(np.float32),
    }
    for prototype_k, record in prototype_results.items():
        method_name = f"prototype_k{prototype_k}"
        method_scores[method_name] = np.asarray(record["test_scores"], dtype=np.float32)
        method_thresholds[method_name] = np.asarray(record["thresholds"], dtype=np.float32)
        method_logits[method_name] = np.asarray(record["test_logits"], dtype=np.float32)

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

    pairwise_eval = {
        "all_test": {
            method_name: {
                "top2_accuracy": _top2_disambiguation_accuracy(
                    slr_test_logits,
                    method_logits[method_name],
                    test_base["labels"],
                    sample_mask=None,
                ),
                "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(
                    method_logits[method_name],
                    test_base["labels"],
                    prototype_candidate_pairs,
                    sample_mask=None,
                ),
            }
            for method_name in method_logits
        },
        "low_margin": {
            method_name: {
                "top2_accuracy": _top2_disambiguation_accuracy(
                    slr_test_logits,
                    method_logits[method_name],
                    test_base["labels"],
                    sample_mask=subset_masks["low_margin"],
                ),
                "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(
                    method_logits[method_name],
                    test_base["labels"],
                    prototype_candidate_pairs,
                    sample_mask=subset_masks["low_margin"],
                ),
            }
            for method_name in method_logits
        },
    }

    candidate_recall = _candidate_recall_stats(slr_test_logits, test_base["labels"], int(args.topk))
    oracle_topk = _oracle_multilabel_metrics(slr_test_logits, test_base["labels"], int(args.topk))

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "slr": {
            "source": slr_source,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "v2_reference_config": {
            "experts": selected_experts,
            "relation_variant": str(args.relation_variant),
            "profile_topn": int(args.profile_topn),
            "pair_profile_topn": int(args.pair_profile_topn),
            "activation_topm": int(args.activation_topm),
            "gate_gamma": float(args.v2_gate_gamma),
            "beta": float(args.v2_beta),
            "fusion_mode": str(args.v2_fusion_mode),
            "confusion_neighbor_topn": int(args.confusion_neighbor_topn),
            "confusion_topk": int(args.confusion_topk),
        },
        "prototype_memory": {
            "k_list": [int(k) for k in prototype_k_list],
            "source": str(args.prototype_source),
            "source_topm": None if prototype_source_topm is None else int(prototype_source_topm),
            "min_positive_samples": int(args.prototype_min_positive_samples),
            "min_cluster_size": int(args.prototype_min_cluster_size),
            "random_state": int(args.prototype_random_state),
            "selection_mode": "best_prototype",
        },
        "bank_stats": build_expert_bank_statistics(phrase_banks),
        "candidate_recall_topk": candidate_recall,
        "oracle_topk_upper_bound": oracle_topk,
        "slr_c": _bundle_summary(slr_eval["classwise"], include_per_class=True),
        "v2_reference": {
            "global": _bundle_summary(v2_eval["global"], include_per_class=True),
            "classwise": _bundle_summary(v2_eval["classwise"], include_per_class=True),
            "verification_gap": _verification_gap(
                slr_test_logits,
                v2_residual_test,
                test_base["labels"],
                int(args.topk),
            ),
        },
        "prototype_results": {
            str(prototype_k): {
                "stats": _json_ready(record["bundle"]["stats"]),
                "global": _bundle_summary(record["eval"]["global"], include_per_class=True),
                "classwise": _bundle_summary(record["eval"]["classwise"], include_per_class=True),
                "verification_gap": _verification_gap(
                    slr_test_logits,
                    np.asarray(record["test_residual"], dtype=np.float32),
                    test_base["labels"],
                    int(args.topk),
                ),
                "fallback_reasons": record["bundle"]["fallback_reasons"],
            }
            for prototype_k, record in prototype_results.items()
        },
        "main_comparison": main_comparison_rows,
        "hard_subset_eval": hard_subset_eval,
        "pairwise_eval": pairwise_eval,
        "subset_masks": {
            subset_name: {
                "num_samples": int(np.asarray(mask, dtype=np.bool_).sum()),
                "ratio": float(np.asarray(mask, dtype=np.bool_).mean()),
            }
            for subset_name, mask in subset_masks.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
