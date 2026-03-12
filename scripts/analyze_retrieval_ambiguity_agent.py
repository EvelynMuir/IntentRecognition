#!/usr/bin/env python3
"""Analyze a conservative retrieval-based ambiguity agent on top of fixed scenario SLR-C."""

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
)
from scripts.analyze_text_prior_boundary import (
    DEFAULT_BASELINE_CKPT,
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
    _write_csv,
)
from src.utils.evidence_verification import (
    DEFAULT_EXPERT_PROMPTS,
    EXPERT_NAMES,
    build_benchmark_expert_phrase_banks,
    build_confusion_neighborhoods,
    build_margin_aware_gate,
    build_pairwise_relation_profiles,
    compute_expert_phrase_scores,
    compute_pairwise_comparative_scores,
    encode_expert_phrase_banks,
    learn_data_driven_relations,
)
from src.utils.region_grounded_reasoning import normalize_topk_candidate_matrix
from src.utils.retrieval_ambiguity import (
    build_retrieval_evidence_scores,
    build_retrieval_memory_indices,
    compute_classwise_topk_mean_similarity,
    compute_similarity_matrix,
)
from src.utils.text_prior_analysis import apply_topk_rerank_fusion


DEFAULT_REUSE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
PHASE1_SETTINGS = ("support_only", "support_minus_global_refute")
PHASE2_SETTING = "support_minus_confusion_refute"


def _parse_int_list(raw_value: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_float_list(raw_value: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a conservative retrieval-based ambiguity agent experiment on top of fixed SLR-C."
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
        "--reuse-cache-dir",
        type=str,
        default=str(DEFAULT_REUSE_CACHE_DIR) if DEFAULT_REUSE_CACHE_DIR.exists() else None,
    )
    parser.add_argument("--r-list", type=str, default="3,5,8")
    parser.add_argument("--beta-list", type=str, default="0.005,0.01,0.02,0.05")
    parser.add_argument("--gate-gamma", type=float, default=2.0)
    parser.add_argument("--confusion-topk", type=int, default=10)
    parser.add_argument("--confusion-neighbor-topn", type=int, default=3)
    parser.add_argument("--subset-margin-tau", type=float, default=1.0)
    parser.add_argument("--similarity-chunk-size", type=int, default=512)
    parser.add_argument("--normalize-candidate-evidence", action="store_true")
    parser.add_argument("--v2-profile-topn", type=int, default=20)
    parser.add_argument("--v2-pair-profile-topn", type=int, default=5)
    parser.add_argument("--v2-activation-topm", type=int, default=5)
    parser.add_argument("--v2-hard-negative-topn", type=int, default=3)
    parser.add_argument("--v2-beta", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_retrieval_ambiguity_agent"
    return Path(output_dir_arg)


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


def _resolve_runtime_device(device_arg: str) -> torch.device:
    resolved = _resolve_device(device_arg)
    if resolved.type != "cuda":
        return resolved
    try:
        torch.empty(1, device=resolved)
        return resolved
    except Exception:
        return torch.device("cpu")


def _combine_with_gate(
    slr_logits: np.ndarray,
    residual: np.ndarray,
    gate: np.ndarray,
    topk: int,
    beta: float,
) -> np.ndarray:
    gated_residual = np.asarray(residual, dtype=np.float32) * np.asarray(gate, dtype=np.float32)[:, None]
    return apply_topk_rerank_fusion(
        np.asarray(slr_logits, dtype=np.float32),
        gated_residual,
        topk=int(topk),
        alpha=float(beta),
        mode="add",
    )


def _topk_candidate_pairs(candidate_logits: np.ndarray, topk: int) -> List[List[tuple[int, int]]]:
    ordered = _ordered_topk_indices(candidate_logits, topk=topk)
    output: List[List[tuple[int, int]]] = []
    for row in ordered.tolist():
        sample_pairs: List[tuple[int, int]] = []
        for left_idx in range(len(row)):
            for right_idx in range(left_idx + 1, len(row)):
                sample_pairs.append((int(row[left_idx]), int(row[right_idx])))
        output.append(sample_pairs)
    return output


def _pool_stats(class_memory_ids: Sequence[np.ndarray | Sequence[int]]) -> Dict[str, Any]:
    counts = np.asarray([len(item) for item in class_memory_ids], dtype=np.int64)
    return {
        "mean": float(counts.mean()) if counts.size > 0 else 0.0,
        "min": int(counts.min()) if counts.size > 0 else 0,
        "max": int(counts.max()) if counts.size > 0 else 0,
        "num_zero_classes": int(np.sum(counts == 0)),
        "counts": counts.tolist(),
    }


def _select_best(
    current_best: Dict[str, Any] | None,
    candidate_row: Dict[str, Any],
    candidate_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    if current_best is None:
        return {"row": candidate_row, "bundle": candidate_bundle}

    current_metrics = current_best["bundle"]["classwise"]["test"]
    candidate_metrics = candidate_bundle["classwise"]["test"]
    current_key = (
        float(current_best["bundle"]["classwise"]["val"]["macro"]),
        float(current_metrics["macro"]),
        float(current_metrics["hard"]),
    )
    candidate_key = (
        float(candidate_bundle["classwise"]["val"]["macro"]),
        float(candidate_metrics["macro"]),
        float(candidate_metrics["hard"]),
    )
    if candidate_key > current_key:
        return {"row": candidate_row, "bundle": candidate_bundle}
    return current_best


def _build_top_confusion_pair(
    proposal_logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    top2 = _ordered_topk_indices(proposal_logits, topk=2)
    binary = (np.asarray(labels, dtype=np.float32) > 0.0)
    counts: Dict[tuple[int, int], int] = {}
    for sample_idx in range(top2.shape[0]):
        class_i = int(top2[sample_idx, 0])
        class_j = int(top2[sample_idx, 1])
        target_i = bool(binary[sample_idx, class_i])
        target_j = bool(binary[sample_idx, class_j])
        if target_i == target_j:
            continue
        positive_class = class_i if target_i else class_j
        negative_class = class_j if target_i else class_i
        pair = (positive_class, negative_class)
        counts[pair] = counts.get(pair, 0) + 1

    if not counts:
        return {"positive_class": None, "negative_class": None, "count": 0}

    pair, count = max(counts.items(), key=lambda item: item[1])
    return {
        "positive_class": int(pair[0]),
        "negative_class": int(pair[1]),
        "count": int(count),
    }


def _top_confusion_pair_mask(
    proposal_logits: np.ndarray,
    positive_class: int | None,
    negative_class: int | None,
) -> np.ndarray:
    if positive_class is None or negative_class is None:
        return np.zeros(proposal_logits.shape[0], dtype=np.bool_)

    top2 = _ordered_topk_indices(proposal_logits, topk=2)
    mask = np.zeros(top2.shape[0], dtype=np.bool_)
    expected = {int(positive_class), int(negative_class)}
    for sample_idx in range(top2.shape[0]):
        if {int(top2[sample_idx, 0]), int(top2[sample_idx, 1])} == expected:
            mask[sample_idx] = True
    return mask


def _fixed_v2_reference(
    clip_model: torch.nn.Module,
    train_clip: Mapping[str, Any],
    val_clip: Mapping[str, Any],
    test_clip: Mapping[str, Any],
    train_binary_labels: np.ndarray,
    slr_train_logits: np.ndarray,
    slr_val_logits: np.ndarray,
    slr_test_logits: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    topk: int,
    profile_topn: int,
    pair_profile_topn: int,
    activation_topm: int,
    hard_negative_topn: int,
    gate_gamma: float,
    beta: float,
    confusion_topk: int,
) -> Dict[str, Any]:
    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    phrase_banks = build_benchmark_expert_phrase_banks(include_activity=True)
    bank_embeddings = encode_expert_phrase_banks(
        clip_model=clip_model,
        phrase_banks=phrase_banks,
        prompt_templates=DEFAULT_EXPERT_PROMPTS,
        batch_size=64,
    )
    train_expert_phrase_scores = compute_expert_phrase_scores(train_clip["features"], bank_embeddings, clip_logit_scale)
    val_expert_phrase_scores = compute_expert_phrase_scores(val_clip["features"], bank_embeddings, clip_logit_scale)
    test_expert_phrase_scores = compute_expert_phrase_scores(test_clip["features"], bank_embeddings, clip_logit_scale)

    relation_bundle = learn_data_driven_relations(
        train_expert_phrase_scores,
        train_binary_labels,
        selected_experts=list(EXPERT_NAMES),
        relation_mode="hard_negative_diff",
        profile_topn=int(profile_topn),
        hard_negative_topn=int(hard_negative_topn),
        hard_negative_ids=None,
        positive_only_scores=True,
    )
    pairwise_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=list(EXPERT_NAMES),
        pair_profile_topn=int(pair_profile_topn),
        contradiction_lambda=0.0,
    )
    pairwise_val = compute_pairwise_comparative_scores(
        val_expert_phrase_scores,
        pairwise_profiles,
        slr_val_logits,
        selected_experts=list(EXPERT_NAMES),
        candidate_topk=int(topk),
        activation_topm=int(activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )
    pairwise_test = compute_pairwise_comparative_scores(
        test_expert_phrase_scores,
        pairwise_profiles,
        slr_test_logits,
        selected_experts=list(EXPERT_NAMES),
        candidate_topk=int(topk),
        activation_topm=int(activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )

    gate_val = build_margin_aware_gate(
        slr_val_logits,
        mode="exp" if float(gate_gamma) > 0.0 else "none",
        gamma=float(gate_gamma),
    )
    gate_test = build_margin_aware_gate(
        slr_test_logits,
        mode="exp" if float(gate_gamma) > 0.0 else "none",
        gamma=float(gate_gamma),
    )
    val_logits = _combine_with_gate(slr_val_logits, pairwise_val, gate_val, topk=int(topk), beta=float(beta))
    test_logits = _combine_with_gate(slr_test_logits, pairwise_test, gate_test, topk=int(topk), beta=float(beta))
    val_scores = _sigmoid(val_logits)
    test_scores = _sigmoid(test_logits)
    evaluation = _evaluate_score_bundle(val_scores, val_labels, test_scores, test_labels)
    return {
        "classwise": evaluation["classwise"],
        "global": evaluation["global"],
        "val_logits": np.asarray(val_logits, dtype=np.float32),
        "test_logits": np.asarray(test_logits, dtype=np.float32),
        "val_scores": np.asarray(val_scores, dtype=np.float32),
        "test_scores": np.asarray(test_scores, dtype=np.float32),
        "pairwise_val": np.asarray(pairwise_val, dtype=np.float32),
        "pairwise_test": np.asarray(pairwise_test, dtype=np.float32),
        "gate_val": np.asarray(gate_val, dtype=np.float32),
        "gate_test": np.asarray(gate_test, dtype=np.float32),
        "settings": {
            "relation_variant": "hard_negative_diff",
            "profile_topn": int(profile_topn),
            "pair_profile_topn": int(pair_profile_topn),
            "activation_topm": int(activation_topm),
            "hard_negative_topn": int(hard_negative_topn),
            "gate_gamma": float(gate_gamma),
            "beta": float(beta),
            "confusion_topk": int(confusion_topk),
        },
    }


def main() -> None:
    args = _parse_args()
    r_list = _parse_int_list(args.r_list)
    beta_list = _parse_float_list(args.beta_list)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cache_dir = Path(args.reuse_cache_dir) if args.reuse_cache_dir is not None else output_dir / "_cache"
    base_cache_dir.mkdir(parents=True, exist_ok=True)

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
    device = _resolve_runtime_device(args.device)
    batch_size = int(getattr(cfg.data, "batch_size", 64))

    datamodule = None
    clip_model = None
    clip_preprocess = None
    if _cache_is_ready(base_cache_dir):
        import clip

        clip_model_name = OmegaConf.select(cfg, "model.net.clip_model_name", default="ViT-L/14")
        clip_model, clip_preprocess = clip.load(str(clip_model_name), device=device)
        clip_model = clip_model.eval().to(device)
        train_base = _load_cache_bundle(base_cache_dir / "train_base.npz")
        train_clip = _load_cache_bundle(base_cache_dir / "train_clip.npz")
        val_base = _load_cache_bundle(base_cache_dir / "val_base.npz")
        val_clip = _load_cache_bundle(base_cache_dir / "val_clip.npz")
        test_base = _load_cache_bundle(base_cache_dir / "test_base.npz")
        test_clip = _load_cache_bundle(base_cache_dir / "test_clip.npz")
    else:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.pin_memory = bool(args.pin_memory)
        datamodule = instantiate(cfg.data)
        model = instantiate(cfg.model)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _normalize_state_dict_keys(checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=args.strict_load)
        model = model.eval().to(device)

        clip_model = getattr(getattr(model, "net", model), "clip_model", None)
        clip_preprocess = getattr(getattr(model, "net", model), "clip_preprocess", None)
        if clip_model is None or clip_preprocess is None:
            import clip

            clip_model_name = OmegaConf.select(cfg, "model.net.clip_model_name", default="ViT-L/14")
            clip_model, clip_preprocess = clip.load(str(clip_model_name), device=device)
        clip_model = clip_model.eval().to(device)

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
            base_cache_dir / "train_base.npz",
            lambda: _collect_model_outputs(model, train_loader_base, device, max_samples=args.max_samples),
        )
        train_clip = _get_or_compute_bundle(
            base_cache_dir / "train_clip.npz",
            lambda: _collect_clip_features(clip_model, train_loader_clip, device, max_samples=args.max_samples),
        )
        val_base = _get_or_compute_bundle(
            base_cache_dir / "val_base.npz",
            lambda: _collect_model_outputs(model, val_loader_base, device, max_samples=args.max_samples),
        )
        val_clip = _get_or_compute_bundle(
            base_cache_dir / "val_clip.npz",
            lambda: _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples),
        )
        test_base = _get_or_compute_bundle(
            base_cache_dir / "test_base.npz",
            lambda: _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples),
        )
        test_clip = _get_or_compute_bundle(
            base_cache_dir / "test_clip.npz",
            lambda: _collect_clip_features(clip_model, test_loader_clip, device, max_samples=args.max_samples),
        )

    if train_base["image_ids"] != train_clip["image_ids"]:
        raise RuntimeError("Train image order mismatch between baseline and CLIP features.")
    if val_base["image_ids"] != val_clip["image_ids"]:
        raise RuntimeError("Validation image order mismatch between baseline and CLIP features.")
    if test_base["image_ids"] != test_clip["image_ids"]:
        raise RuntimeError("Test image order mismatch between baseline and CLIP features.")
    if clip_model is None:
        raise RuntimeError("CLIP model must be available to rebuild SLR-C.")

    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    text_pools = _build_text_pools(class_names, gemini_file)
    scenario_text_embeddings = _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False)
    train_prior_logits = _text_logits_from_features(train_clip["features"], scenario_text_embeddings, clip_logit_scale)
    val_prior_logits = _text_logits_from_features(val_clip["features"], scenario_text_embeddings, clip_logit_scale)
    test_prior_logits = _text_logits_from_features(test_clip["features"], scenario_text_embeddings, clip_logit_scale)

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
    slr_eval = _evaluate_score_bundle(slr_val_scores, val_base["labels"], slr_test_scores, test_base["labels"])

    train_binary_labels = (np.asarray(train_clip["soft_labels"], dtype=np.float32) > 0.0).astype(np.float32)
    confusion_neighborhoods = build_confusion_neighborhoods(
        slr_train_logits,
        train_binary_labels,
        topk=int(args.confusion_topk),
        top_n=int(args.confusion_neighbor_topn),
    )
    memory_indices = build_retrieval_memory_indices(train_binary_labels, confusion_neighborhoods=confusion_neighborhoods)
    memory_stats_rows = []
    support_counts = memory_indices["support"]
    global_refute_counts = memory_indices["global_refute"]
    confusion_refute_counts = memory_indices["confusion_refute"]
    for class_idx, class_name in enumerate(class_names):
        memory_stats_rows.append(
            {
                "class_id": int(class_idx),
                "class_name": str(class_name),
                "support_count": int(len(support_counts[class_idx])),
                "global_refute_count": int(len(global_refute_counts[class_idx])),
                "confusion_refute_count": int(len(confusion_refute_counts[class_idx])),
                "confusion_neighbors": "|".join(str(int(x)) for x in confusion_neighborhoods[class_idx]),
            }
        )
    _write_csv(output_dir / "retrieval_class_stats.csv", memory_stats_rows)

    v2_reference = _fixed_v2_reference(
        clip_model=clip_model,
        train_clip=train_clip,
        val_clip=val_clip,
        test_clip=test_clip,
        train_binary_labels=train_binary_labels,
        slr_train_logits=slr_train_logits,
        slr_val_logits=slr_val_logits,
        slr_test_logits=slr_test_logits,
        val_labels=val_base["labels"],
        test_labels=test_base["labels"],
        topk=int(args.topk),
        profile_topn=int(args.v2_profile_topn),
        pair_profile_topn=int(args.v2_pair_profile_topn),
        activation_topm=int(args.v2_activation_topm),
        hard_negative_topn=int(args.v2_hard_negative_topn),
        gate_gamma=float(args.gate_gamma),
        beta=float(args.v2_beta),
        confusion_topk=int(args.confusion_topk),
    )

    retrieval_gate_val = build_margin_aware_gate(
        slr_val_logits,
        mode="exp" if float(args.gate_gamma) > 0.0 else "none",
        gamma=float(args.gate_gamma),
    )
    retrieval_gate_test = build_margin_aware_gate(
        slr_test_logits,
        mode="exp" if float(args.gate_gamma) > 0.0 else "none",
        gamma=float(args.gate_gamma),
    )
    candidate_topk_val = _ordered_topk_indices(slr_val_logits, topk=int(args.topk))
    candidate_topk_test = _ordered_topk_indices(slr_test_logits, topk=int(args.topk))

    val_similarity = compute_similarity_matrix(
        val_clip["features"],
        train_clip["features"],
        chunk_size=int(args.similarity_chunk_size),
    )
    test_similarity = compute_similarity_matrix(
        test_clip["features"],
        train_clip["features"],
        chunk_size=int(args.similarity_chunk_size),
    )

    support_means_val = compute_classwise_topk_mean_similarity(val_similarity, memory_indices["support"], r_list)
    support_means_test = compute_classwise_topk_mean_similarity(test_similarity, memory_indices["support"], r_list)
    global_refute_means_val = compute_classwise_topk_mean_similarity(
        val_similarity,
        memory_indices["global_refute"],
        r_list,
    )
    global_refute_means_test = compute_classwise_topk_mean_similarity(
        test_similarity,
        memory_indices["global_refute"],
        r_list,
    )
    confusion_refute_means_val = compute_classwise_topk_mean_similarity(
        val_similarity,
        memory_indices["confusion_refute"],
        r_list,
    )
    confusion_refute_means_test = compute_classwise_topk_mean_similarity(
        test_similarity,
        memory_indices["confusion_refute"],
        r_list,
    )

    search_rows: List[Dict[str, Any]] = []
    best_by_setting: Dict[str, Dict[str, Any]] = {}
    best_by_r: Dict[int, Dict[str, Any]] = {}

    for r in r_list:
        raw_scores = {
            "support_only": {
                "val": build_retrieval_evidence_scores(support_means_val[r]),
                "test": build_retrieval_evidence_scores(support_means_test[r]),
            },
            "support_minus_global_refute": {
                "val": build_retrieval_evidence_scores(support_means_val[r], global_refute_means_val[r]),
                "test": build_retrieval_evidence_scores(support_means_test[r], global_refute_means_test[r]),
            },
            "support_minus_confusion_refute": {
                "val": build_retrieval_evidence_scores(support_means_val[r], confusion_refute_means_val[r]),
                "test": build_retrieval_evidence_scores(support_means_test[r], confusion_refute_means_test[r]),
            },
        }
        for setting_name, score_bundle in raw_scores.items():
            val_residual = np.asarray(score_bundle["val"], dtype=np.float32)
            test_residual = np.asarray(score_bundle["test"], dtype=np.float32)
            if args.normalize_candidate_evidence:
                val_residual = normalize_topk_candidate_matrix(val_residual, candidate_topk_val)
                test_residual = normalize_topk_candidate_matrix(test_residual, candidate_topk_test)

            for beta in beta_list:
                val_logits = _combine_with_gate(
                    slr_val_logits,
                    val_residual,
                    retrieval_gate_val,
                    topk=int(args.topk),
                    beta=float(beta),
                )
                test_logits = _combine_with_gate(
                    slr_test_logits,
                    test_residual,
                    retrieval_gate_test,
                    topk=int(args.topk),
                    beta=float(beta),
                )
                val_scores = _sigmoid(val_logits)
                test_scores = _sigmoid(test_logits)
                evaluation = _evaluate_score_bundle(val_scores, val_base["labels"], test_scores, test_base["labels"])
                row = {
                    "setting": str(setting_name),
                    "r": int(r),
                    "beta": float(beta),
                    "gate_gamma": float(args.gate_gamma),
                    "normalize_candidate_evidence": bool(args.normalize_candidate_evidence),
                    "val_macro_classwise": float(evaluation["classwise"]["val"]["macro"]),
                    "test_macro_classwise": float(evaluation["classwise"]["test"]["macro"]),
                    "test_hard_classwise": float(evaluation["classwise"]["test"]["hard"]),
                }
                search_rows.append(row)
                bundle = {
                    "classwise": evaluation["classwise"],
                    "global": evaluation["global"],
                    "val_logits": np.asarray(val_logits, dtype=np.float32),
                    "test_logits": np.asarray(test_logits, dtype=np.float32),
                    "val_scores": np.asarray(val_scores, dtype=np.float32),
                    "test_scores": np.asarray(test_scores, dtype=np.float32),
                    "val_residual": np.asarray(val_residual, dtype=np.float32),
                    "test_residual": np.asarray(test_residual, dtype=np.float32),
                    "val_raw_scores": np.asarray(score_bundle["val"], dtype=np.float32),
                    "test_raw_scores": np.asarray(score_bundle["test"], dtype=np.float32),
                }
                best_by_setting[setting_name] = _select_best(best_by_setting.get(setting_name), row, bundle)
                best_by_r[int(r)] = _select_best(best_by_r.get(int(r)), row, bundle)

    phase1_candidates = [best_by_setting[name] for name in PHASE1_SETTINGS]
    phase1_best = max(
        phase1_candidates,
        key=lambda item: (
            float(item["bundle"]["classwise"]["val"]["macro"]),
            float(item["bundle"]["classwise"]["test"]["macro"]),
            float(item["bundle"]["classwise"]["test"]["hard"]),
        ),
    )
    phase2_best = best_by_setting[PHASE2_SETTING]
    support_global_best = best_by_setting["support_minus_global_refute"]

    comparison_rows = [
        _comparison_row("scenario_slr_c", {"classwise": slr_eval["classwise"]}),
        _comparison_row("v2_best_reference", {"classwise": v2_reference["classwise"]}),
        _comparison_row("retrieval_phase1_image_only_best", phase1_best["bundle"]),
    ]
    confusion_distinct = phase2_best["row"] != support_global_best["row"]
    if confusion_distinct:
        comparison_rows.append(_comparison_row("retrieval_confusion_aware_best", phase2_best["bundle"]))

    _write_csv(output_dir / "main_comparison.csv", comparison_rows)
    _write_csv(output_dir / "retrieval_search_results.csv", search_rows)
    _write_csv(
        output_dir / "retrieval_setting_ablation.csv",
        [
            {
                "setting": str(setting_name),
                **best_by_setting[setting_name]["row"],
                "macro": float(best_by_setting[setting_name]["bundle"]["classwise"]["test"]["macro"]) * 100.0,
                "micro": float(best_by_setting[setting_name]["bundle"]["classwise"]["test"]["micro"]) * 100.0,
                "samples": float(best_by_setting[setting_name]["bundle"]["classwise"]["test"]["samples"]) * 100.0,
                "mAP": float(best_by_setting[setting_name]["bundle"]["classwise"]["test"]["mAP"]),
                "hard": float(best_by_setting[setting_name]["bundle"]["classwise"]["test"]["hard"]) * 100.0,
            }
            for setting_name in ["support_only", "support_minus_global_refute", "support_minus_confusion_refute"]
        ],
    )
    _write_csv(
        output_dir / "retrieval_r_ablation.csv",
        [
            {
                "r": int(r),
                "setting": str(best_by_r[int(r)]["row"]["setting"]),
                "beta": float(best_by_r[int(r)]["row"]["beta"]),
                "macro": float(best_by_r[int(r)]["bundle"]["classwise"]["test"]["macro"]) * 100.0,
                "micro": float(best_by_r[int(r)]["bundle"]["classwise"]["test"]["micro"]) * 100.0,
                "samples": float(best_by_r[int(r)]["bundle"]["classwise"]["test"]["samples"]) * 100.0,
                "mAP": float(best_by_r[int(r)]["bundle"]["classwise"]["test"]["mAP"]),
                "hard": float(best_by_r[int(r)]["bundle"]["classwise"]["test"]["hard"]) * 100.0,
            }
            for r in sorted(best_by_r)
        ],
    )

    method_scores = {
        "slr_c": np.asarray(slr_test_scores, dtype=np.float32),
        "v2_best": np.asarray(v2_reference["test_scores"], dtype=np.float32),
        "retrieval_support_only_best": np.asarray(best_by_setting["support_only"]["bundle"]["test_scores"], dtype=np.float32),
        "retrieval_support_minus_global_refute_best": np.asarray(
            best_by_setting["support_minus_global_refute"]["bundle"]["test_scores"],
            dtype=np.float32,
        ),
        "retrieval_support_minus_confusion_refute_best": np.asarray(
            best_by_setting["support_minus_confusion_refute"]["bundle"]["test_scores"],
            dtype=np.float32,
        ),
        "retrieval_phase1_image_only_best": np.asarray(phase1_best["bundle"]["test_scores"], dtype=np.float32),
    }
    method_thresholds = {
        "slr_c": np.asarray(slr_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32),
        "v2_best": np.asarray(v2_reference["classwise"]["val"]["class_thresholds"], dtype=np.float32),
        "retrieval_support_only_best": np.asarray(
            best_by_setting["support_only"]["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
        "retrieval_support_minus_global_refute_best": np.asarray(
            best_by_setting["support_minus_global_refute"]["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
        "retrieval_support_minus_confusion_refute_best": np.asarray(
            best_by_setting["support_minus_confusion_refute"]["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
        "retrieval_phase1_image_only_best": np.asarray(
            phase1_best["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
    }
    method_logits = {
        "slr_c": np.asarray(slr_test_logits, dtype=np.float32),
        "v2_best": np.asarray(v2_reference["test_logits"], dtype=np.float32),
        "retrieval_support_only_best": np.asarray(best_by_setting["support_only"]["bundle"]["test_logits"], dtype=np.float32),
        "retrieval_support_minus_global_refute_best": np.asarray(
            best_by_setting["support_minus_global_refute"]["bundle"]["test_logits"],
            dtype=np.float32,
        ),
        "retrieval_support_minus_confusion_refute_best": np.asarray(
            best_by_setting["support_minus_confusion_refute"]["bundle"]["test_logits"],
            dtype=np.float32,
        ),
        "retrieval_phase1_image_only_best": np.asarray(phase1_best["bundle"]["test_logits"], dtype=np.float32),
    }

    low_margin_top2 = _ordered_topk_indices(slr_test_logits, topk=2)
    low_margin_values = slr_test_logits[np.arange(slr_test_logits.shape[0]), low_margin_top2[:, 0]] - slr_test_logits[
        np.arange(slr_test_logits.shape[0]), low_margin_top2[:, 1]
    ]
    low_margin_mask = np.asarray(low_margin_values < float(args.subset_margin_tau), dtype=np.bool_)
    dominant_confusion_pair = _build_top_confusion_pair(slr_train_logits, train_binary_labels)
    top_confusion_pair_mask = _top_confusion_pair_mask(
        slr_test_logits,
        dominant_confusion_pair["positive_class"],
        dominant_confusion_pair["negative_class"],
    )
    candidate_pairs = _topk_candidate_pairs(slr_test_logits, topk=int(args.topk))

    subset_diagnostics = {
        "low_margin_subset": {
            key: _subset_metrics(value, test_base["labels"], method_thresholds[key], low_margin_mask)
            for key, value in method_scores.items()
        },
        "top_confusion_pair_subset": {
            key: _subset_metrics(value, test_base["labels"], method_thresholds[key], top_confusion_pair_mask)
            for key, value in method_scores.items()
        },
        "top2_disambiguation_accuracy": {
            key: _top2_disambiguation_accuracy(slr_test_logits, method_logits[key], test_base["labels"])
            for key in method_logits
        },
        "pairwise_ranking_accuracy": {
            key: _pairwise_ranking_accuracy(method_logits[key], test_base["labels"], candidate_pairs)
            for key in method_logits
        },
    }

    candidate_recall = _candidate_recall_stats(slr_test_logits, test_base["labels"], int(args.topk))
    oracle_topk = _oracle_multilabel_metrics(slr_test_logits, test_base["labels"], int(args.topk))
    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "base_cache_dir": str(base_cache_dir),
        "slr": {
            "source": "scenario",
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "memory": {
            "source": "train_clip_features_plus_labels",
            "query_source": "val_test_clip_features",
            "similarity": "cosine",
            "normalize_candidate_evidence": bool(args.normalize_candidate_evidence),
            "confusion_topk": int(args.confusion_topk),
            "confusion_neighbor_topn": int(args.confusion_neighbor_topn),
            "support_pool_stats": _pool_stats(memory_indices["support"]),
            "global_refute_pool_stats": _pool_stats(memory_indices["global_refute"]),
            "confusion_refute_pool_stats": _pool_stats(memory_indices["confusion_refute"]),
        },
        "v2_reference": {
            **v2_reference["settings"],
            "classwise": _bundle_summary(v2_reference["classwise"], include_per_class=True),
        },
        "slr_c": {
            "classwise": _bundle_summary(slr_eval["classwise"], include_per_class=True),
        },
        "retrieval": {
            "search_space": {
                "settings": ["support_only", "support_minus_global_refute", "support_minus_confusion_refute"],
                "r_list": r_list,
                "beta_list": beta_list,
                "gate_gamma": float(args.gate_gamma),
                "normalize_candidate_evidence": bool(args.normalize_candidate_evidence),
            },
            "best_by_setting": {
                key: {
                    "row": value["row"],
                    "classwise": _bundle_summary(value["bundle"]["classwise"], include_per_class=True),
                }
                for key, value in best_by_setting.items()
            },
            "best_by_r": {
                str(r): {
                    "row": value["row"],
                    "classwise": _bundle_summary(value["bundle"]["classwise"], include_per_class=True),
                }
                for r, value in sorted(best_by_r.items())
            },
            "phase1_image_only_best": {
                "row": phase1_best["row"],
                "classwise": _bundle_summary(phase1_best["bundle"]["classwise"], include_per_class=True),
            },
            "phase2_confusion_aware_best": {
                "row": phase2_best["row"],
                "classwise": _bundle_summary(phase2_best["bundle"]["classwise"], include_per_class=True),
                "distinct_from_global_negative": bool(confusion_distinct),
            },
        },
        "diagnostics": {
            "candidate_recall_topk": candidate_recall,
            "oracle_topk_upper_bound": oracle_topk,
            "dominant_confusion_pair": {
                **dominant_confusion_pair,
                "positive_class_name": None
                if dominant_confusion_pair["positive_class"] is None
                else str(class_names[int(dominant_confusion_pair["positive_class"])]),
                "negative_class_name": None
                if dominant_confusion_pair["negative_class"] is None
                else str(class_names[int(dominant_confusion_pair["negative_class"])]),
                "test_subset_size": int(np.sum(top_confusion_pair_mask)),
            },
        },
        "subset_diagnostics": subset_diagnostics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
