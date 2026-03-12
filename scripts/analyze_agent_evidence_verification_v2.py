#!/usr/bin/env python3
"""Analyze comparative evidence verification v2 on top of scenario SLR-C."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_data_driven_agent_evidence_verification import (
    _bundle_summary,
    _candidate_recall_stats,
    _evaluate_with_class_thresholds,
    _json_ready,
    _oracle_multilabel_metrics,
    _parse_expert_subsets,
    _parse_float_list,
    _parse_int_list,
    _parse_profile_topn_list,
    _parse_str_list,
    _pearson_correlation,
    _select_best_row,
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
    DEFAULT_EXPERT_PROMPTS,
    build_benchmark_expert_phrase_banks,
    build_confusion_neighborhoods,
    build_expert_bank_statistics,
    build_intent_evidence_templates,
    build_margin_aware_gate,
    build_pairwise_relation_profiles,
    build_template_statistics,
    compute_data_driven_verification_scores,
    compute_expert_phrase_scores,
    compute_pairwise_comparative_scores,
    encode_expert_phrase_banks,
    learn_data_driven_relations,
)
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    class_gain_rows,
    evaluate_with_validation_threshold,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgentEvidenceVerification v2 with comparative verification and margin-aware gate."
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
        "--relation-variants",
        type=str,
        default="hard_negative_diff,confusion_hard_negative_diff,support_contradiction",
        help=(
            "Comma-separated relation variants. "
            "Supported: hard_negative_diff, confusion_hard_negative_diff, support_contradiction."
        ),
    )
    parser.add_argument(
        "--expert-subsets",
        type=str,
        default="all,scene",
        help="Comma-separated subsets. Use `all` or `scene+activity` style tokens.",
    )
    parser.add_argument(
        "--profile-topn-list",
        type=str,
        default="10,20",
        help="Comma-separated sparse profile sizes.",
    )
    parser.add_argument(
        "--activation-topm-list",
        type=str,
        default="5",
        help="Comma-separated activation top-m list.",
    )
    parser.add_argument(
        "--pair-profile-topn-list",
        type=str,
        default="5,10",
        help="Comma-separated pairwise discriminative profile sizes.",
    )
    parser.add_argument(
        "--beta-list",
        type=str,
        default="0.05,0.08,0.1,0.15,0.2,0.3",
        help="Comma-separated verification strengths.",
    )
    parser.add_argument(
        "--gate-gamma-list",
        type=str,
        default="0,1,2,4",
        help="Comma-separated margin-aware gate gamma list. `0` means no gating.",
    )
    parser.add_argument(
        "--contradiction-lambda-list",
        type=str,
        default="0.5,0.8,1.0",
        help="Comma-separated contradiction lambda list for support_contradiction.",
    )
    parser.add_argument(
        "--confusion-neighbor-topn",
        type=int,
        default=3,
        help="How many confusion neighbors to use for confusion_hard_negative_diff.",
    )
    parser.add_argument(
        "--confusion-topk",
        type=int,
        default=10,
        help="How many top candidates to inspect when building confusion neighborhoods.",
    )
    parser.add_argument("--bank-encode-batch-size", type=int, default=64)
    parser.add_argument(
        "--verification-fusion-mode",
        type=str,
        default="add_norm",
        choices=["add", "add_norm"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _resolve_relation_config(
    relation_variant: str,
    confusion_hard_negative_ids: List[List[int]],
) -> Dict[str, Any]:
    variant = str(relation_variant).strip().lower()
    if variant == "hard_negative_diff":
        return {
            "relation_mode": "hard_negative_diff",
            "hard_negative_ids": None,
            "contradiction_lambda_values": [0.0],
        }
    if variant == "confusion_hard_negative_diff":
        return {
            "relation_mode": "hard_negative_diff",
            "hard_negative_ids": confusion_hard_negative_ids,
            "contradiction_lambda_values": [0.0],
        }
    if variant == "support_contradiction":
        return {
            "relation_mode": "support_contradiction",
            "hard_negative_ids": None,
            "contradiction_lambda_values": None,
        }
    raise ValueError(f"Unsupported relation variant: {relation_variant}")


def _save_cache_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    arrays: Dict[str, Any] = {}
    for key, value in bundle.items():
        if isinstance(value, list):
            arrays[key] = np.asarray(value)
        else:
            arrays[key] = np.asarray(value)
    np.savez(path, **arrays)


def _load_cache_bundle(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    bundle: Dict[str, Any] = {}
    for key in data.files:
        value = data[key]
        if key == "image_ids":
            bundle[key] = [str(item) for item in value.tolist()]
        else:
            bundle[key] = value
    return bundle


def _get_or_compute_bundle(
    cache_path: Path,
    compute_fn,
) -> Dict[str, Any]:
    if cache_path.exists():
        return _load_cache_bundle(cache_path)
    bundle = compute_fn()
    _save_cache_bundle(cache_path, bundle)
    return bundle


def main() -> None:
    args = _parse_args()
    relation_variants = _parse_str_list(args.relation_variants)
    expert_subsets = _parse_expert_subsets(args.expert_subsets)
    profile_topn_list = _parse_profile_topn_list(args.profile_topn_list)
    activation_topm_list = _parse_int_list(args.activation_topm_list)
    pair_profile_topn_list = _parse_int_list(args.pair_profile_topn_list)
    beta_list = _parse_float_list(args.beta_list)
    gate_gamma_list = _parse_float_list(args.gate_gamma_list)
    contradiction_lambda_list = _parse_float_list(args.contradiction_lambda_list)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_agent_evidence_verification_v2"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_cache_files = [
        cache_dir / "train_base.npz",
        cache_dir / "train_clip.npz",
        cache_dir / "val_base.npz",
        cache_dir / "val_clip.npz",
        cache_dir / "test_base.npz",
        cache_dir / "test_clip.npz",
    ]
    cache_ready = all(path.exists() for path in required_cache_files)

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
    slr_class_thresholds = search_classwise_thresholds(slr_val_scores, val_base["labels"])
    slr_classwise = _evaluate_with_class_thresholds(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        slr_class_thresholds,
    )
    slr_global = evaluate_with_validation_threshold(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        use_inference_strategy=False,
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

    templates = build_intent_evidence_templates(
        class_names=class_names,
        gemini_file=gemini_file,
        intent2concepts_file=None,
        max_items_per_expert=8,
    )

    result_rows: List[Dict[str, Any]] = []
    best_record: Dict[str, Any] | None = None
    best_row: Dict[str, Any] | None = None
    best_by_variant: Dict[str, Dict[str, Any]] = {}

    gate_cache = {
        float(gamma): {
            "val": build_margin_aware_gate(slr_val_logits, mode="exp" if float(gamma) > 0.0 else "none", gamma=float(gamma)),
            "test": build_margin_aware_gate(slr_test_logits, mode="exp" if float(gamma) > 0.0 else "none", gamma=float(gamma)),
        }
        for gamma in gate_gamma_list
    }

    for subset in expert_subsets:
        selected_experts = list(subset["experts"])
        subset_name = str(subset["name"])
        for relation_variant in relation_variants:
            variant_cfg = _resolve_relation_config(
                relation_variant,
                confusion_hard_negative_ids=confusion_hard_negative_ids,
            )
            contradiction_values = (
                contradiction_lambda_list
                if variant_cfg["contradiction_lambda_values"] is None
                else variant_cfg["contradiction_lambda_values"]
            )
            for profile_topn in profile_topn_list:
                relation_bundle = learn_data_driven_relations(
                    train_expert_phrase_scores,
                    train_binary_labels,
                    selected_experts=selected_experts,
                    relation_mode=str(variant_cfg["relation_mode"]),
                    profile_topn=profile_topn,
                    hard_negative_topn=int(args.confusion_neighbor_topn),
                    hard_negative_ids=variant_cfg["hard_negative_ids"],
                    positive_only_scores=True,
                )

                for contradiction_lambda in contradiction_values:
                    for pair_profile_topn in pair_profile_topn_list:
                        pair_profiles = build_pairwise_relation_profiles(
                            relation_bundle,
                            selected_experts=selected_experts,
                            pair_profile_topn=int(pair_profile_topn),
                            contradiction_lambda=float(contradiction_lambda),
                        )
                        for activation_topm in activation_topm_list:
                            pairwise_val = compute_pairwise_comparative_scores(
                                val_expert_phrase_scores,
                                pair_profiles,
                                slr_val_logits,
                                selected_experts=selected_experts,
                                candidate_topk=int(args.topk),
                                activation_topm=int(activation_topm),
                                activation_positive_only=True,
                                aggregate_mode="mean",
                            )
                            pairwise_test = compute_pairwise_comparative_scores(
                                test_expert_phrase_scores,
                                pair_profiles,
                                slr_test_logits,
                                selected_experts=selected_experts,
                                candidate_topk=int(args.topk),
                                activation_topm=int(activation_topm),
                                activation_positive_only=True,
                                aggregate_mode="mean",
                            )

                            for gate_gamma in gate_gamma_list:
                                gated_val = pairwise_val * gate_cache[float(gate_gamma)]["val"][:, None]
                                gated_test = pairwise_test * gate_cache[float(gate_gamma)]["test"][:, None]

                                for beta in beta_list:
                                    v2_val_logits = apply_topk_rerank_fusion(
                                        slr_val_logits,
                                        gated_val,
                                        topk=int(args.topk),
                                        alpha=float(beta),
                                        mode=args.verification_fusion_mode,
                                    )
                                    v2_test_logits = apply_topk_rerank_fusion(
                                        slr_test_logits,
                                        gated_test,
                                        topk=int(args.topk),
                                        alpha=float(beta),
                                        mode=args.verification_fusion_mode,
                                    )
                                    v2_val_scores = _sigmoid(v2_val_logits)
                                    v2_test_scores = _sigmoid(v2_test_logits)

                                    global_metrics = evaluate_with_validation_threshold(
                                        v2_val_scores,
                                        val_base["labels"],
                                        v2_test_scores,
                                        test_base["labels"],
                                        use_inference_strategy=False,
                                    )
                                    class_thresholds = search_classwise_thresholds(
                                        v2_val_scores,
                                        val_base["labels"],
                                    )
                                    classwise_metrics = _evaluate_with_class_thresholds(
                                        v2_val_scores,
                                        val_base["labels"],
                                        v2_test_scores,
                                        test_base["labels"],
                                        class_thresholds,
                                    )

                                    row = {
                                        "method": "comparative_verification_v2",
                                        "subset": subset_name,
                                        "experts": "+".join(selected_experts),
                                        "relation_variant": relation_variant,
                                        "profile_topn": "all" if profile_topn is None else int(profile_topn),
                                        "pair_profile_topn": int(pair_profile_topn),
                                        "activation_topm": int(activation_topm),
                                        "gate_gamma": float(gate_gamma),
                                        "contradiction_lambda": float(contradiction_lambda),
                                        "beta": float(beta),
                                        "val_macro_global": float(global_metrics["val"]["macro"]),
                                        "test_macro_global": float(global_metrics["test"]["macro"]),
                                        "val_macro_classwise": float(classwise_metrics["val"]["macro"]),
                                        "test_macro_classwise": float(classwise_metrics["test"]["macro"]),
                                        "val_hard_classwise": float(classwise_metrics["val"]["hard"]),
                                        "test_hard_classwise": float(classwise_metrics["test"]["hard"]),
                                    }
                                    result_rows.append(row)

                                    current = best_by_variant.get(str(relation_variant))
                                    if current is None or (
                                        float(row["val_macro_classwise"]),
                                        float(row["test_macro_classwise"]),
                                    ) > (
                                        float(current["val_macro_classwise"]),
                                        float(current["test_macro_classwise"]),
                                    ):
                                        best_by_variant[str(relation_variant)] = row

                                    if best_record is None or (
                                        float(classwise_metrics["val"]["macro"]),
                                        float(classwise_metrics["test"]["macro"]),
                                        float(classwise_metrics["test"]["hard"]),
                                    ) > (
                                        float(best_record["classwise"]["val"]["macro"]),
                                        float(best_record["classwise"]["test"]["macro"]),
                                        float(best_record["classwise"]["test"]["hard"]),
                                    ):
                                        best_row = row
                                        best_record = {
                                            "config": row,
                                            "global": global_metrics,
                                            "classwise": classwise_metrics,
                                            "val_scores": v2_val_scores.astype(np.float32),
                                            "test_scores": v2_test_scores.astype(np.float32),
                                            "val_logits": v2_val_logits.astype(np.float32),
                                            "test_logits": v2_test_logits.astype(np.float32),
                                            "val_verification": gated_val.astype(np.float32),
                                            "test_verification": gated_test.astype(np.float32),
                                        }

    if best_record is None or best_row is None:
        raise RuntimeError("No v2 comparative verification result was produced.")

    candidate_recall = _candidate_recall_stats(slr_test_logits, test_base["labels"], int(args.topk))
    oracle_topk = _oracle_multilabel_metrics(slr_test_logits, test_base["labels"], int(args.topk))
    diagnostics = {
        "candidate_recall_topk": candidate_recall,
        "oracle_topk_upper_bound": oracle_topk,
        "verification_gap": _verification_gap(
            slr_test_logits,
            np.asarray(best_record["test_verification"], dtype=np.float32),
            test_base["labels"],
            int(args.topk),
        ),
        "correlation": {
            "all": _pearson_correlation(
                slr_test_logits,
                np.asarray(best_record["test_verification"], dtype=np.float32),
                topk_mask=None,
            ),
        },
    }

    best_per_class = np.asarray(best_record["classwise"]["test"]["per_class_f1"], dtype=np.float32)
    slr_per_class = np.asarray(slr_classwise["test"]["per_class_f1"], dtype=np.float32)
    per_class_gains = class_gain_rows(slr_per_class, best_per_class, class_names, top_n=min(10, len(class_names)))

    _write_csv(output_dir / "v2_search_results.csv", result_rows)
    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "slr": {
            "source": slr_source,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "verification": {
            "relation_variants": relation_variants,
            "expert_subsets": expert_subsets,
            "profile_topn_list": ["all" if x is None else int(x) for x in profile_topn_list],
            "activation_topm_list": [int(x) for x in activation_topm_list],
            "pair_profile_topn_list": [int(x) for x in pair_profile_topn_list],
            "beta_list": beta_list,
            "gate_gamma_list": gate_gamma_list,
            "contradiction_lambda_list": contradiction_lambda_list,
            "confusion_neighbor_topn": int(args.confusion_neighbor_topn),
            "confusion_topk": int(args.confusion_topk),
        },
        "template_stats": build_template_statistics(templates),
        "bank_stats": build_expert_bank_statistics(phrase_banks),
        "slr_c": {
            "global": _bundle_summary(slr_global, include_per_class=True),
            "classwise": _bundle_summary(slr_classwise, include_per_class=True),
        },
        "best_v2": {
            "row": best_row,
            "global": _bundle_summary(best_record["global"], include_per_class=True),
            "classwise": _bundle_summary(best_record["classwise"], include_per_class=True),
        },
        "best_by_variant": _json_ready(best_by_variant),
        "diagnostics": diagnostics,
        "per_class_gains_vs_slr": per_class_gains,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
