#!/usr/bin/env python3
"""Analyze v1 multi-expert evidence verification on top of SLR-C."""

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
    EXPERT_NAMES,
    aggregate_verification_scores,
    aggregate_verification_scores_with_routing,
    build_benchmark_expert_phrase_banks,
    build_classwise_routing_matrix,
    build_expert_bank_statistics,
    build_expert_phrase_banks,
    build_indexed_templates,
    build_intent_evidence_templates,
    build_template_statistics,
    compute_expert_match_scores,
    compute_expert_match_scores_with_similarity,
    compute_expert_phrase_scores,
    encode_template_phrase_sets,
    encode_expert_phrase_banks,
    resolve_default_intent2concepts_file,
    top_evidence_rows,
)
from src.utils.metrics import eval_validation_set
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    class_gain_rows,
    evaluate_fixed_threshold,
    evaluate_with_validation_threshold,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v1 multi-expert evidence verification analysis on top of SLR-C."
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=str(DEFAULT_BASELINE_CKPT) if DEFAULT_BASELINE_CKPT.exists() else None,
    )
    parser.add_argument("--gemini-file", type=str, default=None)
    parser.add_argument(
        "--intent2concepts-file",
        type=str,
        default=None,
        help="Optional override for intent2concepts.json.",
    )
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
        choices=[
            "lexical",
            "canonical",
            "scenario",
            "discriminative",
            "lexical_plus_canonical",
            "short_plus_detailed",
        ],
    )
    parser.add_argument(
        "--beta-list",
        type=str,
        default="0.05,0.1,0.2,0.3",
        help="Comma-separated verification strengths.",
    )
    parser.add_argument(
        "--verification-fusion-modes",
        type=str,
        default="add_norm",
        help="Comma-separated verification fusion modes. Supported: add,add_norm.",
    )
    parser.add_argument(
        "--expert-subsets",
        type=str,
        default="object,scene,style,activity,all",
        help=(
            "Comma-separated expert subsets. Use `all` or `object+scene+style` style tokens."
        ),
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="template_aware",
        choices=["template_aware", "equal", "fixed"],
    )
    parser.add_argument(
        "--expert-weights",
        type=str,
        default="object=1,scene=1,style=1,activity=1",
        help="Comma-separated expert weights, e.g. object=1,scene=1.5,style=1.2,activity=1.",
    )
    parser.add_argument("--max-items-per-expert", type=int, default=8)
    parser.add_argument(
        "--bank-source",
        type=str,
        default="benchmark",
        choices=["benchmark", "generic", "template"],
        help="Evidence extraction bank source. 'benchmark' uses standard benchmark label sets.",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="similarity",
        choices=["similarity", "exact"],
        help="How intent templates are matched against evidence banks.",
    )
    parser.add_argument(
        "--allow-negative-matching",
        action="store_true",
        help="Do not clip evidence scores at zero before similarity matching.",
    )
    parser.add_argument("--bank-encode-batch-size", type=int, default=64)
    parser.add_argument("--top-evidence-preview", type=int, default=20)
    parser.add_argument("--bank-topm", type=int, default=5)
    parser.add_argument(
        "--routing-modes",
        type=str,
        default="top1_positive,top2_soft",
        help="Comma-separated MEV v2 routing modes.",
    )
    parser.add_argument(
        "--routing-gamma-list",
        type=str,
        default="4,8",
        help="Comma-separated gamma list for soft routing modes.",
    )
    parser.add_argument(
        "--routing-gain-floor-list",
        type=str,
        default="0.0",
        help="Comma-separated gain floor list for routing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_expert_weights(raw: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid expert weight token: {item}")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in EXPERT_NAMES:
            raise ValueError(f"Unsupported expert name in weights: {key}")
        output[key] = float(value.strip())
    return output


def _parse_expert_subsets(raw: str) -> List[Dict[str, Any]]:
    subsets: List[Dict[str, Any]] = []
    for token in raw.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item == "all":
            experts = list(EXPERT_NAMES)
            name = "all"
        else:
            experts = [expert.strip() for expert in item.split("+") if expert.strip()]
            for expert in experts:
                if expert not in EXPERT_NAMES:
                    raise ValueError(f"Unsupported expert in subset `{item}`: {expert}")
            name = "+".join(experts)
        subsets.append({"name": name, "experts": experts})
    if not subsets:
        raise ValueError("At least one expert subset is required.")
    return subsets


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


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

    from src.utils.metrics import compute_difficulty_scores
    from src.utils.metrics import compute_f1
    from src.utils.metrics import compute_mAP

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


def _metrics_summary(metrics: Mapping[str, Any], include_per_class: bool = False) -> Dict[str, Any]:
    output = {
        "macro": float(metrics["macro"]),
        "micro": float(metrics["micro"]),
        "samples": float(metrics["samples"]),
        "mAP": float(metrics["mAP"]),
        "threshold": float(metrics["threshold"]),
        "easy": float(metrics["easy"]),
        "medium": float(metrics["medium"]),
        "hard": float(metrics["hard"]),
    }
    if include_per_class:
        output["per_class_f1"] = [float(x) for x in metrics["per_class_f1"]]
    if "class_thresholds" in metrics:
        output["class_thresholds"] = [float(x) for x in metrics["class_thresholds"]]
    return output


def _bundle_summary(bundle: Mapping[str, Mapping[str, Any]], include_per_class: bool = False) -> Dict[str, Any]:
    return {
        split: _metrics_summary(metrics, include_per_class=include_per_class)
        for split, metrics in bundle.items()
    }


def _select_best_row(rows: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row[metric_key]), float(row["test_macro_classwise"])))


def main() -> None:
    args = _parse_args()
    beta_list = _parse_float_list(args.beta_list)
    fusion_modes = _parse_str_list(args.verification_fusion_modes)
    expert_subsets = _parse_expert_subsets(args.expert_subsets)
    expert_weights = _parse_expert_weights(args.expert_weights)
    routing_modes = _parse_str_list(args.routing_modes)
    routing_gamma_list = _parse_float_list(args.routing_gamma_list)
    routing_gain_floor_list = _parse_float_list(args.routing_gain_floor_list)

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

    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    gemini_file = _resolve_gemini_file(cfg, args.gemini_file)
    class_names = _load_class_names(Path(str(cfg.data.annotation_dir)) / str(cfg.data.val_annotation))

    intent2concepts_file = (
        Path(args.intent2concepts_file)
        if args.intent2concepts_file is not None
        else resolve_default_intent2concepts_file(
            data_root=Path(str(cfg.data.data_dir)),
            project_root=PROJECT_ROOT,
        )
    )
    if intent2concepts_file is not None and not Path(intent2concepts_file).exists():
        raise FileNotFoundError(f"intent2concepts file not found: {intent2concepts_file}")

    batch_size = int(getattr(cfg.data, "batch_size", datamodule.batch_size_per_device))
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

    val_base = _collect_model_outputs(model, val_loader_base, device, max_samples=args.max_samples)
    test_base = _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples)
    val_clip = _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples)
    test_clip = _collect_clip_features(clip_model, test_loader_clip, device, max_samples=args.max_samples)

    if val_base["image_ids"] != val_clip["image_ids"]:
        raise RuntimeError("Validation image order mismatch between baseline and CLIP features.")
    if test_base["image_ids"] != test_clip["image_ids"]:
        raise RuntimeError("Test image order mismatch between baseline and CLIP features.")

    slr_source = _normalize_source_name(args.slr_source)
    text_pools = _build_text_pools(class_names, gemini_file)
    if slr_source == "short_plus_detailed":
        lexical_embeddings = _encode_text_pool(
            clip_model,
            text_pools["lexical"],
            wrap_prompt=True,
        )
        canonical_embeddings = _encode_text_pool(
            clip_model,
            text_pools["canonical"],
            wrap_prompt=True,
        )
        lexical_val_prior_logits = _text_logits_from_features(
            val_clip["features"],
            lexical_embeddings,
            clip_logit_scale,
        )
        lexical_test_prior_logits = _text_logits_from_features(
            test_clip["features"],
            lexical_embeddings,
            clip_logit_scale,
        )
        canonical_val_prior_logits = _text_logits_from_features(
            val_clip["features"],
            canonical_embeddings,
            clip_logit_scale,
        )
        canonical_test_prior_logits = _text_logits_from_features(
            test_clip["features"],
            canonical_embeddings,
            clip_logit_scale,
        )
        slr_val_prior_logits = 0.5 * (lexical_val_prior_logits + canonical_val_prior_logits)
        slr_test_prior_logits = 0.5 * (lexical_test_prior_logits + canonical_test_prior_logits)
    else:
        if slr_source not in text_pools:
            raise ValueError(f"Unsupported SLR source: {slr_source}")
        wrap_prompt = slr_source in {"lexical", "canonical"}
        slr_text_embeddings = _encode_text_pool(
            clip_model,
            text_pools[slr_source],
            wrap_prompt=wrap_prompt,
        )

        slr_val_prior_logits = _text_logits_from_features(
            val_clip["features"],
            slr_text_embeddings,
            clip_logit_scale,
        )
        slr_test_prior_logits = _text_logits_from_features(
            test_clip["features"],
            slr_text_embeddings,
            clip_logit_scale,
        )

    slr_val_logits = apply_topk_rerank_fusion(
        val_base["logits"],
        slr_val_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_test_logits = apply_topk_rerank_fusion(
        test_base["logits"],
        slr_test_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode="add_norm",
    )
    slr_val_scores = _sigmoid(slr_val_logits)
    slr_test_scores = _sigmoid(slr_test_logits)

    baseline_global = evaluate_with_validation_threshold(
        val_base["scores"],
        val_base["labels"],
        test_base["scores"],
        test_base["labels"],
        use_inference_strategy=False,
    )
    baseline_class_thresholds = search_classwise_thresholds(
        val_base["scores"],
        val_base["labels"],
    )
    baseline_classwise = _evaluate_with_class_thresholds(
        val_base["scores"],
        val_base["labels"],
        test_base["scores"],
        test_base["labels"],
        baseline_class_thresholds,
    )

    slr_global = evaluate_with_validation_threshold(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        use_inference_strategy=False,
    )
    slr_class_thresholds = search_classwise_thresholds(
        slr_val_scores,
        val_base["labels"],
    )
    slr_classwise = _evaluate_with_class_thresholds(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        slr_class_thresholds,
    )

    templates = build_intent_evidence_templates(
        class_names=class_names,
        gemini_file=gemini_file,
        intent2concepts_file=intent2concepts_file,
        max_items_per_expert=int(args.max_items_per_expert),
    )
    if args.bank_source in {"benchmark", "generic"}:
        phrase_banks = build_benchmark_expert_phrase_banks(include_activity=True)
    else:
        phrase_banks = build_expert_phrase_banks(templates)
    bank_embeddings = encode_expert_phrase_banks(
        clip_model=clip_model,
        phrase_banks=phrase_banks,
        prompt_templates=DEFAULT_EXPERT_PROMPTS,
        batch_size=int(args.bank_encode_batch_size),
    )

    val_expert_phrase_scores = compute_expert_phrase_scores(
        val_clip["features"],
        bank_embeddings,
        clip_logit_scale,
    )
    test_expert_phrase_scores = compute_expert_phrase_scores(
        test_clip["features"],
        bank_embeddings,
        clip_logit_scale,
    )
    if args.match_mode == "similarity":
        template_phrase_embeddings = encode_template_phrase_sets(
            clip_model=clip_model,
            templates=templates,
            prompt_templates=DEFAULT_EXPERT_PROMPTS,
            batch_size=int(args.bank_encode_batch_size),
        )
        val_expert_match_scores = compute_expert_match_scores_with_similarity(
            val_expert_phrase_scores,
            bank_embeddings,
            template_phrase_embeddings,
            aggregation_mode="average",
            positive_only=not bool(args.allow_negative_matching),
        )
        test_expert_match_scores = compute_expert_match_scores_with_similarity(
            test_expert_phrase_scores,
            bank_embeddings,
            template_phrase_embeddings,
            aggregation_mode="average",
            positive_only=not bool(args.allow_negative_matching),
        )
        indexed_templates = build_indexed_templates(templates, build_expert_phrase_banks(templates))
    else:
        indexed_templates = build_indexed_templates(templates, phrase_banks)
        val_expert_match_scores = compute_expert_match_scores(val_expert_phrase_scores, indexed_templates)
        test_expert_match_scores = compute_expert_match_scores(test_expert_phrase_scores, indexed_templates)

    result_rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []
    best_per_subset: Dict[str, Dict[str, Any]] = {}
    raw_records: List[Dict[str, Any]] = []

    for subset in expert_subsets:
        subset_name = str(subset["name"])
        selected_experts = list(subset["experts"])

        verification_val = aggregate_verification_scores(
            val_expert_match_scores,
            indexed_templates,
            selected_experts=selected_experts,
            expert_weights=expert_weights,
            weight_mode=args.weight_mode,
        )
        verification_test = aggregate_verification_scores(
            test_expert_match_scores,
            indexed_templates,
            selected_experts=selected_experts,
            expert_weights=expert_weights,
            weight_mode=args.weight_mode,
        )

        for fusion_mode in fusion_modes:
            if fusion_mode not in {"add", "add_norm"}:
                raise ValueError(f"Unsupported verification fusion mode: {fusion_mode}")
            for beta in beta_list:
                mev_val_logits = apply_topk_rerank_fusion(
                    slr_val_logits,
                    verification_val,
                    topk=int(args.topk),
                    alpha=float(beta),
                    mode=fusion_mode,
                )
                mev_test_logits = apply_topk_rerank_fusion(
                    slr_test_logits,
                    verification_test,
                    topk=int(args.topk),
                    alpha=float(beta),
                    mode=fusion_mode,
                )
                mev_val_scores = _sigmoid(mev_val_logits)
                mev_test_scores = _sigmoid(mev_test_logits)

                global_metrics = evaluate_with_validation_threshold(
                    mev_val_scores,
                    val_base["labels"],
                    mev_test_scores,
                    test_base["labels"],
                    use_inference_strategy=False,
                )
                class_thresholds = search_classwise_thresholds(
                    mev_val_scores,
                    val_base["labels"],
                )
                classwise_metrics = _evaluate_with_class_thresholds(
                    mev_val_scores,
                    val_base["labels"],
                    mev_test_scores,
                    test_base["labels"],
                    class_thresholds,
                )

                row = {
                    "subset": subset_name,
                    "experts": "+".join(selected_experts),
                    "fusion_mode": fusion_mode,
                    "beta": float(beta),
                    "val_macro_global": float(global_metrics["val"]["macro"]),
                    "val_hard_global": float(global_metrics["val"]["hard"]),
                    "test_macro_global": float(global_metrics["test"]["macro"]),
                    "test_hard_global": float(global_metrics["test"]["hard"]),
                    "val_macro_classwise": float(classwise_metrics["val"]["macro"]),
                    "val_hard_classwise": float(classwise_metrics["val"]["hard"]),
                    "test_macro_classwise": float(classwise_metrics["test"]["macro"]),
                    "test_hard_classwise": float(classwise_metrics["test"]["hard"]),
                }
                result_rows.append(row)

                detailed_result = {
                    "config": {
                        "subset": subset_name,
                        "experts": selected_experts,
                        "fusion_mode": fusion_mode,
                        "beta": float(beta),
                        "weight_mode": args.weight_mode,
                        "expert_weights": expert_weights,
                    },
                    "global": _bundle_summary(global_metrics, include_per_class=True),
                    "classwise": _bundle_summary(classwise_metrics, include_per_class=True),
                    "verification_preview": {
                        "val_macro": float(np.mean(verification_val)),
                        "test_macro": float(np.mean(verification_test)),
                    },
                }
                detailed_results.append(detailed_result)

                raw_record = {
                    "config": {
                        "subset": subset_name,
                        "experts": selected_experts,
                        "fusion_mode": fusion_mode,
                        "beta": float(beta),
                        "weight_mode": args.weight_mode,
                        "expert_weights": expert_weights,
                    },
                    "global": global_metrics,
                    "classwise": classwise_metrics,
                    "val_scores": mev_val_scores.astype(np.float32),
                    "test_scores": mev_test_scores.astype(np.float32),
                }
                raw_records.append(raw_record)

                current_best = best_per_subset.get(subset_name)
                current_score = float(classwise_metrics["val"]["macro"])
                if current_best is None or current_score > float(current_best["classwise"]["val"]["macro"]):
                    best_per_subset[subset_name] = raw_record

    routing_rows: List[Dict[str, Any]] = []
    routing_details: List[Dict[str, Any]] = []
    best_routing_global: Dict[str, Any] | None = None
    best_routing_classwise: Dict[str, Any] | None = None
    best_routing_bundle: Dict[str, Any] | None = None

    single_expert_records = {
        expert: best_per_subset[expert]
        for expert in EXPERT_NAMES
        if expert in best_per_subset
    }
    if single_expert_records:
        selected_routing_experts = list(single_expert_records.keys())
        expert_val_per_class_f1 = {
            expert: np.asarray(
                single_expert_records[expert]["classwise"]["val"]["per_class_f1"],
                dtype=np.float32,
            )
            for expert in selected_routing_experts
        }
        slr_val_per_class_f1 = np.asarray(
            slr_classwise["val"]["per_class_f1"],
            dtype=np.float32,
        )

        for routing_mode in routing_modes:
            for routing_gain_floor in routing_gain_floor_list:
                gamma_values = routing_gamma_list if routing_mode in {"top2_soft", "soft_all"} else [0.0]
                for routing_gamma in gamma_values:
                    routing_matrix, routing_assignments = build_classwise_routing_matrix(
                        expert_val_per_class_f1=expert_val_per_class_f1,
                        slr_val_per_class_f1=slr_val_per_class_f1,
                        selected_experts=selected_routing_experts,
                        mode=routing_mode,
                        gamma=float(routing_gamma),
                        gain_floor=float(routing_gain_floor),
                    )
                    routing_summary = {
                        "num_routed_classes": int(np.sum(routing_matrix.sum(axis=1) > 0.0)),
                        "expert_class_counts": {
                            expert: int(np.sum(routing_matrix[:, idx] > 0.0))
                            for idx, expert in enumerate(selected_routing_experts)
                        },
                    }
                    routed_val_verification = aggregate_verification_scores_with_routing(
                        val_expert_match_scores,
                        indexed_templates,
                        routing_matrix,
                        selected_experts=selected_routing_experts,
                    )
                    routed_test_verification = aggregate_verification_scores_with_routing(
                        test_expert_match_scores,
                        indexed_templates,
                        routing_matrix,
                        selected_experts=selected_routing_experts,
                    )

                    for fusion_mode in fusion_modes:
                        if fusion_mode not in {"add", "add_norm"}:
                            raise ValueError(f"Unsupported verification fusion mode: {fusion_mode}")
                        for beta in beta_list:
                            routed_val_logits = apply_topk_rerank_fusion(
                                slr_val_logits,
                                routed_val_verification,
                                topk=int(args.topk),
                                alpha=float(beta),
                                mode=fusion_mode,
                            )
                            routed_test_logits = apply_topk_rerank_fusion(
                                slr_test_logits,
                                routed_test_verification,
                                topk=int(args.topk),
                                alpha=float(beta),
                                mode=fusion_mode,
                            )
                            routed_val_scores = _sigmoid(routed_val_logits)
                            routed_test_scores = _sigmoid(routed_test_logits)

                            global_metrics = evaluate_with_validation_threshold(
                                routed_val_scores,
                                val_base["labels"],
                                routed_test_scores,
                                test_base["labels"],
                                use_inference_strategy=False,
                            )
                            class_thresholds = search_classwise_thresholds(
                                routed_val_scores,
                                val_base["labels"],
                            )
                            classwise_metrics = _evaluate_with_class_thresholds(
                                routed_val_scores,
                                val_base["labels"],
                                routed_test_scores,
                                test_base["labels"],
                                class_thresholds,
                            )

                            row = {
                                "routing_mode": routing_mode,
                                "routing_gamma": float(routing_gamma),
                                "routing_gain_floor": float(routing_gain_floor),
                                "fusion_mode": fusion_mode,
                                "beta": float(beta),
                                "num_routed_classes": int(routing_summary["num_routed_classes"]),
                                "val_macro_global": float(global_metrics["val"]["macro"]),
                                "val_hard_global": float(global_metrics["val"]["hard"]),
                                "test_macro_global": float(global_metrics["test"]["macro"]),
                                "test_hard_global": float(global_metrics["test"]["hard"]),
                                "val_macro_classwise": float(classwise_metrics["val"]["macro"]),
                                "val_hard_classwise": float(classwise_metrics["val"]["hard"]),
                                "test_macro_classwise": float(classwise_metrics["test"]["macro"]),
                                "test_hard_classwise": float(classwise_metrics["test"]["hard"]),
                            }
                            routing_rows.append(row)

                            detail = {
                                "config": {
                                    "routing_mode": routing_mode,
                                    "routing_gamma": float(routing_gamma),
                                    "routing_gain_floor": float(routing_gain_floor),
                                    "fusion_mode": fusion_mode,
                                    "beta": float(beta),
                                    "routing_experts": selected_routing_experts,
                                },
                                "routing_summary": routing_summary,
                                "routing_assignments": routing_assignments,
                                "global": _bundle_summary(global_metrics, include_per_class=True),
                                "classwise": _bundle_summary(classwise_metrics, include_per_class=True),
                            }
                            routing_details.append(detail)

                            if best_routing_global is None or (
                                float(global_metrics["val"]["macro"]),
                                float(global_metrics["test"]["macro"]),
                            ) > (
                                float(best_routing_global["val_macro_global"]),
                                float(best_routing_global["test_macro_global"]),
                            ):
                                best_routing_global = row

                            if best_routing_classwise is None or (
                                float(classwise_metrics["val"]["macro"]),
                                float(classwise_metrics["test"]["macro"]),
                                float(classwise_metrics["test"]["hard"]),
                            ) > (
                                float(best_routing_classwise["val_macro_classwise"]),
                                float(best_routing_classwise["test_macro_classwise"]),
                                float(best_routing_classwise["test_hard_classwise"]),
                            ):
                                best_routing_classwise = row
                                best_routing_bundle = {
                                    "config": detail["config"],
                                    "routing_summary": routing_summary,
                                    "routing_assignments": routing_assignments,
                                    "global": global_metrics,
                                    "classwise": classwise_metrics,
                                }

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_agent_evidence_verification"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "search_results.csv", result_rows)
    _write_csv(output_dir / "routing_search_results.csv", routing_rows)
    (output_dir / "search_results.json").write_text(
        json.dumps(_json_ready(detailed_results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "routing_search_results.json").write_text(
        json.dumps(_json_ready(routing_details), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evidence_templates.json").write_text(
        json.dumps(_json_ready(templates), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "expert_phrase_banks.json").write_text(
        json.dumps(_json_ready(phrase_banks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_count = min(int(args.top_evidence_preview), len(val_clip["image_ids"]))
    preview_rows = top_evidence_rows(
        {expert: scores[:preview_count] for expert, scores in val_expert_phrase_scores.items()},
        phrase_banks,
        val_clip["image_ids"][:preview_count],
        top_m=int(args.bank_topm),
    )
    (output_dir / "val_top_evidence_preview.json").write_text(
        json.dumps(_json_ready(preview_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_global_row = _select_best_row(result_rows, metric_key="val_macro_global")
    best_classwise_row = _select_best_row(result_rows, metric_key="val_macro_classwise")

    best_subset_summaries = {
        subset_name: {
            "config": _json_ready(bundle["config"]),
            "global": _bundle_summary(bundle["global"], include_per_class=True),
            "classwise": _bundle_summary(bundle["classwise"], include_per_class=True),
        }
        for subset_name, bundle in best_per_subset.items()
    }

    expert_dependency_rows: List[Dict[str, Any]] = []
    single_expert_names = [expert for expert in EXPERT_NAMES if expert in best_per_subset]
    if single_expert_names:
        slr_val_per_class = np.asarray(slr_classwise["val"]["per_class_f1"], dtype=np.float32)
        for class_idx, class_name in enumerate(class_names):
            best_expert = max(
                single_expert_names,
                key=lambda expert: float(best_per_subset[expert]["classwise"]["val"]["per_class_f1"][class_idx]),
            )
            best_expert_val = float(best_per_subset[best_expert]["classwise"]["val"]["per_class_f1"][class_idx])
            expert_dependency_rows.append(
                {
                    "class_id": int(class_idx),
                    "class_name": class_name,
                    "best_single_expert": best_expert,
                    "best_single_expert_val_f1": best_expert_val,
                    "slr_val_f1": float(slr_val_per_class[class_idx]),
                    "gain_vs_slr_val": best_expert_val - float(slr_val_per_class[class_idx]),
                }
            )
        _write_csv(output_dir / "expert_dependency.csv", expert_dependency_rows)

    if best_classwise_row is None:
        raise RuntimeError("No MEV search result was produced.")

    best_classwise_bundle = best_per_subset[str(best_classwise_row["subset"])]
    best_classwise_test_per_class = np.asarray(
        best_classwise_bundle["classwise"]["test"]["per_class_f1"],
        dtype=np.float32,
    )
    slr_test_per_class = np.asarray(slr_classwise["test"]["per_class_f1"], dtype=np.float32)
    per_class_gain_rows = class_gain_rows(
        slr_test_per_class,
        best_classwise_test_per_class,
        class_names,
        top_n=min(10, len(class_names)),
    )

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "gemini_file": str(gemini_file),
        "intent2concepts_file": str(intent2concepts_file) if intent2concepts_file is not None else None,
        "slr": {
            "source": slr_source,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "verification": {
            "weight_mode": args.weight_mode,
            "expert_weights": expert_weights,
            "beta_list": beta_list,
            "fusion_modes": fusion_modes,
            "expert_subsets": expert_subsets,
            "bank_source": args.bank_source,
            "match_mode": args.match_mode,
            "positive_only_matching": not bool(args.allow_negative_matching),
        },
        "template_stats": build_template_statistics(templates),
        "bank_stats": build_expert_bank_statistics(phrase_banks),
        "baseline": {
            "global": _bundle_summary(baseline_global, include_per_class=True),
            "classwise": _bundle_summary(baseline_classwise, include_per_class=True),
        },
        "slr_c": {
            "global": _bundle_summary(slr_global, include_per_class=True),
            "classwise": _bundle_summary(slr_classwise, include_per_class=True),
        },
        "best_mev_global": best_global_row,
        "best_mev_classwise": best_classwise_row,
        "best_per_subset": best_subset_summaries,
        "routing_v2": {
            "modes": routing_modes,
            "gamma_list": routing_gamma_list,
            "gain_floor_list": routing_gain_floor_list,
            "best_global": best_routing_global,
            "best_classwise": best_routing_classwise,
            "best_classwise_bundle": None
            if best_routing_bundle is None
            else {
                "config": _json_ready(best_routing_bundle["config"]),
                "routing_summary": _json_ready(best_routing_bundle["routing_summary"]),
                "routing_assignments": _json_ready(best_routing_bundle["routing_assignments"]),
                "global": _bundle_summary(best_routing_bundle["global"], include_per_class=True),
                "classwise": _bundle_summary(best_routing_bundle["classwise"], include_per_class=True),
            },
        },
        "per_class_gains_vs_slr": per_class_gain_rows,
        "expert_dependency": expert_dependency_rows,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
