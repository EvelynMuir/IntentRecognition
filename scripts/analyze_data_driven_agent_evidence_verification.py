#!/usr/bin/env python3
"""Analyze data-driven candidate-to-evidence verification on top of the Intentonomy baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
    build_benchmark_expert_phrase_banks,
    build_expert_bank_statistics,
    build_expert_phrase_banks,
    build_indexed_templates,
    build_intent_evidence_templates,
    build_template_statistics,
    compute_data_driven_verification_scores,
    compute_expert_match_scores_with_similarity,
    compute_expert_phrase_scores,
    encode_expert_phrase_banks,
    encode_template_phrase_sets,
    learn_data_driven_relations,
    resolve_default_intent2concepts_file,
    summarize_data_driven_profiles,
    top_evidence_rows,
)
from src.utils.metrics import SUBSET2IDS
from src.utils.metrics import compute_difficulty_scores
from src.utils.metrics import compute_f1
from src.utils.metrics import compute_mAP
from src.utils.metrics import eval_validation_set
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    build_confusion_pairs,
    class_gain_rows,
    compute_sample_f1_scores,
    evaluate_fixed_threshold,
    evaluate_with_validation_threshold,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data-driven candidate-to-evidence verification experiments for Intentonomy."
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=str(DEFAULT_BASELINE_CKPT) if DEFAULT_BASELINE_CKPT.exists() else None,
    )
    parser.add_argument("--gemini-file", type=str, default=None)
    parser.add_argument("--intent2concepts-file", type=str, default=None)
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
        default="0.05,0.1,0.2,0.3,0.5",
        help="Comma-separated rerank beta list.",
    )
    parser.add_argument(
        "--verification-fusion-modes",
        type=str,
        default="add_norm",
        help="Comma-separated fusion modes. Supported: add,add_norm.",
    )
    parser.add_argument(
        "--expert-subsets",
        type=str,
        default="object,scene,style,activity,all",
        help="Comma-separated expert subsets. Use `all` or `object+scene+style` style tokens.",
    )
    parser.add_argument(
        "--relation-modes",
        type=str,
        default="positive_mean,pos_neg_diff,hard_negative_diff,support_only,support_contradiction",
        help=(
            "Comma-separated relation modes. Supported: "
            "positive_mean,pos_neg_diff,hard_negative_diff,support_only,support_contradiction."
        ),
    )
    parser.add_argument(
        "--profile-topn-list",
        type=str,
        default="5,10,20,all",
        help="Comma-separated profile sparsity list. Use `all` to keep all elements.",
    )
    parser.add_argument(
        "--activation-topm",
        type=int,
        default=5,
        help="Top-m activated elements kept per expert at test time.",
    )
    parser.add_argument(
        "--activation-topm-list",
        type=str,
        default=None,
        help="Optional comma-separated activation top-m search list. Overrides --activation-topm when set.",
    )
    parser.add_argument(
        "--hard-negative-topn",
        type=int,
        default=3,
        help="How many hard negative classes are used in hard_negative_diff.",
    )
    parser.add_argument(
        "--contradiction-lambda-list",
        type=str,
        default="0.5,1.0",
        help="Comma-separated contradiction weights for support_contradiction.",
    )
    parser.add_argument("--bank-encode-batch-size", type=int, default=64)
    parser.add_argument("--max-items-per-expert", type=int, default=8)
    parser.add_argument("--case-limit", type=int, default=5)
    parser.add_argument("--profile-preview-topn", type=int, default=5)
    parser.add_argument("--bank-topm", type=int, default=5)
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


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_profile_topn_list(raw: str) -> List[int | None]:
    output: List[int | None] = []
    for token in raw.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item == "all":
            output.append(None)
        else:
            output.append(int(item))
    return output


def _profile_token(value: int | None) -> str:
    return "all" if value is None else str(int(value))


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


def _build_topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    num_classes = scores.shape[1]
    k = max(1, min(int(k), num_classes))
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(scores, dtype=np.int32)
    row_idx = np.arange(scores.shape[0])[:, None]
    mask[row_idx, topk_idx] = 1
    return mask


def _candidate_recall_stats(scores: np.ndarray, targets: np.ndarray, k: int) -> Dict[str, float]:
    mask = _build_topk_mask(scores, k)
    targets = np.asarray(targets, dtype=np.int32)
    positive_counts = np.maximum(targets.sum(axis=1), 1)
    covered_targets = targets * mask
    return {
        "label_recall": float(covered_targets.sum() / max(targets.sum(), 1)),
        "sample_any_recall": float(np.mean((covered_targets.sum(axis=1) > 0).astype(np.float32))),
        "sample_all_recall": float(np.mean((covered_targets.sum(axis=1) == targets.sum(axis=1)).astype(np.float32))),
        "mean_positive_coverage": float(np.mean(covered_targets.sum(axis=1) / positive_counts)),
    }


def _oracle_multilabel_metrics(scores: np.ndarray, targets: np.ndarray, k: int) -> Dict[str, float]:
    from sklearn.metrics import f1_score

    mask = _build_topk_mask(scores, k)
    oracle_pred = (np.asarray(targets, dtype=np.int32) * mask).astype(np.float32)
    return {
        "macro": float(f1_score(targets, oracle_pred, average="macro", zero_division=0)),
        "micro": float(f1_score(targets, oracle_pred, average="micro", zero_division=0)),
        "samples": float(f1_score(targets, oracle_pred, average="samples", zero_division=0)),
        "coverage_label_recall": float(oracle_pred.sum() / max(np.asarray(targets).sum(), 1)),
    }


def _verification_gap(
    candidate_scores: np.ndarray,
    verification_scores: np.ndarray,
    targets: np.ndarray,
    topk: int,
) -> Dict[str, float]:
    candidate_mask = _build_topk_mask(candidate_scores, topk)
    targets_int = np.asarray(targets, dtype=np.int32)
    positive_mask = candidate_mask == 1
    correct_mask = positive_mask & (targets_int == 1)
    wrong_mask = positive_mask & (targets_int == 0)

    correct_values = np.asarray(verification_scores, dtype=np.float32)[correct_mask]
    wrong_values = np.asarray(verification_scores, dtype=np.float32)[wrong_mask]

    sample_gaps: List[float] = []
    for sample_idx in range(candidate_scores.shape[0]):
        correct_ids = np.where(correct_mask[sample_idx])[0]
        wrong_ids = np.where(wrong_mask[sample_idx])[0]
        if correct_ids.size == 0 or wrong_ids.size == 0:
            continue
        gap = float(
            np.mean(verification_scores[sample_idx, correct_ids])
            - np.mean(verification_scores[sample_idx, wrong_ids])
        )
        sample_gaps.append(gap)

    return {
        "correct_candidate_mean": float(correct_values.mean()) if correct_values.size > 0 else 0.0,
        "wrong_candidate_mean": float(wrong_values.mean()) if wrong_values.size > 0 else 0.0,
        "correct_minus_wrong": (
            float(correct_values.mean() - wrong_values.mean())
            if correct_values.size > 0 and wrong_values.size > 0
            else 0.0
        ),
        "sample_mean_gap": float(np.mean(sample_gaps)) if sample_gaps else 0.0,
        "num_candidate_positive_pairs": int(correct_values.size),
        "num_candidate_negative_pairs": int(wrong_values.size),
        "num_valid_gap_samples": int(len(sample_gaps)),
    }


def _pearson_correlation(
    left: np.ndarray,
    right: np.ndarray,
    topk_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    if topk_mask is not None:
        mask = np.asarray(topk_mask, dtype=np.int32) > 0
        left_arr = left_arr[mask]
        right_arr = right_arr[mask]
    else:
        left_arr = left_arr.reshape(-1)
        right_arr = right_arr.reshape(-1)

    if left_arr.size == 0 or right_arr.size == 0:
        return {"pearson": 0.0, "count": 0}
    if np.std(left_arr) < 1e-8 or np.std(right_arr) < 1e-8:
        return {"pearson": 0.0, "count": int(left_arr.size)}
    return {
        "pearson": float(np.corrcoef(left_arr, right_arr)[0, 1]),
        "count": int(left_arr.size),
    }


def _candidate_rows(
    base_logits: np.ndarray,
    verification: np.ndarray,
    reranked_logits: np.ndarray,
    class_names: Sequence[str],
    k: int,
) -> List[Dict[str, Any]]:
    scores = np.asarray(reranked_logits, dtype=np.float32)
    k = max(1, min(int(k), scores.shape[0]))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    order = idx[np.argsort(-scores[idx])]
    return [
        {
            "class_id": int(class_idx),
            "class_name": str(class_names[int(class_idx)]),
            "base_logit": float(base_logits[int(class_idx)]),
            "verification": float(verification[int(class_idx)]),
            "reranked_logit": float(reranked_logits[int(class_idx)]),
        }
        for class_idx in order.tolist()
    ]


def _prediction_rows(scores: np.ndarray, thresholds: np.ndarray, class_names: Sequence[str]) -> List[str]:
    pred = np.asarray(scores, dtype=np.float32) > np.asarray(thresholds, dtype=np.float32)
    return [str(class_names[idx]) for idx in np.where(pred)[0].tolist()]


def _build_case_studies(
    image_ids: Sequence[str],
    image_id_to_path: Mapping[str, str],
    labels: np.ndarray,
    class_names: Sequence[str],
    slr_scores: np.ndarray,
    slr_logits: np.ndarray,
    slr_thresholds: np.ndarray,
    best_data_record: Mapping[str, Any],
    test_expert_phrase_scores: Mapping[str, np.ndarray],
    phrase_banks: Mapping[str, Sequence[str]],
    case_limit: int,
    topk: int,
) -> List[Dict[str, Any]]:
    data_thresholds = np.asarray(
        best_data_record["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )
    data_scores = np.asarray(best_data_record["test_scores"], dtype=np.float32)
    data_logits = np.asarray(best_data_record["test_logits"], dtype=np.float32)
    data_verification = np.asarray(best_data_record["test_verification"], dtype=np.float32)
    profiles = list(best_data_record["profile_preview"])

    slr_pred = (np.asarray(slr_scores, dtype=np.float32) > np.asarray(slr_thresholds, dtype=np.float32)).astype(np.int32)
    data_pred = (data_scores > data_thresholds).astype(np.int32)
    slr_sample_f1 = compute_sample_f1_scores(labels, slr_pred)
    data_sample_f1 = compute_sample_f1_scores(labels, data_pred)
    delta = data_sample_f1 - slr_sample_f1
    hard_ids = set(SUBSET2IDS["hard"])

    hard_mask = np.asarray(
        [any(int(idx) in hard_ids for idx in np.where(labels[row] > 0)[0].tolist()) for row in range(labels.shape[0])],
        dtype=bool,
    )
    candidate_ids = np.where(hard_mask)[0]
    if candidate_ids.size == 0:
        candidate_ids = np.arange(labels.shape[0])

    order = candidate_ids[np.argsort(-delta[candidate_ids])]
    chosen_ids = order[: max(1, int(case_limit))]

    evidence_preview = top_evidence_rows(
        {expert: scores[chosen_ids] for expert, scores in test_expert_phrase_scores.items()},
        phrase_banks,
        [str(image_ids[idx]) for idx in chosen_ids.tolist()],
        top_m=5,
    )
    evidence_by_image = {row["image_id"]: row for row in evidence_preview}

    rows: List[Dict[str, Any]] = []
    for sample_idx in chosen_ids.tolist():
        target_ids = np.where(labels[sample_idx] > 0)[0].tolist()
        image_id = str(image_ids[sample_idx])
        top_candidates = _candidate_rows(
            base_logits=np.asarray(slr_logits[sample_idx], dtype=np.float32),
            verification=np.asarray(data_verification[sample_idx], dtype=np.float32),
            reranked_logits=np.asarray(data_logits[sample_idx], dtype=np.float32),
            class_names=class_names,
            k=topk,
        )
        profile_rows = []
        for candidate in top_candidates[: min(3, len(top_candidates))]:
            profile_rows.append(
                {
                    "class_id": int(candidate["class_id"]),
                    "class_name": str(candidate["class_name"]),
                    "support_profile": profiles[int(candidate["class_id"])]["support"],
                    "contradiction_profile": profiles[int(candidate["class_id"])]["contradiction"],
                }
            )

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_id_to_path.get(image_id, "")),
                "target_labels": [str(class_names[idx]) for idx in target_ids],
                "slr_predictions": _prediction_rows(slr_scores[sample_idx], slr_thresholds, class_names),
                "data_predictions": _prediction_rows(data_scores[sample_idx], data_thresholds, class_names),
                "slr_sample_f1": float(slr_sample_f1[sample_idx]),
                "data_sample_f1": float(data_sample_f1[sample_idx]),
                "sample_f1_gain": float(delta[sample_idx]),
                "top_candidates": top_candidates,
                "top_elements": evidence_by_image.get(image_id, {}),
                "candidate_profiles": profile_rows,
            }
        )
    return rows


def _select_best_row(rows: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row[metric_key]), float(row["test_macro_classwise"])))


def main() -> None:
    args = _parse_args()
    beta_list = _parse_float_list(args.beta_list)
    fusion_modes = _parse_str_list(args.verification_fusion_modes)
    expert_subsets = _parse_expert_subsets(args.expert_subsets)
    relation_modes = _parse_str_list(args.relation_modes)
    profile_topn_list = _parse_profile_topn_list(args.profile_topn_list)
    contradiction_lambda_list = _parse_float_list(args.contradiction_lambda_list)
    activation_topm_list = (
        _parse_int_list(args.activation_topm_list)
        if args.activation_topm_list is not None
        else [int(args.activation_topm)]
    )

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
    _, test_loader_base, test_image_id_to_path = _build_dataset(
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

    train_clip = _collect_clip_features(clip_model, train_loader_clip, device, max_samples=args.max_samples)
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
        lexical_embeddings = _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True)
        canonical_embeddings = _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True)
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
    baseline_class_thresholds = search_classwise_thresholds(val_base["scores"], val_base["labels"])
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
    slr_class_thresholds = search_classwise_thresholds(slr_val_scores, val_base["labels"])
    slr_classwise = _evaluate_with_class_thresholds(
        slr_val_scores,
        val_base["labels"],
        slr_test_scores,
        test_base["labels"],
        slr_class_thresholds,
    )

    phrase_banks = build_benchmark_expert_phrase_banks(include_activity=True)
    bank_embeddings = encode_expert_phrase_banks(
        clip_model=clip_model,
        phrase_banks=phrase_banks,
        prompt_templates=DEFAULT_EXPERT_PROMPTS,
        batch_size=int(args.bank_encode_batch_size),
    )
    train_expert_phrase_scores = compute_expert_phrase_scores(
        train_clip["features"],
        bank_embeddings,
        clip_logit_scale,
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

    train_binary_labels = (np.asarray(train_clip["soft_labels"], dtype=np.float32) > 0.0).astype(np.float32)

    templates = build_intent_evidence_templates(
        class_names=class_names,
        gemini_file=gemini_file,
        intent2concepts_file=intent2concepts_file,
        max_items_per_expert=int(args.max_items_per_expert),
    )
    template_phrase_embeddings = encode_template_phrase_sets(
        clip_model=clip_model,
        templates=templates,
        prompt_templates=DEFAULT_EXPERT_PROMPTS,
        batch_size=int(args.bank_encode_batch_size),
    )
    template_match_scores_val = compute_expert_match_scores_with_similarity(
        val_expert_phrase_scores,
        bank_embeddings,
        template_phrase_embeddings,
        aggregation_mode="average",
        positive_only=True,
    )
    template_match_scores_test = compute_expert_match_scores_with_similarity(
        test_expert_phrase_scores,
        bank_embeddings,
        template_phrase_embeddings,
        aggregation_mode="average",
        positive_only=True,
    )
    indexed_templates = build_indexed_templates(templates, build_expert_phrase_banks(templates))

    template_rows: List[Dict[str, Any]] = []
    best_template_record: Dict[str, Any] | None = None
    best_template_row: Dict[str, Any] | None = None

    for subset in expert_subsets:
        selected_experts = list(subset["experts"])
        subset_name = str(subset["name"])
        verification_val = aggregate_verification_scores(
            template_match_scores_val,
            indexed_templates,
            selected_experts=selected_experts,
            expert_weights=None,
            weight_mode="template_aware",
        )
        verification_test = aggregate_verification_scores(
            template_match_scores_test,
            indexed_templates,
            selected_experts=selected_experts,
            expert_weights=None,
            weight_mode="template_aware",
        )

        for fusion_mode in fusion_modes:
            if fusion_mode not in {"add", "add_norm"}:
                raise ValueError(f"Unsupported verification fusion mode: {fusion_mode}")
            for beta in beta_list:
                template_val_logits = apply_topk_rerank_fusion(
                    slr_val_logits,
                    verification_val,
                    topk=int(args.topk),
                    alpha=float(beta),
                    mode=fusion_mode,
                )
                template_test_logits = apply_topk_rerank_fusion(
                    slr_test_logits,
                    verification_test,
                    topk=int(args.topk),
                    alpha=float(beta),
                    mode=fusion_mode,
                )
                template_val_scores = _sigmoid(template_val_logits)
                template_test_scores = _sigmoid(template_test_logits)

                global_metrics = evaluate_with_validation_threshold(
                    template_val_scores,
                    val_base["labels"],
                    template_test_scores,
                    test_base["labels"],
                    use_inference_strategy=False,
                )
                class_thresholds = search_classwise_thresholds(template_val_scores, val_base["labels"])
                classwise_metrics = _evaluate_with_class_thresholds(
                    template_val_scores,
                    val_base["labels"],
                    template_test_scores,
                    test_base["labels"],
                    class_thresholds,
                )

                row = {
                    "method": "template_verification",
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
                template_rows.append(row)

                if best_template_record is None or (
                    float(classwise_metrics["val"]["macro"]),
                    float(classwise_metrics["test"]["macro"]),
                    float(classwise_metrics["test"]["hard"]),
                ) > (
                    float(best_template_record["classwise"]["val"]["macro"]),
                    float(best_template_record["classwise"]["test"]["macro"]),
                    float(best_template_record["classwise"]["test"]["hard"]),
                ):
                    best_template_row = row
                    best_template_record = {
                        "config": {
                            "subset": subset_name,
                            "experts": selected_experts,
                            "fusion_mode": fusion_mode,
                            "beta": float(beta),
                        },
                        "global": global_metrics,
                        "classwise": classwise_metrics,
                        "val_scores": template_val_scores.astype(np.float32),
                        "test_scores": template_test_scores.astype(np.float32),
                        "val_logits": template_val_logits.astype(np.float32),
                        "test_logits": template_test_logits.astype(np.float32),
                        "val_verification": verification_val.astype(np.float32),
                        "test_verification": verification_test.astype(np.float32),
                    }

    data_rows: List[Dict[str, Any]] = []
    best_data_record: Dict[str, Any] | None = None
    best_data_row: Dict[str, Any] | None = None
    best_by_relation: Dict[str, Dict[str, Any]] = {}
    best_by_subset: Dict[str, Dict[str, Any]] = {}
    best_by_profile: Dict[str, Dict[str, Any]] = {}

    for subset in expert_subsets:
        selected_experts = list(subset["experts"])
        subset_name = str(subset["name"])
        for relation_mode in relation_modes:
            lambda_values = contradiction_lambda_list if relation_mode == "support_contradiction" else [0.0]
            for profile_topn in profile_topn_list:
                relation_bundle = learn_data_driven_relations(
                    train_expert_phrase_scores,
                    train_binary_labels,
                    selected_experts=selected_experts,
                    relation_mode=relation_mode,
                    profile_topn=profile_topn,
                    hard_negative_topn=int(args.hard_negative_topn),
                    positive_only_scores=True,
                )
                profile_preview = summarize_data_driven_profiles(
                    relation_bundle,
                    phrase_banks,
                    class_names,
                    top_n=int(args.profile_preview_topn),
                )

                for contradiction_lambda in lambda_values:
                    for activation_topm in activation_topm_list:
                        verification_val = compute_data_driven_verification_scores(
                            val_expert_phrase_scores,
                            relation_bundle,
                            selected_experts=selected_experts,
                            activation_topm=int(activation_topm),
                            contradiction_lambda=float(contradiction_lambda),
                            activation_positive_only=True,
                        )
                        verification_test = compute_data_driven_verification_scores(
                            test_expert_phrase_scores,
                            relation_bundle,
                            selected_experts=selected_experts,
                            activation_topm=int(activation_topm),
                            contradiction_lambda=float(contradiction_lambda),
                            activation_positive_only=True,
                        )

                        for fusion_mode in fusion_modes:
                            if fusion_mode not in {"add", "add_norm"}:
                                raise ValueError(f"Unsupported verification fusion mode: {fusion_mode}")
                            for beta in beta_list:
                                data_val_logits = apply_topk_rerank_fusion(
                                    slr_val_logits,
                                    verification_val,
                                    topk=int(args.topk),
                                    alpha=float(beta),
                                    mode=fusion_mode,
                                )
                                data_test_logits = apply_topk_rerank_fusion(
                                    slr_test_logits,
                                    verification_test,
                                    topk=int(args.topk),
                                    alpha=float(beta),
                                    mode=fusion_mode,
                                )
                                data_val_scores = _sigmoid(data_val_logits)
                                data_test_scores = _sigmoid(data_test_logits)

                                global_metrics = evaluate_with_validation_threshold(
                                    data_val_scores,
                                    val_base["labels"],
                                    data_test_scores,
                                    test_base["labels"],
                                    use_inference_strategy=False,
                                )
                                class_thresholds = search_classwise_thresholds(data_val_scores, val_base["labels"])
                                classwise_metrics = _evaluate_with_class_thresholds(
                                    data_val_scores,
                                    val_base["labels"],
                                    data_test_scores,
                                    test_base["labels"],
                                    class_thresholds,
                                )

                                row = {
                                    "method": "data_driven_verification",
                                    "subset": subset_name,
                                    "experts": "+".join(selected_experts),
                                    "relation_mode": relation_mode,
                                    "profile_topn": _profile_token(profile_topn),
                                    "activation_topm": int(activation_topm),
                                    "contradiction_lambda": float(contradiction_lambda),
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
                                data_rows.append(row)

                                relation_key = relation_mode
                                subset_key = subset_name
                                profile_key = _profile_token(profile_topn)
                                for key, bucket in [
                                    (relation_key, best_by_relation),
                                    (subset_key, best_by_subset),
                                    (profile_key, best_by_profile),
                                ]:
                                    current = bucket.get(key)
                                    if current is None or (
                                        float(row["val_macro_classwise"]),
                                        float(row["test_macro_classwise"]),
                                    ) > (
                                        float(current["val_macro_classwise"]),
                                        float(current["test_macro_classwise"]),
                                    ):
                                        bucket[key] = row

                                if best_data_record is None or (
                                    float(classwise_metrics["val"]["macro"]),
                                    float(classwise_metrics["test"]["macro"]),
                                    float(classwise_metrics["test"]["hard"]),
                                ) > (
                                    float(best_data_record["classwise"]["val"]["macro"]),
                                    float(best_data_record["classwise"]["test"]["macro"]),
                                    float(best_data_record["classwise"]["test"]["hard"]),
                                ):
                                    best_data_row = row
                                    best_data_record = {
                                        "config": {
                                            "subset": subset_name,
                                            "experts": selected_experts,
                                            "relation_mode": relation_mode,
                                            "profile_topn": profile_topn,
                                            "activation_topm": int(activation_topm),
                                            "contradiction_lambda": float(contradiction_lambda),
                                            "fusion_mode": fusion_mode,
                                            "beta": float(beta),
                                        },
                                        "global": global_metrics,
                                        "classwise": classwise_metrics,
                                        "val_scores": data_val_scores.astype(np.float32),
                                        "test_scores": data_test_scores.astype(np.float32),
                                        "val_logits": data_val_logits.astype(np.float32),
                                        "test_logits": data_test_logits.astype(np.float32),
                                        "val_verification": verification_val.astype(np.float32),
                                        "test_verification": verification_test.astype(np.float32),
                                        "profile_preview": profile_preview,
                                        "hard_negative_ids": relation_bundle["hard_negative_ids"],
                                    }

    if best_template_record is None or best_template_row is None:
        raise RuntimeError("No template verification result was produced.")
    if best_data_record is None or best_data_row is None:
        raise RuntimeError("No data-driven verification result was produced.")

    candidate_recall = _candidate_recall_stats(slr_test_logits, test_base["labels"], int(args.topk))
    oracle_topk = _oracle_multilabel_metrics(slr_test_logits, test_base["labels"], int(args.topk))
    candidate_mask = _build_topk_mask(slr_test_logits, int(args.topk))

    diagnostics = {
        "candidate_recall_topk": candidate_recall,
        "oracle_topk_upper_bound": oracle_topk,
        "verification_gap": {
            "template": _verification_gap(
                slr_test_logits,
                np.asarray(best_template_record["test_verification"], dtype=np.float32),
                test_base["labels"],
                int(args.topk),
            ),
            "data_driven": _verification_gap(
                slr_test_logits,
                np.asarray(best_data_record["test_verification"], dtype=np.float32),
                test_base["labels"],
                int(args.topk),
            ),
        },
        "correlation": {
            "template_all": _pearson_correlation(
                slr_test_logits,
                np.asarray(best_template_record["test_verification"], dtype=np.float32),
                topk_mask=None,
            ),
            "template_topk": _pearson_correlation(
                slr_test_logits,
                np.asarray(best_template_record["test_verification"], dtype=np.float32),
                topk_mask=candidate_mask,
            ),
            "data_driven_all": _pearson_correlation(
                slr_test_logits,
                np.asarray(best_data_record["test_verification"], dtype=np.float32),
                topk_mask=None,
            ),
            "data_driven_topk": _pearson_correlation(
                slr_test_logits,
                np.asarray(best_data_record["test_verification"], dtype=np.float32),
                topk_mask=candidate_mask,
            ),
        },
    }

    best_data_test_per_class = np.asarray(best_data_record["classwise"]["test"]["per_class_f1"], dtype=np.float32)
    slr_test_per_class = np.asarray(slr_classwise["test"]["per_class_f1"], dtype=np.float32)
    per_class_gain_rows = class_gain_rows(
        slr_test_per_class,
        best_data_test_per_class,
        class_names,
        top_n=min(10, len(class_names)),
    )

    slr_predictions = (slr_test_scores > np.asarray(slr_classwise["val"]["class_thresholds"], dtype=np.float32)).astype(np.int32)
    data_predictions = (
        np.asarray(best_data_record["test_scores"], dtype=np.float32)
        > np.asarray(best_data_record["classwise"]["val"]["class_thresholds"], dtype=np.float32)
    ).astype(np.int32)
    confusion_shift = build_confusion_pairs(
        np.asarray(test_base["labels"], dtype=np.int32),
        data_predictions,
        class_names,
        top_n=10,
    )

    case_studies = _build_case_studies(
        image_ids=test_clip["image_ids"],
        image_id_to_path=test_image_id_to_path,
        labels=np.asarray(test_base["labels"], dtype=np.int32),
        class_names=class_names,
        slr_scores=np.asarray(slr_test_scores, dtype=np.float32),
        slr_logits=np.asarray(slr_test_logits, dtype=np.float32),
        slr_thresholds=np.asarray(slr_classwise["val"]["class_thresholds"], dtype=np.float32),
        best_data_record=best_data_record,
        test_expert_phrase_scores=test_expert_phrase_scores,
        phrase_banks=phrase_banks,
        case_limit=int(args.case_limit),
        topk=int(args.topk),
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_data_driven_agent_evidence_verification"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "template_search_results.csv", template_rows)
    _write_csv(output_dir / "data_driven_search_results.csv", data_rows)
    (output_dir / "evidence_templates.json").write_text(
        json.dumps(_json_ready(templates), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "expert_phrase_banks.json").write_text(
        json.dumps(_json_ready(phrase_banks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "data_driven_profile_preview.json").write_text(
        json.dumps(_json_ready(best_data_record["profile_preview"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(_json_ready(diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "case_studies.json").write_text(
        json.dumps(_json_ready(case_studies), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "confusion_shift.json").write_text(
        json.dumps(_json_ready(confusion_shift), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_class_gain_rows.json").write_text(
        json.dumps(_json_ready(per_class_gain_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
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
            "beta_list": beta_list,
            "fusion_modes": fusion_modes,
            "expert_subsets": expert_subsets,
            "relation_modes": relation_modes,
            "profile_topn_list": [_profile_token(item) for item in profile_topn_list],
            "activation_topm_list": [int(item) for item in activation_topm_list],
            "hard_negative_topn": int(args.hard_negative_topn),
            "contradiction_lambda_list": contradiction_lambda_list,
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
        "template_verification": {
            "best_row": best_template_row,
            "best_bundle": {
                "config": _json_ready(best_template_record["config"]),
                "global": _bundle_summary(best_template_record["global"], include_per_class=True),
                "classwise": _bundle_summary(best_template_record["classwise"], include_per_class=True),
            },
        },
        "data_driven_verification": {
            "best_row": best_data_row,
            "best_bundle": {
                "config": _json_ready(best_data_record["config"]),
                "global": _bundle_summary(best_data_record["global"], include_per_class=True),
                "classwise": _bundle_summary(best_data_record["classwise"], include_per_class=True),
            },
            "best_by_relation": _json_ready(best_by_relation),
            "best_by_subset": _json_ready(best_by_subset),
            "best_by_profile_topn": _json_ready(best_by_profile),
        },
        "diagnostics": diagnostics,
        "per_class_gains_vs_slr": per_class_gain_rows,
        "confusion_shift": confusion_shift,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
