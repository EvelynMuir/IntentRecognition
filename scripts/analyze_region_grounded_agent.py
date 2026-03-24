#!/usr/bin/env python3
"""Analyze a conservative region-grounded belief update MVP on top of fixed scenario SLR-C."""

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
import torch.nn.functional as F
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
    build_benchmark_expert_phrase_banks,
    build_confusion_neighborhoods,
    build_margin_aware_gate,
    build_pairwise_relation_profiles,
    compute_expert_phrase_scores,
    compute_pairwise_comparative_scores,
    encode_expert_phrase_banks,
    learn_data_driven_relations,
)
from src.utils.region_grounded_reasoning import (
    apply_soft_routing,
    build_expert_stack,
    compute_candidate_class_evidence_scores,
    compute_candidate_phrase_scores,
    compute_candidate_region_summaries,
    compute_class_evidence_scores,
    compute_soft_routing_weights,
    gather_topk_values,
    normalize_topk_candidate_matrix,
    scatter_candidate_values,
)
from src.utils.text_prior_analysis import apply_topk_rerank_fusion


DEFAULT_REUSE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
EXPERT_ORDER = ["object", "activity", "scene", "style"]


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
        description="Run a conservative Region-Grounded Agent Phase 1 MVP with optional Phase 2 soft routing."
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
    parser.add_argument("--region-cache-dir", type=str, default=None)
    parser.add_argument("--bank-encode-batch-size", type=int, default=64)
    parser.add_argument("--profile-topn", type=int, default=20)
    parser.add_argument("--activation-topm", type=int, default=5)
    parser.add_argument("--hard-negative-topn", type=int, default=3)
    parser.add_argument("--confusion-topk", type=int, default=10)
    parser.add_argument("--subset-margin-tau", type=float, default=1.0)
    parser.add_argument("--v2-pair-profile-topn", type=int, default=5)
    parser.add_argument("--v2-gate-gamma", type=float, default=2.0)
    parser.add_argument("--v2-beta", type=float, default=0.01)
    parser.add_argument("--region-attn-scale-list", type=str, default="5,10,20")
    parser.add_argument("--phase1-local-weight-list", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--phase1-global-weight-list", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--phase1-beta-list", type=str, default="0.005,0.01,0.02,0.05")
    parser.add_argument("--phase1-gate-gamma-list", type=str, default="0,2")
    parser.add_argument("--phase2-temperature-list", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--phase2-beta-list", type=str, default="0.005,0.01,0.02,0.05")
    parser.add_argument("--phase2-gate-gamma-list", type=str, default="0,2")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _resolve_runtime_device(device_arg: str) -> torch.device:
    resolved = _resolve_device(device_arg)
    if resolved.type != "cuda":
        return resolved
    try:
        torch.empty(1, device=resolved)
        return resolved
    except Exception:
        return torch.device("cpu")


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_region_grounded_agent"
    return Path(output_dir_arg)


def _required_base_cache_files(cache_dir: Path) -> List[Path]:
    return [
        cache_dir / "train_base.npz",
        cache_dir / "train_clip.npz",
        cache_dir / "val_base.npz",
        cache_dir / "val_clip.npz",
        cache_dir / "test_base.npz",
        cache_dir / "test_clip.npz",
    ]


def _cache_is_ready(cache_dir: Path) -> bool:
    return all(path.exists() for path in _required_base_cache_files(cache_dir))


def _region_cache_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}_clip_region.npz"


def _save_region_cache(path: Path, bundle: Mapping[str, Any]) -> None:
    np.savez(
        path,
        features=np.asarray(bundle["features"], dtype=np.float32),
        patch_tokens=np.asarray(bundle["patch_tokens"], dtype=np.float16),
        image_ids=np.asarray(bundle["image_ids"]),
    )


def _load_region_cache(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "features": np.asarray(data["features"], dtype=np.float32),
        "patch_tokens": np.asarray(data["patch_tokens"]),
        "image_ids": [str(item) for item in data["image_ids"].tolist()],
    }


def _extract_projected_features_and_tokens(
    clip_model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = clip_model.visual

    x = backbone.conv1(images)
    batch_size, channels, height, width = x.shape
    x = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)

    class_token = backbone.class_embedding.to(dtype=x.dtype, device=x.device)
    class_token = class_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
    x = torch.cat([class_token, x], dim=1)
    x = x + backbone.positional_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(0)
    x = backbone.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = backbone.transformer(x)
    x = x.permute(1, 0, 2)
    if hasattr(backbone, "ln_post"):
        x = backbone.ln_post(x)

    cls_token = x[:, 0:1, :]
    patch_tokens = x[:, 1:, :]
    if getattr(backbone, "proj", None) is not None:
        cls_token = cls_token @ backbone.proj
        patch_tokens = patch_tokens @ backbone.proj

    global_features = F.normalize(cls_token[:, 0, :].float(), dim=-1)
    return global_features, patch_tokens.float()


def _collect_clip_region_features(
    clip_model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    features_all: List[np.ndarray] = []
    patch_tokens_all: List[np.ndarray] = []
    image_ids_all: List[str] = []

    collected = 0
    clip_model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            global_features, patch_tokens = _extract_projected_features_and_tokens(clip_model, images)

            feats_cpu = global_features.detach().cpu()
            patch_cpu = patch_tokens.detach().cpu()
            image_ids = batch["image_id"]
            if torch.is_tensor(image_ids):
                image_ids_batch = [str(item) for item in image_ids.detach().cpu().tolist()]
            else:
                image_ids_batch = [str(item) for item in image_ids]

            if max_samples is not None:
                remaining = max_samples - collected
                if remaining <= 0:
                    break
                if feats_cpu.shape[0] > remaining:
                    feats_cpu = feats_cpu[:remaining]
                    patch_cpu = patch_cpu[:remaining]
                    image_ids_batch = image_ids_batch[:remaining]

            features_all.append(feats_cpu.numpy().astype(np.float32))
            patch_tokens_all.append(patch_cpu.numpy().astype(np.float16))
            image_ids_all.extend(image_ids_batch)
            collected += len(image_ids_batch)
            if max_samples is not None and collected >= max_samples:
                break

    return {
        "features": np.concatenate(features_all, axis=0),
        "patch_tokens": np.concatenate(patch_tokens_all, axis=0),
        "image_ids": image_ids_all,
    }


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


def _top2_confusion_mask(
    candidate_logits: np.ndarray,
    confusion_neighborhoods: Sequence[Sequence[int]],
) -> np.ndarray:
    top2 = _ordered_topk_indices(candidate_logits, topk=2)
    mask = np.zeros(top2.shape[0], dtype=np.bool_)
    for sample_idx in range(top2.shape[0]):
        class_i = int(top2[sample_idx, 0])
        class_j = int(top2[sample_idx, 1])
        if class_j in confusion_neighborhoods[class_i] or class_i in confusion_neighborhoods[class_j]:
            mask[sample_idx] = True
    return mask


def _phase_key(bundle: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = bundle["classwise"]["test"]
    return (
        float(bundle["classwise"]["val"]["macro"]),
        float(metrics["macro"]),
        float(metrics["hard"]),
    )


def _select_best(
    current_best: Dict[str, Any] | None,
    candidate_row: Dict[str, Any],
    candidate_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    if current_best is None:
        return {
            "row": candidate_row,
            "bundle": candidate_bundle,
        }
    current_key = _phase_key(current_best["bundle"])
    candidate_key = _phase_key(candidate_bundle)
    if candidate_key > current_key:
        return {
            "row": candidate_row,
            "bundle": candidate_bundle,
        }
    return current_best


def _group_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32))


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


def _channel_from_global_scores(
    expert_phrase_scores: np.ndarray,
    support_matrix: np.ndarray,
    contradiction_matrix: np.ndarray,
    candidate_indices: np.ndarray,
    activation_topm: int,
    num_classes: int,
) -> np.ndarray:
    class_evidence = compute_class_evidence_scores(
        expert_phrase_scores,
        support_matrix,
        contradiction_matrix=contradiction_matrix,
        topm=activation_topm,
        positive_only=True,
    )
    return scatter_candidate_values(
        gather_topk_values(class_evidence, candidate_indices),
        candidate_indices,
        num_classes=num_classes,
    )


def _local_region_channels(
    patch_tokens: np.ndarray,
    candidate_indices: np.ndarray,
    query_embeddings: np.ndarray,
    bank_embeddings: Mapping[str, np.ndarray],
    relation_bundle: Mapping[str, Any],
    attn_scale: float,
    activation_topm: int,
    num_classes: int,
    clip_logit_scale: float,
) -> Dict[str, Any]:
    outputs = compute_candidate_region_summaries(
        patch_tokens,
        candidate_indices,
        query_embeddings,
        attn_logit_scale=float(attn_scale),
        return_attention=False,
    )
    summaries = outputs["summaries"]
    channels: Dict[str, np.ndarray] = {}
    for expert in ["object", "activity"]:
        candidate_phrase_scores = compute_candidate_phrase_scores(
            summaries,
            np.asarray(bank_embeddings[expert], dtype=np.float32),
            logit_scale=clip_logit_scale,
        )
        channels[expert] = compute_candidate_class_evidence_scores(
            candidate_phrase_scores,
            candidate_indices,
            np.asarray(relation_bundle["support"][expert], dtype=np.float32),
            contradiction_matrix=np.asarray(relation_bundle["contradiction"][expert], dtype=np.float32),
            topm=activation_topm,
            positive_only=True,
            num_classes=num_classes,
        )
    channels["attention_entropy"] = np.asarray(outputs["attention_entropy"], dtype=np.float32)
    return channels


def _routing_stats(routing_weights: np.ndarray) -> Dict[str, Any]:
    weights = np.asarray(routing_weights, dtype=np.float32)
    avg_weights = weights.mean(axis=(0, 1))
    entropy = (-weights * np.log(np.maximum(weights, 1e-8))).sum(axis=-1).mean()
    dominant = np.argmax(weights, axis=-1).reshape(-1)
    dominant_hist = {
        expert: int(np.sum(dominant == expert_idx))
        for expert_idx, expert in enumerate(EXPERT_ORDER)
    }
    return {
        "avg_weights": {expert: float(avg_weights[idx]) for idx, expert in enumerate(EXPERT_ORDER)},
        "mean_entropy": float(entropy),
        "dominant_histogram": dominant_hist,
    }


def main() -> None:
    args = _parse_args()
    region_attn_scale_list = _parse_float_list(args.region_attn_scale_list)
    phase1_local_weight_list = _parse_float_list(args.phase1_local_weight_list)
    phase1_global_weight_list = _parse_float_list(args.phase1_global_weight_list)
    phase1_beta_list = _parse_float_list(args.phase1_beta_list)
    phase1_gate_gamma_list = _parse_float_list(args.phase1_gate_gamma_list)
    phase2_temperature_list = _parse_float_list(args.phase2_temperature_list)
    phase2_beta_list = _parse_float_list(args.phase2_beta_list)
    phase2_gate_gamma_list = _parse_float_list(args.phase2_gate_gamma_list)

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache_dir = Path(args.reuse_cache_dir) if args.reuse_cache_dir is not None else output_dir / "_cache"
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    region_cache_dir = Path(args.region_cache_dir) if args.region_cache_dir is not None else output_dir / "_region_cache"
    region_cache_dir.mkdir(parents=True, exist_ok=True)

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
    num_classes = len(class_names)
    batch_size = int(getattr(cfg.data, "batch_size", 64))
    device = _resolve_runtime_device(args.device)

    datamodule = None
    clip_preprocess = None
    clip_model = None

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

    if clip_model is None or clip_preprocess is None:
        raise RuntimeError("CLIP model and preprocess must be available for region feature extraction.")

    for split in ["val", "test"]:
        cache_path = _region_cache_path(region_cache_dir, split)
        if cache_path.exists():
            continue
        if datamodule is None:
            datamodule = instantiate(cfg.data)
        _, loader_clip, _ = _build_dataset(
            cfg,
            datamodule,
            split,
            clip_preprocess,
            batch_size,
            args.num_workers,
            args.pin_memory,
        )
        bundle = _collect_clip_region_features(
            clip_model,
            loader_clip,
            device,
            max_samples=args.max_samples,
        )
        _save_region_cache(cache_path, bundle)

    val_region = _load_region_cache(_region_cache_path(region_cache_dir, "val"))
    test_region = _load_region_cache(_region_cache_path(region_cache_dir, "test"))
    if val_region["image_ids"] != val_clip["image_ids"]:
        raise RuntimeError("Validation image order mismatch between CLIP global cache and region cache.")
    if test_region["image_ids"] != test_clip["image_ids"]:
        raise RuntimeError("Test image order mismatch between CLIP global cache and region cache.")

    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    val_region_feature_diff = float(
        np.max(np.abs(np.asarray(val_region["features"], dtype=np.float32) - np.asarray(val_clip["features"], dtype=np.float32)))
    )
    test_region_feature_diff = float(
        np.max(np.abs(np.asarray(test_region["features"], dtype=np.float32) - np.asarray(test_clip["features"], dtype=np.float32)))
    )

    text_pools = _build_text_pools(class_names, gemini_file)
    scenario_text_embeddings = _encode_text_pool(
        clip_model,
        text_pools["scenario"],
        wrap_prompt=False,
    )
    train_prior_logits = _text_logits_from_features(train_clip["features"], scenario_text_embeddings, clip_logit_scale)
    val_prior_logits = _text_logits_from_features(val_clip["features"], scenario_text_embeddings, clip_logit_scale)
    test_prior_logits = _text_logits_from_features(test_clip["features"], scenario_text_embeddings, clip_logit_scale)

    baseline_eval = _evaluate_score_bundle(
        np.asarray(val_base["scores"], dtype=np.float32),
        np.asarray(val_base["labels"], dtype=np.float32),
        np.asarray(test_base["scores"], dtype=np.float32),
        np.asarray(test_base["labels"], dtype=np.float32),
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
        np.asarray(val_base["labels"], dtype=np.float32),
        slr_test_scores,
        np.asarray(test_base["labels"], dtype=np.float32),
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

    relation_bundle = learn_data_driven_relations(
        train_expert_phrase_scores,
        train_binary_labels,
        selected_experts=EXPERT_ORDER,
        relation_mode="hard_negative_diff",
        profile_topn=int(args.profile_topn),
        hard_negative_topn=int(args.hard_negative_topn),
        hard_negative_ids=None,
        positive_only_scores=True,
    )

    v2_pair_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=EXPERT_ORDER,
        pair_profile_topn=int(args.v2_pair_profile_topn),
        contradiction_lambda=0.0,
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
    v2_pairwise_val = compute_pairwise_comparative_scores(
        val_expert_phrase_scores,
        v2_pair_profiles,
        slr_val_logits,
        selected_experts=EXPERT_ORDER,
        candidate_topk=int(args.topk),
        activation_topm=int(args.activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )
    v2_pairwise_test = compute_pairwise_comparative_scores(
        test_expert_phrase_scores,
        v2_pair_profiles,
        slr_test_logits,
        selected_experts=EXPERT_ORDER,
        candidate_topk=int(args.topk),
        activation_topm=int(args.activation_topm),
        activation_positive_only=True,
        aggregate_mode="mean",
    )
    v2_val_logits = _combine_with_gate(
        slr_val_logits,
        v2_pairwise_val,
        v2_gate_val,
        topk=int(args.topk),
        beta=float(args.v2_beta),
    )
    v2_test_logits = _combine_with_gate(
        slr_test_logits,
        v2_pairwise_test,
        v2_gate_test,
        topk=int(args.topk),
        beta=float(args.v2_beta),
    )
    v2_val_scores = _sigmoid(v2_val_logits)
    v2_test_scores = _sigmoid(v2_test_logits)
    v2_eval = _evaluate_score_bundle(
        v2_val_scores,
        np.asarray(val_base["labels"], dtype=np.float32),
        v2_test_scores,
        np.asarray(test_base["labels"], dtype=np.float32),
    )

    candidate_topk_val = _ordered_topk_indices(slr_val_logits, topk=int(args.topk))
    candidate_topk_test = _ordered_topk_indices(slr_test_logits, topk=int(args.topk))

    phase_gate_cache = {
        float(gamma): {
            "val": build_margin_aware_gate(
                slr_val_logits,
                mode="exp" if float(gamma) > 0.0 else "none",
                gamma=float(gamma),
            ),
            "test": build_margin_aware_gate(
                slr_test_logits,
                mode="exp" if float(gamma) > 0.0 else "none",
                gamma=float(gamma),
            ),
        }
        for gamma in sorted({*phase1_gate_gamma_list, *phase2_gate_gamma_list, float(args.v2_gate_gamma)})
    }

    global_channels_raw: Dict[str, Dict[str, np.ndarray]] = {"val": {}, "test": {}}
    for expert in EXPERT_ORDER:
        global_channels_raw["val"][expert] = _channel_from_global_scores(
            np.asarray(val_expert_phrase_scores[expert], dtype=np.float32),
            np.asarray(relation_bundle["support"][expert], dtype=np.float32),
            np.asarray(relation_bundle["contradiction"][expert], dtype=np.float32),
            candidate_topk_val,
            activation_topm=int(args.activation_topm),
            num_classes=num_classes,
        )
        global_channels_raw["test"][expert] = _channel_from_global_scores(
            np.asarray(test_expert_phrase_scores[expert], dtype=np.float32),
            np.asarray(relation_bundle["support"][expert], dtype=np.float32),
            np.asarray(relation_bundle["contradiction"][expert], dtype=np.float32),
            candidate_topk_test,
            activation_topm=int(args.activation_topm),
            num_classes=num_classes,
        )

    global_channels_norm = {
        split: {
            expert: normalize_topk_candidate_matrix(matrix, candidate_topk_val if split == "val" else candidate_topk_test)
            for expert, matrix in channel_map.items()
        }
        for split, channel_map in global_channels_raw.items()
    }

    local_channels_by_scale: Dict[float, Dict[str, Any]] = {}
    for attn_scale in region_attn_scale_list:
        val_local = _local_region_channels(
            np.asarray(val_region["patch_tokens"], dtype=np.float32),
            candidate_topk_val,
            np.asarray(scenario_text_embeddings, dtype=np.float32),
            bank_embeddings=bank_embeddings,
            relation_bundle=relation_bundle,
            attn_scale=float(attn_scale),
            activation_topm=int(args.activation_topm),
            num_classes=num_classes,
            clip_logit_scale=clip_logit_scale,
        )
        test_local = _local_region_channels(
            np.asarray(test_region["patch_tokens"], dtype=np.float32),
            candidate_topk_test,
            np.asarray(scenario_text_embeddings, dtype=np.float32),
            bank_embeddings=bank_embeddings,
            relation_bundle=relation_bundle,
            attn_scale=float(attn_scale),
            activation_topm=int(args.activation_topm),
            num_classes=num_classes,
            clip_logit_scale=clip_logit_scale,
        )
        local_channels_by_scale[float(attn_scale)] = {
            "val_raw": {expert: np.asarray(val_local[expert], dtype=np.float32) for expert in ["object", "activity"]},
            "test_raw": {expert: np.asarray(test_local[expert], dtype=np.float32) for expert in ["object", "activity"]},
            "val_norm": {
                expert: normalize_topk_candidate_matrix(np.asarray(val_local[expert], dtype=np.float32), candidate_topk_val)
                for expert in ["object", "activity"]
            },
            "test_norm": {
                expert: normalize_topk_candidate_matrix(np.asarray(test_local[expert], dtype=np.float32), candidate_topk_test)
                for expert in ["object", "activity"]
            },
            "val_attention_entropy": float(np.asarray(val_local["attention_entropy"], dtype=np.float32).mean()),
            "test_attention_entropy": float(np.asarray(test_local["attention_entropy"], dtype=np.float32).mean()),
        }

    phase1_rows: List[Dict[str, Any]] = []
    phase1_best_by_setup: Dict[str, Dict[str, Any]] = {}
    phase1_best_hybrid: Dict[str, Any] | None = None

    scene_style_val = _group_average(global_channels_norm["val"]["scene"], global_channels_norm["val"]["style"])
    scene_style_test = _group_average(global_channels_norm["test"]["scene"], global_channels_norm["test"]["style"])
    global_obj_act_val = _group_average(global_channels_norm["val"]["object"], global_channels_norm["val"]["activity"])
    global_obj_act_test = _group_average(global_channels_norm["test"]["object"], global_channels_norm["test"]["activity"])

    for global_weight in phase1_global_weight_list:
        for beta in phase1_beta_list:
            for gate_gamma in phase1_gate_gamma_list:
                val_residual = float(global_weight) * scene_style_val
                test_residual = float(global_weight) * scene_style_test
                val_logits = _combine_with_gate(
                    slr_val_logits,
                    val_residual,
                    phase_gate_cache[float(gate_gamma)]["val"],
                    topk=int(args.topk),
                    beta=float(beta),
                )
                test_logits = _combine_with_gate(
                    slr_test_logits,
                    test_residual,
                    phase_gate_cache[float(gate_gamma)]["test"],
                    topk=int(args.topk),
                    beta=float(beta),
                )
                val_scores = _sigmoid(val_logits)
                test_scores = _sigmoid(test_logits)
                bundle = _evaluate_score_bundle(val_scores, val_base["labels"], test_scores, test_base["labels"])
                row = {
                    "setup": "scene_style_global_only",
                    "region_attn_scale": "none",
                    "local_weight": 0.0,
                    "global_weight": float(global_weight),
                    "gate_gamma": float(gate_gamma),
                    "beta": float(beta),
                    "val_macro_classwise": float(bundle["classwise"]["val"]["macro"]),
                    "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                    "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                }
                phase1_rows.append(row)
                phase1_best_by_setup["scene_style_global_only"] = _select_best(
                    phase1_best_by_setup.get("scene_style_global_only"),
                    row,
                    {
                        **bundle,
                        "val_logits": val_logits.astype(np.float32),
                        "test_logits": test_logits.astype(np.float32),
                        "val_scores": val_scores.astype(np.float32),
                        "test_scores": test_scores.astype(np.float32),
                    },
                )

    for local_weight in phase1_local_weight_list:
        for global_weight in phase1_global_weight_list:
            for beta in phase1_beta_list:
                for gate_gamma in phase1_gate_gamma_list:
                    val_residual = float(local_weight) * global_obj_act_val + float(global_weight) * scene_style_val
                    test_residual = float(local_weight) * global_obj_act_test + float(global_weight) * scene_style_test
                    val_logits = _combine_with_gate(
                        slr_val_logits,
                        val_residual,
                        phase_gate_cache[float(gate_gamma)]["val"],
                        topk=int(args.topk),
                        beta=float(beta),
                    )
                    test_logits = _combine_with_gate(
                        slr_test_logits,
                        test_residual,
                        phase_gate_cache[float(gate_gamma)]["test"],
                        topk=int(args.topk),
                        beta=float(beta),
                    )
                    val_scores = _sigmoid(val_logits)
                    test_scores = _sigmoid(test_logits)
                    bundle = _evaluate_score_bundle(val_scores, val_base["labels"], test_scores, test_base["labels"])
                    row = {
                        "setup": "all_global",
                        "region_attn_scale": "none",
                        "local_weight": float(local_weight),
                        "global_weight": float(global_weight),
                        "gate_gamma": float(gate_gamma),
                        "beta": float(beta),
                        "val_macro_classwise": float(bundle["classwise"]["val"]["macro"]),
                        "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                        "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                    }
                    phase1_rows.append(row)
                    phase1_best_by_setup["all_global"] = _select_best(
                        phase1_best_by_setup.get("all_global"),
                        row,
                        {
                            **bundle,
                            "val_logits": val_logits.astype(np.float32),
                            "test_logits": test_logits.astype(np.float32),
                            "val_scores": val_scores.astype(np.float32),
                            "test_scores": test_scores.astype(np.float32),
                        },
                    )

    for attn_scale in region_attn_scale_list:
        hybrid_val = _group_average(
            local_channels_by_scale[float(attn_scale)]["val_norm"]["object"],
            local_channels_by_scale[float(attn_scale)]["val_norm"]["activity"],
        )
        hybrid_test = _group_average(
            local_channels_by_scale[float(attn_scale)]["test_norm"]["object"],
            local_channels_by_scale[float(attn_scale)]["test_norm"]["activity"],
        )
        for local_weight in phase1_local_weight_list:
            for global_weight in phase1_global_weight_list:
                for beta in phase1_beta_list:
                    for gate_gamma in phase1_gate_gamma_list:
                        val_residual = float(local_weight) * hybrid_val + float(global_weight) * scene_style_val
                        test_residual = float(local_weight) * hybrid_test + float(global_weight) * scene_style_test
                        val_logits = _combine_with_gate(
                            slr_val_logits,
                            val_residual,
                            phase_gate_cache[float(gate_gamma)]["val"],
                            topk=int(args.topk),
                            beta=float(beta),
                        )
                        test_logits = _combine_with_gate(
                            slr_test_logits,
                            test_residual,
                            phase_gate_cache[float(gate_gamma)]["test"],
                            topk=int(args.topk),
                            beta=float(beta),
                        )
                        val_scores = _sigmoid(val_logits)
                        test_scores = _sigmoid(test_logits)
                        bundle = _evaluate_score_bundle(val_scores, val_base["labels"], test_scores, test_base["labels"])
                        row = {
                            "setup": "region_obj_act_plus_global_scene_style",
                            "region_attn_scale": float(attn_scale),
                            "local_weight": float(local_weight),
                            "global_weight": float(global_weight),
                            "gate_gamma": float(gate_gamma),
                            "beta": float(beta),
                            "val_macro_classwise": float(bundle["classwise"]["val"]["macro"]),
                            "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                            "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                        }
                        phase1_rows.append(row)
                        selected = _select_best(
                            phase1_best_by_setup.get("region_obj_act_plus_global_scene_style"),
                            row,
                            {
                                **bundle,
                                "val_logits": val_logits.astype(np.float32),
                                "test_logits": test_logits.astype(np.float32),
                                "val_scores": val_scores.astype(np.float32),
                                "test_scores": test_scores.astype(np.float32),
                            },
                        )
                        phase1_best_by_setup["region_obj_act_plus_global_scene_style"] = selected
                        if selected["row"] == row:
                            selected["attention"] = {
                                "val_entropy_mean": local_channels_by_scale[float(attn_scale)]["val_attention_entropy"],
                                "test_entropy_mean": local_channels_by_scale[float(attn_scale)]["test_attention_entropy"],
                            }
                        phase1_best_hybrid = _select_best(
                            phase1_best_hybrid,
                            row,
                            {
                                **bundle,
                                "val_logits": val_logits.astype(np.float32),
                                "test_logits": test_logits.astype(np.float32),
                                "val_scores": val_scores.astype(np.float32),
                                "test_scores": test_scores.astype(np.float32),
                            },
                        )
                        if phase1_best_hybrid["row"] == row:
                            phase1_best_hybrid["attention"] = {
                                "val_entropy_mean": local_channels_by_scale[float(attn_scale)]["val_attention_entropy"],
                                "test_entropy_mean": local_channels_by_scale[float(attn_scale)]["test_attention_entropy"],
                            }

    if phase1_best_hybrid is None:
        raise RuntimeError("No Phase 1 hybrid result was produced.")

    phase2_best: Dict[str, Any] | None = None
    phase2_rows: List[Dict[str, Any]] = []
    if not args.skip_phase2:
        best_attn_scale = float(phase1_best_hybrid["row"]["region_attn_scale"])
        phase2_val_channels = {
            "object": local_channels_by_scale[best_attn_scale]["val_norm"]["object"],
            "activity": local_channels_by_scale[best_attn_scale]["val_norm"]["activity"],
            "scene": global_channels_norm["val"]["scene"],
            "style": global_channels_norm["val"]["style"],
        }
        phase2_test_channels = {
            "object": local_channels_by_scale[best_attn_scale]["test_norm"]["object"],
            "activity": local_channels_by_scale[best_attn_scale]["test_norm"]["activity"],
            "scene": global_channels_norm["test"]["scene"],
            "style": global_channels_norm["test"]["style"],
        }
        phase2_val_stack = build_expert_stack(phase2_val_channels, candidate_topk_val, EXPERT_ORDER)
        phase2_test_stack = build_expert_stack(phase2_test_channels, candidate_topk_test, EXPERT_ORDER)
        phase2_val_topk_logits = gather_topk_values(slr_val_logits, candidate_topk_val)
        phase2_test_topk_logits = gather_topk_values(slr_test_logits, candidate_topk_test)

        for temperature in phase2_temperature_list:
            val_weights = compute_soft_routing_weights(phase2_val_stack, phase2_val_topk_logits, temperature=float(temperature))
            test_weights = compute_soft_routing_weights(phase2_test_stack, phase2_test_topk_logits, temperature=float(temperature))
            val_routed = scatter_candidate_values(
                apply_soft_routing(phase2_val_stack, val_weights),
                candidate_topk_val,
                num_classes=num_classes,
            )
            test_routed = scatter_candidate_values(
                apply_soft_routing(phase2_test_stack, test_weights),
                candidate_topk_test,
                num_classes=num_classes,
            )
            for beta in phase2_beta_list:
                for gate_gamma in phase2_gate_gamma_list:
                    val_logits = _combine_with_gate(
                        slr_val_logits,
                        val_routed,
                        phase_gate_cache[float(gate_gamma)]["val"],
                        topk=int(args.topk),
                        beta=float(beta),
                    )
                    test_logits = _combine_with_gate(
                        slr_test_logits,
                        test_routed,
                        phase_gate_cache[float(gate_gamma)]["test"],
                        topk=int(args.topk),
                        beta=float(beta),
                    )
                    val_scores = _sigmoid(val_logits)
                    test_scores = _sigmoid(test_logits)
                    bundle = _evaluate_score_bundle(val_scores, val_base["labels"], test_scores, test_base["labels"])
                    row = {
                        "setup": "phase2_soft_routing",
                        "region_attn_scale": best_attn_scale,
                        "temperature": float(temperature),
                        "gate_gamma": float(gate_gamma),
                        "beta": float(beta),
                        "val_macro_classwise": float(bundle["classwise"]["val"]["macro"]),
                        "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                        "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                    }
                    phase2_rows.append(row)
                    phase2_best = _select_best(
                        phase2_best,
                        row,
                        {
                            **bundle,
                            "val_logits": val_logits.astype(np.float32),
                            "test_logits": test_logits.astype(np.float32),
                            "val_scores": val_scores.astype(np.float32),
                            "test_scores": test_scores.astype(np.float32),
                        },
                    )
                    if phase2_best["row"] == row:
                        phase2_best["routing"] = {
                            "val": _routing_stats(val_weights),
                            "test": _routing_stats(test_weights),
                        }

    phase1_scene_best = phase1_best_by_setup["scene_style_global_only"]
    phase1_global_best = phase1_best_by_setup["all_global"]

    phase1_identity_logits = _combine_with_gate(
        slr_test_logits,
        np.zeros_like(slr_test_logits, dtype=np.float32),
        np.ones(slr_test_logits.shape[0], dtype=np.float32),
        topk=int(args.topk),
        beta=0.0,
    )
    phase1_identity_max_abs_diff = float(
        np.max(np.abs(np.asarray(phase1_identity_logits, dtype=np.float32) - np.asarray(slr_test_logits, dtype=np.float32)))
    )

    low_margin_top2 = _ordered_topk_indices(slr_test_logits, topk=2)
    low_margin_values = slr_test_logits[np.arange(slr_test_logits.shape[0]), low_margin_top2[:, 0]] - slr_test_logits[
        np.arange(slr_test_logits.shape[0]), low_margin_top2[:, 1]
    ]
    low_margin_mask = np.asarray(low_margin_values < float(args.subset_margin_tau), dtype=np.bool_)
    confusion_neighborhoods = build_confusion_neighborhoods(
        slr_train_logits,
        train_binary_labels,
        topk=int(args.confusion_topk),
        top_n=int(args.hard_negative_topn),
    )
    top2_confusion_mask = _top2_confusion_mask(slr_test_logits, confusion_neighborhoods)
    candidate_pairs = _topk_candidate_pairs(slr_test_logits, topk=int(args.topk))

    comparison_rows = [
        _comparison_row("baseline", baseline_eval),
        _comparison_row("scenario_slr_c", slr_eval),
        _comparison_row("v2_best_reference", v2_eval),
        _comparison_row("phase1_scene_style_global_only", phase1_scene_best["bundle"]),
        _comparison_row("phase1_all_global", phase1_global_best["bundle"]),
        _comparison_row("phase1_region_conditioned_updater", phase1_best_hybrid["bundle"]),
    ]
    if phase2_best is not None:
        comparison_rows.append(_comparison_row("phase2_soft_routing", phase2_best["bundle"]))

    _write_csv(output_dir / "main_comparison.csv", comparison_rows)
    _write_csv(output_dir / "phase1_search_results.csv", phase1_rows)
    if phase2_rows:
        _write_csv(output_dir / "phase2_search_results.csv", phase2_rows)

    method_scores = {
        "slr_c": np.asarray(slr_test_scores, dtype=np.float32),
        "v2_best": np.asarray(v2_test_scores, dtype=np.float32),
        "phase1_scene_style_global_only": np.asarray(phase1_scene_best["bundle"]["test_scores"], dtype=np.float32),
        "phase1_all_global": np.asarray(phase1_global_best["bundle"]["test_scores"], dtype=np.float32),
        "phase1_region_conditioned_updater": np.asarray(phase1_best_hybrid["bundle"]["test_scores"], dtype=np.float32),
    }
    method_thresholds = {
        "slr_c": np.asarray(slr_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32),
        "v2_best": np.asarray(v2_eval["classwise"]["val"]["class_thresholds"], dtype=np.float32),
        "phase1_scene_style_global_only": np.asarray(
            phase1_scene_best["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
        "phase1_all_global": np.asarray(
            phase1_global_best["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
        "phase1_region_conditioned_updater": np.asarray(
            phase1_best_hybrid["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        ),
    }
    method_logits = {
        "slr_c": np.asarray(slr_test_logits, dtype=np.float32),
        "v2_best": np.asarray(v2_test_logits, dtype=np.float32),
        "phase1_scene_style_global_only": np.asarray(phase1_scene_best["bundle"]["test_logits"], dtype=np.float32),
        "phase1_all_global": np.asarray(phase1_global_best["bundle"]["test_logits"], dtype=np.float32),
        "phase1_region_conditioned_updater": np.asarray(phase1_best_hybrid["bundle"]["test_logits"], dtype=np.float32),
    }
    if phase2_best is not None:
        method_scores["phase2_soft_routing"] = np.asarray(phase2_best["bundle"]["test_scores"], dtype=np.float32)
        method_thresholds["phase2_soft_routing"] = np.asarray(
            phase2_best["bundle"]["classwise"]["val"]["class_thresholds"],
            dtype=np.float32,
        )
        method_logits["phase2_soft_routing"] = np.asarray(phase2_best["bundle"]["test_logits"], dtype=np.float32)

    subset_diagnostics = {
        "low_margin_subset": {
            key: _subset_metrics(value, test_base["labels"], method_thresholds[key], low_margin_mask)
            for key, value in method_scores.items()
        },
        "top2_confusion_subset": {
            key: _subset_metrics(value, test_base["labels"], method_thresholds[key], top2_confusion_mask)
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
    diagnostics = {
        "candidate_recall_topk": candidate_recall,
        "oracle_topk_upper_bound": oracle_topk,
        "v2_verification_gap": _verification_gap(
            slr_test_logits,
            np.asarray(v2_pairwise_test * v2_gate_test[:, None], dtype=np.float32),
            test_base["labels"],
            int(args.topk),
        ),
        "region_cache_feature_alignment": {
            "val_max_abs_diff": val_region_feature_diff,
            "test_max_abs_diff": test_region_feature_diff,
        },
        "phase1_identity_max_abs_diff": phase1_identity_max_abs_diff,
    }

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "base_cache_dir": str(base_cache_dir),
        "region_cache_dir": str(region_cache_dir),
        "slr": {
            "source": "scenario",
            "topk": int(args.topk),
            "alpha": float(args.alpha),
        },
        "fixed_relation_backend": {
            "relation_mode": "hard_negative_diff",
            "profile_topn": int(args.profile_topn),
            "activation_topm": int(args.activation_topm),
            "hard_negative_topn": int(args.hard_negative_topn),
        },
        "v2_reference": {
            "pair_profile_topn": int(args.v2_pair_profile_topn),
            "gate_gamma": float(args.v2_gate_gamma),
            "beta": float(args.v2_beta),
            "classwise": _bundle_summary(v2_eval["classwise"], include_per_class=True),
        },
        "baseline": {
            "classwise": _bundle_summary(baseline_eval["classwise"], include_per_class=True),
        },
        "slr_c": {
            "classwise": _bundle_summary(slr_eval["classwise"], include_per_class=True),
        },
        "phase1": {
            "search_space": {
                "region_attn_scale_list": region_attn_scale_list,
                "local_weight_list": phase1_local_weight_list,
                "global_weight_list": phase1_global_weight_list,
                "beta_list": phase1_beta_list,
                "gate_gamma_list": phase1_gate_gamma_list,
            },
            "best_by_setup": {
                key: {
                    "row": value["row"],
                    "classwise": _bundle_summary(value["bundle"]["classwise"], include_per_class=True),
                }
                for key, value in phase1_best_by_setup.items()
            },
            "best_hybrid": {
                "row": phase1_best_hybrid["row"],
                "classwise": _bundle_summary(phase1_best_hybrid["bundle"]["classwise"], include_per_class=True),
                "attention": phase1_best_hybrid.get("attention", {}),
            },
        },
        "phase2": None
        if phase2_best is None
        else {
            "search_space": {
                "temperature_list": phase2_temperature_list,
                "beta_list": phase2_beta_list,
                "gate_gamma_list": phase2_gate_gamma_list,
            },
            "best": {
                "row": phase2_best["row"],
                "classwise": _bundle_summary(phase2_best["bundle"]["classwise"], include_per_class=True),
                "routing": phase2_best.get("routing", {}),
            },
        },
        "diagnostics": diagnostics,
        "subset_diagnostics": subset_diagnostics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
