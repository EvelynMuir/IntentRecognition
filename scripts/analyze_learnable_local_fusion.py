#!/usr/bin/env python3
"""Train and evaluate a very-light learnable local fusion adapter."""

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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_text_prior_boundary import (
    DEFAULT_BASELINE_CKPT,
    _normalize_source_name,
    _assert_same_ids,
    _build_dataset,
    _build_hard_case_rows,
    _build_text_pools,
    _collect_clip_features,
    _collect_model_outputs,
    _encode_text_pool_per_class,
    _load_class_names,
    _metrics_for_json,
    _normalize_state_dict_keys,
    _prediction_shift_summary,
    _resolve_ckpt_path,
    _resolve_device,
    _resolve_gemini_file,
    _resolve_run_dir,
    _text_logits_from_prompt_embeddings,
)
from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.learnable_local_fusion import LearnableLocalFusionAdapter, build_topk_mask
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    evaluate_with_validation_threshold,
    normalize_scores_per_sample,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny local fusion adapter over (baseline_logit, text_prior) "
            "inside the baseline top-k candidate set."
        )
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
    parser.add_argument(
        "--sources",
        type=str,
        default="lexical,canonical,scenario,discriminative,lexical_plus_canonical",
        help="Comma-separated sources for learnable fusion.",
    )
    parser.add_argument(
        "--fusion-modes",
        type=str,
        default="classwise_affine,shared_mlp",
        help="Comma-separated fusion adapter types.",
    )
    parser.add_argument(
        "--mlp-feature-sets",
        type=str,
        default="zs",
        help="Comma-separated feature sets for shared MLP. Supported: zs,zspr.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="1e-3,3e-4",
        help="Comma-separated learning rates.",
    )
    parser.add_argument("--alpha-init", type=float, default=0.3)
    parser.add_argument(
        "--lambda-a-list",
        type=str,
        default="1e-2",
        help="Comma-separated regularization weights for affine slope anchor.",
    )
    parser.add_argument(
        "--lambda-b-list",
        type=str,
        default="1e-3",
        help="Comma-separated regularization weights for affine bias.",
    )
    parser.add_argument(
        "--delta-reg-list",
        type=str,
        default="1e-4",
        help="Comma-separated delta L2 regularization weights for shared MLP.",
    )
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _to_tensor_dataset(
    baseline_logits: np.ndarray,
    text_prior: np.ndarray,
    targets: np.ndarray,
    topk: int,
) -> TensorDataset:
    logits_t = torch.as_tensor(baseline_logits, dtype=torch.float32)
    prior_t = torch.as_tensor(text_prior, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    mask_t = build_topk_mask(logits_t, topk=topk)
    return TensorDataset(logits_t, prior_t, targets_t, mask_t)


def _run_adapter(
    adapter: LearnableLocalFusionAdapter,
    logits: np.ndarray,
    prior: np.ndarray,
    topk: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    adapter.eval()
    dataset = _to_tensor_dataset(logits, prior, np.zeros_like(logits, dtype=np.float32), topk=topk)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fused_list: List[np.ndarray] = []

    with torch.inference_mode():
        for batch_logits, batch_prior, _, batch_mask in loader:
            batch_logits = batch_logits.to(device)
            batch_prior = batch_prior.to(device)
            batch_mask = batch_mask.to(device)
            fused, _ = adapter(batch_logits, batch_prior, batch_mask)
            fused_list.append(fused.detach().cpu().numpy())

    return np.concatenate(fused_list, axis=0)


def _train_one_run(
    *,
    train_logits: np.ndarray,
    train_prior: np.ndarray,
    train_targets: np.ndarray,
    val_logits: np.ndarray,
    val_prior: np.ndarray,
    val_targets: np.ndarray,
    test_logits: np.ndarray,
    test_prior: np.ndarray,
    test_targets: np.ndarray,
    fusion_mode: str,
    hidden_dim: int,
    feature_set: str,
    lr: float,
    alpha_init: float,
    lambda_a: float,
    lambda_b: float,
    delta_reg: float,
    topk: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    adapter = LearnableLocalFusionAdapter(
        num_classes=train_logits.shape[1],
        mode=fusion_mode,
        hidden_dim=hidden_dim,
        alpha_init=float(alpha_init),
        feature_set=feature_set,
    ).to(device)

    criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-5)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=0.0)
    train_dataset = _to_tensor_dataset(train_logits, train_prior, train_targets, topk=topk)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_epoch = -1
    best_val_macro = float("-inf")
    best_wait = 0

    for epoch in range(int(epochs)):
        adapter.train()
        for batch_logits, batch_prior, batch_targets, batch_mask in train_loader:
            batch_logits = batch_logits.to(device)
            batch_prior = batch_prior.to(device)
            batch_targets = batch_targets.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            fused_logits, delta = adapter(batch_logits, batch_prior, batch_mask)
            loss = criterion(fused_logits, batch_targets, reduction="mean")
            if fusion_mode == "classwise_affine":
                reg = float(lambda_a) * ((adapter.a - float(alpha_init)) ** 2).mean()
                reg = reg + float(lambda_b) * (adapter.b ** 2).mean()
                loss = loss + reg
            elif float(delta_reg) > 0.0:
                reg = ((delta * batch_mask) ** 2).mean()
                loss = loss + float(delta_reg) * reg
            loss.backward()
            optimizer.step()

        val_fused_logits = _run_adapter(adapter, val_logits, val_prior, topk=topk, device=device)
        val_scores = 1.0 / (1.0 + np.exp(-val_fused_logits))
        val_metrics = evaluate_with_validation_threshold(
            val_scores,
            val_targets,
            use_inference_strategy=False,
        )
        current_val_macro = float(val_metrics["val"]["macro"])
        if current_val_macro > best_val_macro + 1e-8:
            best_val_macro = current_val_macro
            best_epoch = epoch
            best_wait = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
            }
        else:
            best_wait += 1
            if best_wait >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid best_state.")

    adapter.load_state_dict(best_state["model"])
    adapter = adapter.to(device).eval()

    val_fused_logits = _run_adapter(adapter, val_logits, val_prior, topk=topk, device=device)
    test_fused_logits = _run_adapter(adapter, test_logits, test_prior, topk=topk, device=device)
    val_scores = 1.0 / (1.0 + np.exp(-val_fused_logits))
    test_scores = 1.0 / (1.0 + np.exp(-test_fused_logits))
    metrics = evaluate_with_validation_threshold(
        val_scores,
        val_targets,
        test_scores,
        test_targets,
        use_inference_strategy=False,
    )

    return {
        "best_epoch": int(best_epoch),
        "epochs_trained": int(min(epochs, best_epoch + 1 + best_wait)),
        "val": metrics["val"],
        "test": metrics["test"],
        "test_scores": test_scores,
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
        import clip  # local import to mirror existing scripts

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
    _, test_loader_base, test_id_to_path = _build_dataset(
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

    _assert_same_ids("train", train_base["image_ids"], train_clip["image_ids"])
    _assert_same_ids("val", val_base["image_ids"], val_clip["image_ids"])
    _assert_same_ids("test", test_base["image_ids"], test_clip["image_ids"])

    text_pools = _build_text_pools(class_names, gemini_file)
    prompt_embeddings_per_class = {
        "lexical": _encode_text_pool_per_class(clip_model, text_pools["lexical"], wrap_prompt=True),
        "canonical": _encode_text_pool_per_class(clip_model, text_pools["canonical"], wrap_prompt=True),
        "scenario": _encode_text_pool_per_class(clip_model, text_pools["scenario"], wrap_prompt=False),
        "discriminative": _encode_text_pool_per_class(
            clip_model, text_pools["discriminative"], wrap_prompt=False
        ),
        "lexical_plus_canonical": _encode_text_pool_per_class(
            clip_model, text_pools["lexical_plus_canonical"], wrap_prompt=False
        ),
    }

    source_names = [_normalize_source_name(x) for x in _parse_str_list(args.sources)]
    fusion_modes = _parse_str_list(args.fusion_modes)
    mlp_feature_sets = _parse_str_list(args.mlp_feature_sets)
    learning_rates = _parse_float_list(args.learning_rates)
    lambda_a_list = _parse_float_list(args.lambda_a_list)
    lambda_b_list = _parse_float_list(args.lambda_b_list)
    delta_reg_list = _parse_float_list(args.delta_reg_list)

    baseline_metrics = evaluate_with_validation_threshold(
        val_base["scores"],
        val_base["labels"],
        test_base["scores"],
        test_base["labels"],
        use_inference_strategy=False,
    )

    run_records: List[Dict[str, Any]] = []
    plain_source_refs: Dict[str, Dict[str, Any]] = {}

    for source_name in source_names:
        if source_name not in prompt_embeddings_per_class:
            raise ValueError(f"Unsupported source: {source_name}")
        train_text_logits = _text_logits_from_prompt_embeddings(
            train_clip["features"],
            prompt_embeddings_per_class[source_name],
            clip_logit_scale,
            aggregation_mode="average",
        )
        val_text_logits = _text_logits_from_prompt_embeddings(
            val_clip["features"],
            prompt_embeddings_per_class[source_name],
            clip_logit_scale,
            aggregation_mode="average",
        )
        test_text_logits = _text_logits_from_prompt_embeddings(
            test_clip["features"],
            prompt_embeddings_per_class[source_name],
            clip_logit_scale,
            aggregation_mode="average",
        )

        train_prior = normalize_scores_per_sample(train_text_logits)
        val_prior = normalize_scores_per_sample(val_text_logits)
        test_prior = normalize_scores_per_sample(test_text_logits)

        val_plain_logits = apply_topk_rerank_fusion(
            val_base["logits"],
            val_text_logits,
            topk=args.topk,
            alpha=args.alpha,
            mode="add_norm",
        )
        test_plain_logits = apply_topk_rerank_fusion(
            test_base["logits"],
            test_text_logits,
            topk=args.topk,
            alpha=args.alpha,
            mode="add_norm",
        )
        plain_metrics = evaluate_with_validation_threshold(
            1.0 / (1.0 + np.exp(-val_plain_logits)),
            val_base["labels"],
            1.0 / (1.0 + np.exp(-test_plain_logits)),
            test_base["labels"],
            use_inference_strategy=False,
        )
        plain_source_refs[source_name] = {
            "metrics": plain_metrics,
            "test_scores": 1.0 / (1.0 + np.exp(-test_plain_logits)),
        }

        for fusion_mode in fusion_modes:
            if fusion_mode == "classwise_affine":
                feature_sets = ["na"]
                delta_regs = [0.0]
                lambda_a_values = lambda_a_list
                lambda_b_values = lambda_b_list
            elif fusion_mode == "shared_mlp":
                feature_sets = mlp_feature_sets
                delta_regs = delta_reg_list
                lambda_a_values = [0.0]
                lambda_b_values = [0.0]
            else:
                raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

            for feature_set in feature_sets:
                for lr in learning_rates:
                    for lambda_a in lambda_a_values:
                        for lambda_b in lambda_b_values:
                            for delta_reg in delta_regs:
                                result = _train_one_run(
                                    train_logits=train_base["logits"],
                                    train_prior=train_prior,
                                    train_targets=train_base["labels"],
                                    val_logits=val_base["logits"],
                                    val_prior=val_prior,
                                    val_targets=val_base["labels"],
                                    test_logits=test_base["logits"],
                                    test_prior=test_prior,
                                    test_targets=test_base["labels"],
                                    fusion_mode=fusion_mode,
                                    hidden_dim=int(args.hidden_dim),
                                    feature_set=feature_set,
                                    lr=float(lr),
                                    alpha_init=float(args.alpha_init),
                                    lambda_a=float(lambda_a),
                                    lambda_b=float(lambda_b),
                                    delta_reg=float(delta_reg),
                                    topk=int(args.topk),
                                    epochs=int(args.epochs),
                                    patience=int(args.patience),
                                    batch_size=int(args.batch_size),
                                    device=device,
                                )
                                run_records.append(
                                    {
                                        "source": source_name,
                                        "fusion_mode": fusion_mode,
                                        "feature_set": feature_set,
                                        "lr": float(lr),
                                        "alpha_init": float(args.alpha_init),
                                        "lambda_a": float(lambda_a),
                                        "lambda_b": float(lambda_b),
                                        "delta_reg": float(delta_reg),
                                        "hidden_dim": int(args.hidden_dim),
                                        "best_epoch": int(result["best_epoch"]),
                                        "epochs_trained": int(result["epochs_trained"]),
                                        "val": result["val"],
                                        "test": result["test"],
                                        "test_scores": result["test_scores"],
                                        "test_gain_over_baseline": {
                                            "macro": float(result["test"]["macro"] - baseline_metrics["test"]["macro"]),
                                            "hard": float(result["test"]["hard"] - baseline_metrics["test"]["hard"]),
                                        },
                                        "test_gain_over_plain_source_rerank": {
                                            "macro": float(result["test"]["macro"] - plain_metrics["test"]["macro"]),
                                            "hard": float(result["test"]["hard"] - plain_metrics["test"]["hard"]),
                                        },
                                    }
                                )

    run_records.sort(key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])), reverse=True)
    best_by_val_macro = run_records[0]
    best_by_val_hard = max(run_records, key=lambda item: float(item["val"]["hard"]))

    best_by_source: Dict[str, Dict[str, Any]] = {}
    for source_name in source_names:
        source_records = [row for row in run_records if row["source"] == source_name]
        if source_records:
            best_by_source[source_name] = max(
                source_records,
                key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
            )

    baseline_predictions_test = (test_base["scores"] >= float(baseline_metrics["val"]["threshold"])).astype(np.int32)
    best_plain_source_metrics = plain_source_refs[best_by_val_macro["source"]]["metrics"]
    best_plain_source_scores = plain_source_refs[best_by_val_macro["source"]]["test_scores"]
    best_plain_source_predictions = (
        best_plain_source_scores >= float(best_plain_source_metrics["val"]["threshold"])
    ).astype(np.int32)
    best_fusion_predictions = (
        best_by_val_macro["test_scores"] >= float(best_by_val_macro["val"]["threshold"])
    ).astype(np.int32)
    fusion_shift_vs_plain = _prediction_shift_summary(
        targets=test_base["labels"],
        baseline_predictions=best_plain_source_predictions,
        variant_predictions=best_fusion_predictions,
        class_names=class_names,
        top_n=10,
    )
    fusion_hard_cases_raw = _build_hard_case_rows(
        image_ids=test_base["image_ids"],
        image_id_to_path=test_id_to_path,
        targets=test_base["labels"],
        baseline_scores=best_plain_source_scores,
        baseline_predictions=best_plain_source_predictions,
        variant_scores=best_by_val_macro["test_scores"],
        variant_predictions=best_fusion_predictions,
        text_only_scores=best_plain_source_scores,
        class_names=class_names,
        case_limit=40,
        variant_label="learnable_fusion",
        text_only_label="plain_rerank",
    )
    fusion_hard_cases = {
        "plain_wrong_fusion_right": fusion_hard_cases_raw["baseline_wrong_llm_right"],
        "plain_right_fusion_wrong": fusion_hard_cases_raw["baseline_right_llm_wrong"],
    }

    leaderboard = []
    for record in run_records[: min(30, len(run_records))]:
        leaderboard.append(
            {
                "source": record["source"],
                "fusion_mode": record["fusion_mode"],
                "feature_set": record["feature_set"],
                "lr": float(record["lr"]),
                "alpha_init": float(record["alpha_init"]),
                "lambda_a": float(record["lambda_a"]),
                "lambda_b": float(record["lambda_b"]),
                "delta_reg": float(record["delta_reg"]),
                "best_epoch": int(record["best_epoch"]),
                "val_macro": float(record["val"]["macro"]),
                "val_hard": float(record["val"]["hard"]),
                "test_macro": float(record["test"]["macro"]),
                "test_hard": float(record["test"]["hard"]),
                "test_micro": float(record["test"]["micro"]),
                "test_samples": float(record["test"]["samples"]),
            }
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_learnable_local_fusion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = output_dir / "learnable_local_fusion_leaderboard.csv"
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)
    hard_case_csv = output_dir / "hard_cases_plain_wrong_fusion_right.csv"
    with hard_case_csv.open("w", encoding="utf-8", newline="") as handle:
        import csv

        rows = [
            {
                "image_id": row["image_id"],
                "sample_f1_delta": row["sample_f1_delta"],
                "recovered_labels": "|".join(row["recovered_labels"]),
                "ground_truth_labels": "|".join(row["ground_truth_labels"]),
                "plain_pred_labels": "|".join(row["baseline_pred_labels"]),
                "learnable_fusion_pred_labels": "|".join(row["learnable_fusion_pred_labels"]),
            }
            for row in fusion_hard_cases["plain_wrong_fusion_right"]
        ]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    hard_case_csv = output_dir / "hard_cases_plain_right_fusion_wrong.csv"
    with hard_case_csv.open("w", encoding="utf-8", newline="") as handle:
        import csv

        rows = [
            {
                "image_id": row["image_id"],
                "sample_f1_delta": row["sample_f1_delta"],
                "dropped_true_labels": "|".join(row["dropped_true_labels"]),
                "new_false_positive_labels": "|".join(row["new_false_positive_labels"]),
                "ground_truth_labels": "|".join(row["ground_truth_labels"]),
                "plain_pred_labels": "|".join(row["baseline_pred_labels"]),
                "learnable_fusion_pred_labels": "|".join(row["learnable_fusion_pred_labels"]),
            }
            for row in fusion_hard_cases["plain_right_fusion_wrong"]
        ]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    result = {
        "metadata": {
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "gemini_file": str(gemini_file),
            "device": str(device),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "sources": source_names,
            "fusion_modes": fusion_modes,
            "mlp_feature_sets": mlp_feature_sets,
            "learning_rates": learning_rates,
            "alpha_init": float(args.alpha_init),
            "lambda_a_list": lambda_a_list,
            "lambda_b_list": lambda_b_list,
            "delta_reg_list": delta_reg_list,
            "topk": int(args.topk),
            "alpha": float(args.alpha),
            "hidden_dim": int(args.hidden_dim),
            "output_dir": str(output_dir),
            "max_samples": args.max_samples,
        },
        "baseline": {
            "val": _metrics_for_json(baseline_metrics["val"], include_per_class=True),
            "test": _metrics_for_json(baseline_metrics["test"], include_per_class=True),
        },
        "plain_source_rerank": {
            source_name: {
                "val": _metrics_for_json(item["metrics"]["val"], include_per_class=True),
                "test": _metrics_for_json(item["metrics"]["test"], include_per_class=True),
            }
            for source_name, item in plain_source_refs.items()
        },
        "leaderboard_top30": leaderboard,
        "best_overall_by_val_macro": {
            "config": {
                "source": best_by_val_macro["source"],
                "fusion_mode": best_by_val_macro["fusion_mode"],
                "feature_set": best_by_val_macro["feature_set"],
                    "lr": float(best_by_val_macro["lr"]),
                    "alpha_init": float(best_by_val_macro["alpha_init"]),
                    "lambda_a": float(best_by_val_macro["lambda_a"]),
                    "lambda_b": float(best_by_val_macro["lambda_b"]),
                    "delta_reg": float(best_by_val_macro["delta_reg"]),
                    "hidden_dim": int(best_by_val_macro["hidden_dim"]),
                    "best_epoch": int(best_by_val_macro["best_epoch"]),
            },
            "val": _metrics_for_json(best_by_val_macro["val"], include_per_class=True),
            "test": _metrics_for_json(best_by_val_macro["test"], include_per_class=True),
            "test_gain_over_baseline": best_by_val_macro["test_gain_over_baseline"],
            "test_gain_over_plain_source_rerank": best_by_val_macro["test_gain_over_plain_source_rerank"],
            "prediction_shift_vs_plain_source_rerank": fusion_shift_vs_plain,
        },
        "best_overall_by_val_hard": {
                "config": {
                    "source": best_by_val_hard["source"],
                    "fusion_mode": best_by_val_hard["fusion_mode"],
                    "feature_set": best_by_val_hard["feature_set"],
                    "lr": float(best_by_val_hard["lr"]),
                    "alpha_init": float(best_by_val_hard["alpha_init"]),
                    "lambda_a": float(best_by_val_hard["lambda_a"]),
                    "lambda_b": float(best_by_val_hard["lambda_b"]),
                    "delta_reg": float(best_by_val_hard["delta_reg"]),
                    "hidden_dim": int(best_by_val_hard["hidden_dim"]),
                    "best_epoch": int(best_by_val_hard["best_epoch"]),
            },
            "val": _metrics_for_json(best_by_val_hard["val"]),
            "test": _metrics_for_json(best_by_val_hard["test"]),
            "test_gain_over_baseline": best_by_val_hard["test_gain_over_baseline"],
            "test_gain_over_plain_source_rerank": best_by_val_hard["test_gain_over_plain_source_rerank"],
        },
        "best_by_source": {
            source_name: {
                "config": {
                    "fusion_mode": record["fusion_mode"],
                    "feature_set": record["feature_set"],
                    "lr": float(record["lr"]),
                    "alpha_init": float(record["alpha_init"]),
                    "lambda_a": float(record["lambda_a"]),
                    "lambda_b": float(record["lambda_b"]),
                    "delta_reg": float(record["delta_reg"]),
                    "hidden_dim": int(record["hidden_dim"]),
                    "best_epoch": int(record["best_epoch"]),
                },
                "val": _metrics_for_json(record["val"]),
                "test": _metrics_for_json(record["test"]),
                "test_gain_over_baseline": record["test_gain_over_baseline"],
                "test_gain_over_plain_source_rerank": record["test_gain_over_plain_source_rerank"],
            }
            for source_name, record in best_by_source.items()
        },
        "hard_cases": {
            "plain_wrong_fusion_right_count": len(fusion_hard_cases["plain_wrong_fusion_right"]),
            "plain_right_fusion_wrong_count": len(fusion_hard_cases["plain_right_fusion_wrong"]),
            "plain_wrong_fusion_right": fusion_hard_cases["plain_wrong_fusion_right"],
            "plain_right_fusion_wrong": fusion_hard_cases["plain_right_fusion_wrong"],
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
