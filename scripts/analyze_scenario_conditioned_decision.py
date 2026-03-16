#!/usr/bin/env python3
"""Analyze scenario-conditioned decision learning on top of fixed SLR-C."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clip  # type: ignore

from scripts.analyze_data_driven_agent_evidence_verification import (
    _evaluate_with_class_thresholds,
    _json_ready,
)
from scripts.analyze_text_prior_boundary import (
    _build_text_pools,
    _encode_text_pool,
    _load_class_names,
)
from src.utils.decision_rule_calibration import search_classwise_thresholds
from src.utils.text_prior_analysis import (
    apply_topk_rerank_fusion,
    build_confusion_pairs,
    class_gain_rows,
    evaluate_with_validation_threshold,
    threshold_predictions,
)

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_GEMINI_FILE = PROJECT_ROOT.parent / "Intentonomy" / "data" / "intent_description_gemini.json"
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scenario-conditioned decision sanity-check experiments on top of SLR-C."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION_FILE))
    parser.add_argument("--gemini-file", type=str, default=str(DEFAULT_GEMINI_FILE))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--slr-fusion-mode", type=str, default="add_norm")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lrs", type=str, default="0.03,0.01,0.003")
    parser.add_argument("--weight-decays", type=str, default="0.0,0.0001")
    parser.add_argument("--temperatures", type=str, default="1.0,2.0,4.0")
    return parser.parse_args()


def _parse_float_list(raw_value: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_scenario_conditioned_decision"
    return Path(output_dir_arg)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_cache_bundle(path: Path) -> Dict[str, Any]:
    bundle = np.load(path, allow_pickle=False)
    return {
        key: np.asarray(bundle[key]) if key != "image_ids" else [str(item) for item in bundle[key].tolist()]
        for key in bundle.files
    }


def _assert_same_ids(name: str, left_ids: Sequence[str], right_ids: Sequence[str]) -> None:
    left = [str(item) for item in left_ids]
    right = [str(item) for item in right_ids]
    if left != right:
        raise RuntimeError(f"{name} image order mismatch between caches.")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


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


def _comparison_row(method: str, bundle: Mapping[str, Any], note: str) -> Dict[str, Any]:
    metrics = bundle["classwise"]["test"]
    return {
        "method": method,
        "macro": float(metrics["macro"]) * 100.0,
        "micro": float(metrics["micro"]) * 100.0,
        "samples": float(metrics["samples"]) * 100.0,
        "mAP": float(metrics["mAP"]),
        "hard": float(metrics["hard"]) * 100.0,
        "note": note,
    }


def _text_logits_from_features(
    image_features: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    similarity = np.asarray(image_features, dtype=np.float32) @ np.asarray(
        text_embeddings, dtype=np.float32
    ).T
    return similarity * float(logit_scale)


def _scenario_distribution(prior_logits: np.ndarray, temperature: float, mode: str) -> np.ndarray:
    logits = np.asarray(prior_logits, dtype=np.float32)
    if mode == "hard":
        indices = np.argmax(logits, axis=1)
        probs = np.zeros_like(logits, dtype=np.float32)
        probs[np.arange(logits.shape[0]), indices] = 1.0
        return probs
    scaled = logits / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    return probs.astype(np.float32)


class ScenarioConditionedDecisionModel(nn.Module):
    def __init__(
        self,
        num_scenarios: int,
        num_classes: int,
        use_bias: bool,
        use_threshold: bool,
        conditioned: bool,
    ) -> None:
        super().__init__()
        self.num_scenarios = int(num_scenarios)
        self.num_classes = int(num_classes)
        self.use_bias = bool(use_bias)
        self.use_threshold = bool(use_threshold)
        self.conditioned = bool(conditioned)

        if self.use_bias:
            self.bias_base = nn.Parameter(torch.zeros(self.num_classes))
            if self.conditioned:
                self.bias_matrix = nn.Parameter(torch.zeros(self.num_scenarios, self.num_classes))
            else:
                self.register_parameter("bias_matrix", None)
        else:
            self.register_parameter("bias_base", None)
            self.register_parameter("bias_matrix", None)

        if self.use_threshold:
            self.threshold_base = nn.Parameter(torch.zeros(self.num_classes))
            if self.conditioned:
                self.threshold_matrix = nn.Parameter(torch.zeros(self.num_scenarios, self.num_classes))
            else:
                self.register_parameter("threshold_matrix", None)
        else:
            self.register_parameter("threshold_base", None)
            self.register_parameter("threshold_matrix", None)

    def forward(self, base_logits: torch.Tensor, scenario_probs: torch.Tensor) -> torch.Tensor:
        adjusted = base_logits
        if self.use_bias:
            bias = self.bias_base
            if self.conditioned and self.bias_matrix is not None:
                bias = bias + scenario_probs @ self.bias_matrix
            adjusted = adjusted + bias
        if self.use_threshold:
            threshold_shift = self.threshold_base
            if self.conditioned and self.threshold_matrix is not None:
                threshold_shift = threshold_shift + scenario_probs @ self.threshold_matrix
            adjusted = adjusted - threshold_shift
        return adjusted


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    use_bias: bool
    use_threshold: bool
    conditioned: bool
    scenario_mode: str


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        name="static_bias",
        label="SLR-C + static bias",
        use_bias=True,
        use_threshold=False,
        conditioned=False,
        scenario_mode="soft",
    ),
    VariantSpec(
        name="scenario_bias_soft",
        label="SLR-C + scenario-conditioned bias (soft)",
        use_bias=True,
        use_threshold=False,
        conditioned=True,
        scenario_mode="soft",
    ),
    VariantSpec(
        name="scenario_bias_hard",
        label="SLR-C + scenario-conditioned bias (hard)",
        use_bias=True,
        use_threshold=False,
        conditioned=True,
        scenario_mode="hard",
    ),
    VariantSpec(
        name="scenario_threshold_soft",
        label="SLR-C + scenario-conditioned threshold (soft)",
        use_bias=False,
        use_threshold=True,
        conditioned=True,
        scenario_mode="soft",
    ),
    VariantSpec(
        name="scenario_bias_threshold_soft",
        label="SLR-C + bias + threshold (soft)",
        use_bias=True,
        use_threshold=True,
        conditioned=True,
        scenario_mode="soft",
    ),
]


def _build_tensor_dict(
    base_logits: np.ndarray,
    scenario_probs: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "base_logits": torch.as_tensor(base_logits, dtype=torch.float32, device=device),
        "scenario_probs": torch.as_tensor(scenario_probs, dtype=torch.float32, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.float32, device=device),
    }


def _train_variant(
    spec: VariantSpec,
    train_tensors: Dict[str, torch.Tensor],
    val_base_logits: np.ndarray,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_base_logits: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    lr_values: Sequence[float],
    weight_decay_values: Sequence[float],
    temperature_values: Sequence[float],
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> Dict[str, Any]:
    criterion = nn.BCEWithLogitsLoss()
    best_record: Dict[str, Any] | None = None

    for lr in lr_values:
        for weight_decay in weight_decay_values:
            temperature_grid = [1.0] if spec.scenario_mode == "hard" else temperature_values
            for temperature in temperature_grid:
                model = ScenarioConditionedDecisionModel(
                    num_scenarios=int(train_tensors["scenario_probs"].shape[1]),
                    num_classes=int(train_tensors["labels"].shape[1]),
                    use_bias=spec.use_bias,
                    use_threshold=spec.use_threshold,
                    conditioned=spec.conditioned,
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                )

                stale_epochs = 0
                best_local: Dict[str, Any] | None = None
                history: List[Dict[str, Any]] = []

                train_probs = torch.as_tensor(
                    _scenario_distribution(
                        train_tensors["scenario_probs"].detach().cpu().numpy(),
                        temperature=float(temperature),
                        mode=spec.scenario_mode,
                    ),
                    dtype=torch.float32,
                    device=device,
                )

                val_probs_array = (
                    _scenario_distribution(val_probs, temperature=float(temperature), mode="hard")
                    if spec.scenario_mode == "hard"
                    else _scenario_distribution(val_probs, temperature=float(temperature), mode="soft")
                )
                test_probs_array = (
                    _scenario_distribution(test_probs, temperature=float(temperature), mode="hard")
                    if spec.scenario_mode == "hard"
                    else _scenario_distribution(test_probs, temperature=float(temperature), mode="soft")
                )
                val_probs_tensor = torch.as_tensor(val_probs_array, dtype=torch.float32, device=device)
                test_probs_tensor = torch.as_tensor(test_probs_array, dtype=torch.float32, device=device)
                val_base_tensor = torch.as_tensor(val_base_logits, dtype=torch.float32, device=device)
                test_base_tensor = torch.as_tensor(test_base_logits, dtype=torch.float32, device=device)

                for epoch in range(1, int(max_epochs) + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    train_logits = model(
                        base_logits=train_tensors["base_logits"],
                        scenario_probs=train_probs,
                    )
                    loss = criterion(train_logits, train_tensors["labels"])
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.inference_mode():
                        val_adjusted_logits = model(val_base_tensor, val_probs_tensor).detach().cpu().numpy()
                        test_adjusted_logits = model(test_base_tensor, test_probs_tensor).detach().cpu().numpy()
                    bundle = _evaluate_score_bundle(
                        val_scores=_sigmoid(val_adjusted_logits),
                        val_targets=val_labels,
                        test_scores=_sigmoid(test_adjusted_logits),
                        test_targets=test_labels,
                    )
                    val_macro = float(bundle["classwise"]["val"]["macro"])
                    history.append(
                        {
                            "epoch": int(epoch),
                            "loss": float(loss.detach().cpu()),
                            "val_macro_classwise": val_macro,
                            "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                            "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                        }
                    )

                    if best_local is None or val_macro > float(best_local["bundle"]["classwise"]["val"]["macro"]) + 1e-9:
                        best_local = {
                            "bundle": bundle,
                            "epoch": int(epoch),
                            "state_dict": {
                                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                            },
                            "temperature": float(temperature),
                            "lr": float(lr),
                            "weight_decay": float(weight_decay),
                            "history": history.copy(),
                            "test_logits": test_adjusted_logits.astype(np.float32),
                            "val_logits": val_adjusted_logits.astype(np.float32),
                        }
                        stale_epochs = 0
                    else:
                        stale_epochs += 1

                    if stale_epochs >= int(patience):
                        break

                if best_local is None:
                    continue

                if best_record is None or float(best_local["bundle"]["classwise"]["val"]["macro"]) > float(
                    best_record["bundle"]["classwise"]["val"]["macro"]
                ) + 1e-9:
                    best_record = best_local

    if best_record is None:
        raise RuntimeError(f"Variant {spec.name} did not produce a valid result.")
    return best_record


def _bundle_summary(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "global": {
            "val": _json_ready(bundle["global"]["val"]),
            "test": _json_ready(bundle["global"]["test"]),
        },
        "classwise": {
            "val": _json_ready(bundle["classwise"]["val"]),
            "test": _json_ready(bundle["classwise"]["test"]),
        },
    }


def main() -> None:
    args = _parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = _resolve_device(args.device)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.reuse_cache_dir)

    train_base = _load_cache_bundle(cache_dir / "train_base.npz")
    val_base = _load_cache_bundle(cache_dir / "val_base.npz")
    test_base = _load_cache_bundle(cache_dir / "test_base.npz")
    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")

    _assert_same_ids("train", train_base["image_ids"], train_clip["image_ids"])
    _assert_same_ids("val", val_base["image_ids"], val_clip["image_ids"])
    _assert_same_ids("test", test_base["image_ids"], test_clip["image_ids"])

    class_names = _load_class_names(Path(args.annotation_file))
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    with torch.inference_mode():
        clip_logit_scale = float(clip_model.logit_scale.exp().detach().cpu())

    text_pools = _build_text_pools(class_names, Path(args.gemini_file))
    scenario_text_embeddings = _encode_text_pool(
        clip_model,
        text_pools["scenario"],
        wrap_prompt=False,
    )

    train_prior_logits = _text_logits_from_features(train_clip["features"], scenario_text_embeddings, clip_logit_scale)
    val_prior_logits = _text_logits_from_features(val_clip["features"], scenario_text_embeddings, clip_logit_scale)
    test_prior_logits = _text_logits_from_features(test_clip["features"], scenario_text_embeddings, clip_logit_scale)

    slr_train_logits = apply_topk_rerank_fusion(
        train_base["logits"],
        train_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode=str(args.slr_fusion_mode),
    )
    slr_val_logits = apply_topk_rerank_fusion(
        val_base["logits"],
        val_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode=str(args.slr_fusion_mode),
    )
    slr_test_logits = apply_topk_rerank_fusion(
        test_base["logits"],
        test_prior_logits,
        topk=int(args.topk),
        alpha=float(args.alpha),
        mode=str(args.slr_fusion_mode),
    )

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )
    slr_bundle = _evaluate_score_bundle(
        val_scores=_sigmoid(slr_val_logits),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=_sigmoid(slr_test_logits),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    train_tensors = _build_tensor_dict(
        base_logits=slr_train_logits,
        scenario_probs=train_prior_logits,
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        device=device,
    )

    lr_values = _parse_float_list(args.lrs)
    weight_decay_values = _parse_float_list(args.weight_decays)
    temperature_values = _parse_float_list(args.temperatures)

    variant_results: Dict[str, Dict[str, Any]] = {}
    comparison_rows = [
        _comparison_row("baseline", baseline_bundle, note="recomputed from cached baseline outputs"),
        _comparison_row("scenario SLR-C", slr_bundle, note="reconstructed from cached baseline + scenario prior"),
    ]

    for spec in VARIANTS:
        print(f"[ScenarioCond] running variant={spec.name}")
        result = _train_variant(
            spec=spec,
            train_tensors=train_tensors,
            val_base_logits=slr_val_logits,
            val_probs=val_prior_logits,
            val_labels=np.asarray(val_base["labels"], dtype=np.float32),
            test_base_logits=slr_test_logits,
            test_probs=test_prior_logits,
            test_labels=np.asarray(test_base["labels"], dtype=np.float32),
            lr_values=lr_values,
            weight_decay_values=weight_decay_values,
            temperature_values=temperature_values,
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            device=device,
        )
        variant_results[spec.name] = {
            "spec": spec,
            "result": result,
        }
        comparison_rows.append(
            _comparison_row(
                method=spec.label,
                bundle=result["bundle"],
                note=(
                    f"best_epoch={result['epoch']} lr={result['lr']} wd={result['weight_decay']} "
                    f"temp={result['temperature']} mode={spec.scenario_mode}"
                ),
            )
        )

    best_variant_name = max(
        variant_results,
        key=lambda name: float(variant_results[name]["result"]["bundle"]["classwise"]["val"]["macro"]),
    )
    best_variant = variant_results[best_variant_name]
    best_result = best_variant["result"]
    best_spec = best_variant["spec"]

    slr_thresholds = np.asarray(slr_bundle["classwise"]["val"]["class_thresholds"], dtype=np.float32)
    best_thresholds = np.asarray(best_result["bundle"]["classwise"]["val"]["class_thresholds"], dtype=np.float32)
    slr_predictions = ( _sigmoid(slr_test_logits) > slr_thresholds ).astype(np.int32)
    best_predictions = (_sigmoid(best_result["test_logits"]) > best_thresholds).astype(np.int32)

    diagnostics = {
        "best_variant": best_spec.label,
        "best_variant_key": best_variant_name,
        "class_gain_vs_slr": class_gain_rows(
            baseline_per_class_f1=np.asarray(slr_bundle["classwise"]["test"]["per_class_f1"], dtype=np.float32),
            improved_per_class_f1=np.asarray(best_result["bundle"]["classwise"]["test"]["per_class_f1"], dtype=np.float32),
            class_names=class_names,
            top_n=10,
        ),
        "top_confusion_pairs_slr": build_confusion_pairs(
            targets=np.asarray(test_base["labels"], dtype=np.int32),
            predictions=slr_predictions,
            class_names=class_names,
            top_n=10,
        ),
        "top_confusion_pairs_best_variant": build_confusion_pairs(
            targets=np.asarray(test_base["labels"], dtype=np.int32),
            predictions=best_predictions,
            class_names=class_names,
            top_n=10,
        ),
    }

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "cache_dir": str(cache_dir),
        "annotation_file": str(args.annotation_file),
        "gemini_file": str(args.gemini_file),
        "slr_config": {
            "topk": int(args.topk),
            "alpha": float(args.alpha),
            "fusion_mode": str(args.slr_fusion_mode),
        },
        "train_config": {
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "lrs": [float(x) for x in lr_values],
            "weight_decays": [float(x) for x in weight_decay_values],
            "temperatures": [float(x) for x in temperature_values],
        },
        "baseline": _bundle_summary(baseline_bundle),
        "slr_c": _bundle_summary(slr_bundle),
        "variants": {
            name: {
                "label": variant_results[name]["spec"].label,
                "scenario_mode": variant_results[name]["spec"].scenario_mode,
                "use_bias": bool(variant_results[name]["spec"].use_bias),
                "use_threshold": bool(variant_results[name]["spec"].use_threshold),
                "conditioned": bool(variant_results[name]["spec"].conditioned),
                "best_epoch": int(variant_results[name]["result"]["epoch"]),
                "best_lr": float(variant_results[name]["result"]["lr"]),
                "best_weight_decay": float(variant_results[name]["result"]["weight_decay"]),
                "best_temperature": float(variant_results[name]["result"]["temperature"]),
                "bundle": _bundle_summary(variant_results[name]["result"]["bundle"]),
                "history": _json_ready(variant_results[name]["result"]["history"]),
            }
            for name in variant_results
        },
        "comparison_rows": comparison_rows,
        "diagnostics": diagnostics,
    }

    comparison_path = output_dir / "main_comparison.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(comparison_path, comparison_rows)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    print(
        "[ScenarioCond] best_variant="
        f"{best_spec.label} | "
        f"test_macro={float(best_result['bundle']['classwise']['test']['macro']) * 100.0:.2f} | "
        f"test_hard={float(best_result['bundle']['classwise']['test']['hard']) * 100.0:.2f}"
    )


if __name__ == "__main__":
    main()
