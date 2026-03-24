#!/usr/bin/env python3
"""Analyze confusion-aware multi-prototype learning on top of cached baseline logits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_data_driven_agent_evidence_verification import (
    _evaluate_with_class_thresholds,
    _json_ready,
)
from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.decision_rule_calibration import search_classwise_thresholds
from src.utils.evidence_verification import build_confusion_neighborhoods
from src.utils.text_prior_analysis import evaluate_with_validation_threshold

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
)
REFERENCE_ROWS = [
    {
        "method": "baseline",
        "macro": 48.231551928209086,
        "micro": 52.04489252425337,
        "samples": 52.937000408217514,
        "mAP": 50.28759370113977,
        "hard": 29.705768443993254,
        "note": "cached baseline reference",
    },
    {
        "method": "scenario SLR-C",
        "macro": 51.28005127390507,
        "micro": 59.13214990138067,
        "samples": 58.46663296498823,
        "mAP": 53.66037803129969,
        "hard": 33.97660825373071,
        "note": "repo analysis reference",
    },
    {
        "method": "fixed benchmark-bank best",
        "macro": 51.91781808986489,
        "micro": float("nan"),
        "samples": float("nan"),
        "mAP": float("nan"),
        "hard": 35.63157801259155,
        "note": "hard_negative_diff relation-family best reference",
    },
    {
        "method": "latent basis MVP",
        "macro": 48.524659970224135,
        "micro": 51.40007671653242,
        "samples": 51.994551957545376,
        "mAP": 50.458404319955186,
        "hard": 29.349520475301112,
        "note": "repo full-train LIR reference",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAML-MP style prototype regularization experiments on cached CLIP features."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION_FILE))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lrs", type=str, default="0.001,0.0005")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-proto", type=float, default=0.1)
    parser.add_argument("--lambda-hard", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--confusion-topk", type=int, default=10)
    parser.add_argument("--confusion-topn", type=int, default=3)
    parser.add_argument("--random-topn", type=int, default=3)
    return parser.parse_args()


def _parse_float_list(raw_value: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_confusion_aware_multi_prototype"
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


def _load_class_names(annotation_file: Path) -> List[str]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    return [str(category["name"]) for category in categories]


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


def _build_random_neighborhoods(num_classes: int, top_n: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(int(seed))
    neighborhoods: List[List[int]] = []
    for class_idx in range(num_classes):
        candidates = [idx for idx in range(num_classes) if idx != class_idx]
        sampled = rng.choice(candidates, size=min(int(top_n), len(candidates)), replace=False)
        neighborhoods.append([int(idx) for idx in sampled.tolist()])
    return neighborhoods


def _build_negative_mask(neighborhoods: Sequence[Sequence[int]], num_classes: int) -> np.ndarray:
    mask = np.zeros((num_classes, num_classes), dtype=np.float32)
    for class_idx, neg_ids in enumerate(neighborhoods):
        for neg_idx in neg_ids:
            if int(neg_idx) == int(class_idx):
                continue
            mask[int(class_idx), int(neg_idx)] = 1.0
    return mask


def _build_prototype_tensor(
    features: np.ndarray,
    labels: np.ndarray,
    num_prototypes: int,
    seed: int,
) -> np.ndarray:
    feat = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
    num_classes = int(targets.shape[1])
    output = np.zeros((num_classes, int(num_prototypes), feat.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(int(seed))

    for class_idx in range(num_classes):
        positive_ids = np.where(targets[:, class_idx] > 0)[0]
        if positive_ids.size == 0:
            sample_ids = rng.choice(feat.shape[0], size=int(num_prototypes), replace=True)
            output[class_idx] = feat[sample_ids]
            continue
        class_features = feat[positive_ids]
        if int(num_prototypes) == 1 or class_features.shape[0] < int(num_prototypes):
            center = class_features.mean(axis=0, keepdims=True)
            center = center / np.maximum(np.linalg.norm(center, axis=1, keepdims=True), 1e-8)
            output[class_idx] = np.repeat(center, repeats=int(num_prototypes), axis=0)
            continue
        kmeans = KMeans(
            n_clusters=int(num_prototypes),
            n_init=10,
            random_state=int(seed),
        )
        cluster_centers = kmeans.fit(class_features).cluster_centers_.astype(np.float32)
        cluster_centers = cluster_centers / np.maximum(
            np.linalg.norm(cluster_centers, axis=1, keepdims=True),
            1e-8,
        )
        output[class_idx] = cluster_centers
    return output


class PrototypeResidualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        initial_prototypes: np.ndarray,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.num_prototypes = int(initial_prototypes.shape[1])

        self.adapter = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.residual_head = nn.Linear(self.input_dim, self.num_classes, bias=False)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(self.input_dim))
            self.residual_head.weight.zero_()

        self.prototypes = nn.Parameter(
            torch.as_tensor(initial_prototypes, dtype=torch.float32)
        )

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        adapted = self.adapter(features)
        normalized = F.normalize(adapted, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        proto_sim = torch.einsum("bd,cmd->bcm", normalized, proto_norm)
        class_proto_scores, proto_assignments = proto_sim.max(dim=-1)
        residual_logits = self.residual_head(normalized)
        final_logits = base_logits + residual_logits
        return {
            "final_logits": final_logits,
            "proto_scores": class_proto_scores,
            "proto_assignments": proto_assignments,
        }


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    num_prototypes: int
    hard_mode: str


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        name="single_proto_no_hard",
        label="baseline + prototype",
        num_prototypes=1,
        hard_mode="none",
    ),
    VariantSpec(
        name="single_proto_random_hard",
        label="baseline + prototype + random hard",
        num_prototypes=1,
        hard_mode="random",
    ),
    VariantSpec(
        name="single_proto_confusion_hard",
        label="baseline + prototype + confusion hard",
        num_prototypes=1,
        hard_mode="confusion",
    ),
    VariantSpec(
        name="multi_proto_2_confusion_hard",
        label="baseline + 2 prototypes + confusion hard",
        num_prototypes=2,
        hard_mode="confusion",
    ),
    VariantSpec(
        name="multi_proto_4_confusion_hard",
        label="baseline + 4 prototypes + confusion hard",
        num_prototypes=4,
        hard_mode="confusion",
    ),
    VariantSpec(
        name="multi_proto_8_confusion_hard",
        label="baseline + 8 prototypes + confusion hard",
        num_prototypes=8,
        hard_mode="confusion",
    ),
]


def _prototype_alignment_loss(
    proto_scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scaled = proto_scores / max(float(temperature), 1e-6)
    log_probs = F.log_softmax(scaled, dim=1)
    pos_mask = labels > 0
    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(log_probs * pos_mask.float()).sum(dim=1) / pos_count.float()
    return loss.mean()


def _hard_negative_margin_loss(
    proto_scores: torch.Tensor,
    labels: torch.Tensor,
    negative_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if negative_mask.sum().item() <= 0:
        return torch.zeros((), dtype=proto_scores.dtype, device=proto_scores.device)

    pos_mask = (labels > 0).float()
    if pos_mask.sum().item() <= 0:
        return torch.zeros((), dtype=proto_scores.dtype, device=proto_scores.device)

    pos_scores = proto_scores.unsqueeze(2)
    neg_scores = proto_scores.unsqueeze(1)
    pair_loss = F.relu(float(margin) - pos_scores + neg_scores)
    pair_mask = pos_mask.unsqueeze(2) * negative_mask.unsqueeze(0)
    denom = pair_mask.sum().clamp_min(1.0)
    return (pair_loss * pair_mask).sum() / denom


def _tensorize_split(
    features: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(features, dtype=torch.float32, device=device),
        "base_logits": torch.as_tensor(base_logits, dtype=torch.float32, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.float32, device=device),
    }


def _predict_outputs(
    model: PrototypeResidualModel,
    split_tensors: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        outputs = model(
            features=split_tensors["features"],
            base_logits=split_tensors["base_logits"],
        )
        logits = outputs["final_logits"].detach().cpu().numpy().astype(np.float32)
        scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        assignments = outputs["proto_assignments"].detach().cpu().numpy().astype(np.int64)
    return {
        "logits": logits,
        "scores": scores,
        "assignments": assignments,
    }


def _build_usage_rows(
    labels: np.ndarray,
    assignments: np.ndarray,
    class_names: Sequence[str],
    num_prototypes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    targets = np.asarray(labels, dtype=np.float32)
    assigns = np.asarray(assignments, dtype=np.int64)
    for class_idx, class_name in enumerate(class_names):
        positive_mask = targets[:, class_idx] > 0
        if not np.any(positive_mask):
            for proto_idx in range(num_prototypes):
                rows.append(
                    {
                        "class_id": int(class_idx),
                        "class_name": str(class_name),
                        "prototype_id": int(proto_idx),
                        "selection_count": 0,
                    }
                )
            continue
        proto_choices = assigns[positive_mask, class_idx]
        counts = np.bincount(proto_choices, minlength=int(num_prototypes))
        for proto_idx in range(num_prototypes):
            rows.append(
                {
                    "class_id": int(class_idx),
                    "class_name": str(class_name),
                    "prototype_id": int(proto_idx),
                    "selection_count": int(counts[proto_idx]),
                }
            )
    return rows


def _train_variant(
    spec: VariantSpec,
    train_split: Dict[str, torch.Tensor],
    val_split: Dict[str, torch.Tensor],
    test_split: Dict[str, torch.Tensor],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    negative_mask_matrix: np.ndarray,
    lr_values: Sequence[float],
    weight_decay: float,
    lambda_proto: float,
    lambda_hard: float,
    margin: float,
    temperature: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> Dict[str, Any]:
    criterion = AsymmetricLossOptimized(
        gamma_neg=2,
        gamma_pos=0,
        clip=0.05,
        eps=1e-5,
        disable_torch_grad_focal_loss=False,
    )
    negative_mask = torch.as_tensor(negative_mask_matrix, dtype=torch.float32, device=train_split["features"].device)
    best_record: Dict[str, Any] | None = None

    for lr in lr_values:
        initial_prototypes = _build_prototype_tensor(
            features=train_features,
            labels=train_labels,
            num_prototypes=int(spec.num_prototypes),
            seed=int(seed),
        )
        model = PrototypeResidualModel(
            input_dim=int(train_features.shape[1]),
            num_classes=int(train_labels.shape[1]),
            initial_prototypes=initial_prototypes,
        ).to(train_split["features"].device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        stale_epochs = 0
        best_local: Dict[str, Any] | None = None
        history: List[Dict[str, Any]] = []

        for epoch in range(1, int(max_epochs) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                features=train_split["features"],
                base_logits=train_split["base_logits"],
            )
            cls_loss = criterion(outputs["final_logits"], train_split["labels"], reduction="mean")
            proto_loss = _prototype_alignment_loss(
                outputs["proto_scores"],
                train_split["labels"],
                temperature=float(temperature),
            )
            hard_loss = _hard_negative_margin_loss(
                outputs["proto_scores"],
                train_split["labels"],
                negative_mask=negative_mask,
                margin=float(margin),
            )
            total_loss = cls_loss + float(lambda_proto) * proto_loss + float(lambda_hard) * hard_loss
            total_loss.backward()
            optimizer.step()

            val_outputs = _predict_outputs(model, val_split)
            test_outputs = _predict_outputs(model, test_split)
            bundle = _evaluate_score_bundle(
                val_scores=val_outputs["scores"],
                val_targets=val_labels,
                test_scores=test_outputs["scores"],
                test_targets=test_labels,
            )
            val_macro = float(bundle["classwise"]["val"]["macro"])

            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(total_loss.detach().cpu()),
                    "cls_loss": float(cls_loss.detach().cpu()),
                    "proto_loss": float(proto_loss.detach().cpu()),
                    "hard_loss": float(hard_loss.detach().cpu()),
                    "val_macro_classwise": val_macro,
                    "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                    "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                }
            )

            if best_local is None or val_macro > float(best_local["bundle"]["classwise"]["val"]["macro"]) + 1e-9:
                best_local = {
                    "lr": float(lr),
                    "epoch": int(epoch),
                    "bundle": bundle,
                    "history": history.copy(),
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "test_assignments": test_outputs["assignments"],
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
    confusion_neighborhoods = build_confusion_neighborhoods(
        candidate_logits=np.asarray(train_base["logits"], dtype=np.float32),
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        topk=int(args.confusion_topk),
        top_n=int(args.confusion_topn),
        use_rank_weight=True,
    )
    random_neighborhoods = _build_random_neighborhoods(
        num_classes=int(train_base["labels"].shape[1]),
        top_n=int(args.random_topn),
        seed=int(args.seed),
    )

    train_split = _tensorize_split(
        features=np.asarray(train_clip["features"], dtype=np.float32),
        base_logits=np.asarray(train_base["logits"], dtype=np.float32),
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        device=device,
    )
    val_split = _tensorize_split(
        features=np.asarray(val_clip["features"], dtype=np.float32),
        base_logits=np.asarray(val_base["logits"], dtype=np.float32),
        labels=np.asarray(val_base["labels"], dtype=np.float32),
        device=device,
    )
    test_split = _tensorize_split(
        features=np.asarray(test_clip["features"], dtype=np.float32),
        base_logits=np.asarray(test_base["logits"], dtype=np.float32),
        labels=np.asarray(test_base["labels"], dtype=np.float32),
        device=device,
    )

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    lr_values = _parse_float_list(args.lrs)
    variant_results: Dict[str, Dict[str, Any]] = {}
    comparison_rows = REFERENCE_ROWS.copy()

    for spec in VARIANTS:
        print(f"[CAML-MP] running variant={spec.name}")
        negative_mask_matrix = np.zeros((int(train_base["labels"].shape[1]), int(train_base["labels"].shape[1])), dtype=np.float32)
        if spec.hard_mode == "random":
            negative_mask_matrix = _build_negative_mask(
                random_neighborhoods,
                num_classes=int(train_base["labels"].shape[1]),
            )
        elif spec.hard_mode == "confusion":
            negative_mask_matrix = _build_negative_mask(
                confusion_neighborhoods,
                num_classes=int(train_base["labels"].shape[1]),
            )

        result = _train_variant(
            spec=spec,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            train_features=np.asarray(train_clip["features"], dtype=np.float32),
            train_labels=np.asarray(train_base["labels"], dtype=np.float32),
            val_labels=np.asarray(val_base["labels"], dtype=np.float32),
            test_labels=np.asarray(test_base["labels"], dtype=np.float32),
            negative_mask_matrix=negative_mask_matrix,
            lr_values=lr_values,
            weight_decay=float(args.weight_decay),
            lambda_proto=float(args.lambda_proto),
            lambda_hard=float(args.lambda_hard),
            margin=float(args.margin),
            temperature=float(args.temperature),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            seed=int(args.seed),
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
                    f"best_epoch={result['epoch']} lr={result['lr']} "
                    f"proto={spec.num_prototypes} hard={spec.hard_mode}"
                ),
            )
        )
        if spec.num_prototypes > 1:
            usage_rows = _build_usage_rows(
                labels=np.asarray(test_base["labels"], dtype=np.float32),
                assignments=np.asarray(result["test_assignments"], dtype=np.int64),
                class_names=class_names,
                num_prototypes=int(spec.num_prototypes),
            )
            _write_csv(output_dir / f"prototype_usage_{spec.num_prototypes}.csv", usage_rows)

    best_variant_name = max(
        variant_results,
        key=lambda name: float(variant_results[name]["result"]["bundle"]["classwise"]["val"]["macro"]),
    )
    best_variant = variant_results[best_variant_name]
    best_spec = best_variant["spec"]
    best_result = best_variant["result"]

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "cache_dir": str(cache_dir),
        "train_config": {
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "lrs": [float(x) for x in lr_values],
            "weight_decay": float(args.weight_decay),
            "lambda_proto": float(args.lambda_proto),
            "lambda_hard": float(args.lambda_hard),
            "margin": float(args.margin),
            "temperature": float(args.temperature),
        },
        "hard_negative": {
            "confusion_topk": int(args.confusion_topk),
            "confusion_topn": int(args.confusion_topn),
            "random_topn": int(args.random_topn),
            "confusion_neighborhoods": confusion_neighborhoods,
            "random_neighborhoods": random_neighborhoods,
        },
        "baseline": _bundle_summary(baseline_bundle),
        "variants": {
            name: {
                "label": variant_results[name]["spec"].label,
                "num_prototypes": int(variant_results[name]["spec"].num_prototypes),
                "hard_mode": variant_results[name]["spec"].hard_mode,
                "best_epoch": int(variant_results[name]["result"]["epoch"]),
                "best_lr": float(variant_results[name]["result"]["lr"]),
                "bundle": _bundle_summary(variant_results[name]["result"]["bundle"]),
                "history": _json_ready(variant_results[name]["result"]["history"]),
            }
            for name in variant_results
        },
        "comparison_rows": comparison_rows,
        "best_variant": {
            "name": best_variant_name,
            "label": best_spec.label,
            "bundle": _bundle_summary(best_result["bundle"]),
        },
    }

    comparison_path = output_dir / "main_comparison.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(comparison_path, comparison_rows)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    print(
        "[CAML-MP] best_variant="
        f"{best_spec.label} | "
        f"test_macro={float(best_result['bundle']['classwise']['test']['macro']) * 100.0:.2f} | "
        f"test_hard={float(best_result['bundle']['classwise']['test']['hard']) * 100.0:.2f}"
    )


if __name__ == "__main__":
    main()
