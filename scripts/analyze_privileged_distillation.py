#!/usr/bin/env python3
"""Run privileged distillation experiments on frozen CLIP-ViT-L/14 caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.components.aslloss import AsymmetricLossOptimized

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_TEXT_DIR = PROJECT_ROOT / "logs" / "analysis" / "vlm_full_20260316"
DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "images" / "low"
)
DEFAULT_TRAIN_ANNOTATION = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
)
AGREEMENT_VALUES = [1.0 / 3.0, 2.0 / 3.0, 1.0]
SUBSET2IDS = {
    "easy": [0, 7, 19],
    "medium": [1, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 22, 26],
    "hard": [2, 5, 8, 17, 20, 21, 23, 24, 25, 27],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise-robust privileged distillation on frozen CLIP image features."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument(
        "--train-text-npz",
        type=str,
        default=str(DEFAULT_TEXT_DIR / "rationale_full_bge_features.npz"),
    )
    parser.add_argument(
        "--val-text-npz",
        type=str,
        default=str(DEFAULT_TEXT_DIR / "val_rationale_baseline_pred_bge_features.npz"),
    )
    parser.add_argument(
        "--test-text-npz",
        type=str,
        default=str(DEFAULT_TEXT_DIR / "test_rationale_baseline_pred_bge_features.npz"),
    )
    parser.add_argument("--train-annotation-file", type=str, default=str(DEFAULT_TRAIN_ANNOTATION))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260316)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-hidden-dim", type=int, default=1024)
    parser.add_argument("--student-hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--teacher-input-mode",
        type=str,
        default="text_only",
        choices=["text_only", "image_text"],
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--standard-kd-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--student-agreement-pool",
        type=str,
        default="mean",
        choices=["mean", "min"],
        help="How to pool positive-label agreement into the student gate.",
    )
    parser.add_argument(
        "--slice-agreement-pool",
        type=str,
        default="min",
        choices=["mean", "min"],
        help="How to assign each sample to an agreement slice.",
    )
    parser.add_argument(
        "--slice-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to use for agreement slice analysis.",
    )
    parser.add_argument(
        "--teacher-candidate-limit",
        type=int,
        default=30,
        help="How many teacher correction candidates to export per view.",
    )
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_privileged_distillation"
    return Path(output_dir_arg)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _set_component_seed(base_seed: int, offset: int) -> int:
    seed = int(base_seed) + int(offset)
    _set_seed(seed)
    return seed


def _load_cache_bundle(path: Path) -> Dict[str, Any]:
    bundle = np.load(path, allow_pickle=False)
    return {
        key: np.asarray(bundle[key]) if key != "image_ids" else [str(item) for item in bundle[key].tolist()]
        for key in bundle.files
    }


def _load_text_bundle(path: Path, required_keys: Sequence[str] | None = None) -> Dict[str, Any]:
    bundle = np.load(path, allow_pickle=False)
    keys = list(bundle.files) if required_keys is None else [str(key) for key in required_keys]
    loaded: Dict[str, Any] = {}
    for key in keys:
        value = bundle[key]
        loaded[key] = np.asarray(value) if key != "image_ids" else [str(item) for item in value.tolist()]
    return loaded


def _align_text_bundle_to_clip(text_bundle: Dict[str, Any], clip_bundle: Dict[str, Any]) -> Dict[str, Any]:
    id_to_idx = {str(image_id): idx for idx, image_id in enumerate(text_bundle["image_ids"])}
    row_ids = np.asarray([int(id_to_idx[str(image_id)]) for image_id in clip_bundle["image_ids"]], dtype=np.int64)
    aligned: Dict[str, Any] = {}
    for key, value in text_bundle.items():
        if key == "image_ids":
            aligned[key] = [value[idx] for idx in row_ids.tolist()]
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(text_bundle["image_ids"]):
            aligned[key] = value[row_ids]
        else:
            aligned[key] = value
    return aligned


def _maybe_limit_bundle(bundle: Dict[str, Any], max_samples: int | None) -> Dict[str, Any]:
    if max_samples is None:
        return bundle
    limit = max(0, min(int(max_samples), len(bundle["image_ids"])))
    if limit == len(bundle["image_ids"]):
        return bundle

    limited: Dict[str, Any] = {}
    for key, value in bundle.items():
        if key == "image_ids":
            limited[key] = value[:limit]
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(bundle["image_ids"]):
            limited[key] = value[:limit]
        else:
            limited[key] = value
    return limited


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
    return {"global": global_metrics, "classwise": classwise_metrics}


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


def compute_difficulty_scores(per_class_f1: np.ndarray) -> Dict[str, float]:
    return {
        "easy": float(np.mean(per_class_f1[SUBSET2IDS["easy"]])),
        "medium": float(np.mean(per_class_f1[SUBSET2IDS["medium"]])),
        "hard": float(np.mean(per_class_f1[SUBSET2IDS["hard"]])),
    }


def voc_ap(rec: np.ndarray, prec: np.ndarray, true_num: float) -> float:
    if true_num == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = np.maximum(mpre[idx - 1], mpre[idx])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def compute_mAP(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    sample_num = len(targets)
    class_num = scores.shape[1]
    aps: List[float] = []

    for class_id in range(class_num):
        confidence = scores[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_label = targets[sorted_ind, class_id]

        tp = np.zeros(sample_num, dtype=np.float32)
        fp = np.zeros(sample_num, dtype=np.float32)
        for idx in range(sample_num):
            tp[idx] = float(sorted_label[idx] > 0)
            fp[idx] = float(sorted_label[idx] <= 0)
        true_num = float(tp.sum())
        if true_num == 0:
            aps.append(0.0)
            continue
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / true_num
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        aps.append(voc_ap(rec, prec, true_num) * 100.0)

    return float(np.mean(np.asarray(aps, dtype=np.float32)))


def compute_f1(
    multihot_targets: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    use_inference_strategy: bool = False,
    class_thresholds: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    if class_thresholds is not None:
        predict_labels = scores > class_thresholds
    else:
        predict_labels = scores > float(threshold)

    if use_inference_strategy:
        for row_idx in range(predict_labels.shape[0]):
            if predict_labels[row_idx].sum() == 0:
                max_idx = int(np.argmax(scores[row_idx]))
                predict_labels[row_idx, max_idx] = 1.0
    predict_labels = predict_labels.astype(int)

    micro = f1_score(multihot_targets, predict_labels, average="micro", zero_division=0)
    samples = f1_score(multihot_targets, predict_labels, average="samples", zero_division=0)
    macro = f1_score(multihot_targets, predict_labels, average="macro", zero_division=0)
    none = f1_score(multihot_targets, predict_labels, average=None, zero_division=0)
    return float(micro), float(samples), float(macro), np.asarray(none, dtype=np.float32)


def get_best_f1_scores(
    multihot_targets: np.ndarray,
    scores: np.ndarray,
    threshold_end: float = 0.05,
    use_inference_strategy: bool = False,
    class_thresholds: np.ndarray | None = None,
) -> Dict[str, Any]:
    if class_thresholds is not None:
        micro, samples, macro, none = compute_f1(
            multihot_targets,
            scores,
            threshold=0.5,
            use_inference_strategy=use_inference_strategy,
            class_thresholds=class_thresholds,
        )
        return {
            "micro": micro,
            "macro": macro,
            "samples": samples,
            "none": none,
            "threshold": float(np.mean(class_thresholds)),
        }

    thresholds = np.linspace(
        threshold_end,
        0.95,
        int(np.round((0.95 - threshold_end) / 0.01)) + 1,
        endpoint=True,
    )
    best: Dict[str, Any] | None = None
    for threshold in thresholds.tolist():
        micro, samples, macro, none = compute_f1(
            multihot_targets,
            scores,
            threshold=float(threshold),
            use_inference_strategy=use_inference_strategy,
        )
        if best is None or macro > float(best["macro"]) + 1e-12:
            best = {
                "micro": micro,
                "macro": macro,
                "samples": samples,
                "none": none,
                "threshold": float(threshold),
            }
    if best is None:
        raise RuntimeError("Threshold search failed to produce a result.")
    return best


def eval_validation_set(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    use_inference_strategy: bool = False,
    class_thresholds: np.ndarray | None = None,
) -> Dict[str, Any]:
    f1_dict = get_best_f1_scores(
        val_targets,
        val_scores,
        use_inference_strategy=use_inference_strategy,
        class_thresholds=class_thresholds,
    )
    mAP = compute_mAP(val_scores, val_targets)
    difficulty = compute_difficulty_scores(f1_dict["none"])
    return {
        "val_micro": float(f1_dict["micro"]),
        "val_samples": float(f1_dict["samples"]),
        "val_macro": float(f1_dict["macro"]),
        "val_none": np.asarray(f1_dict["none"], dtype=np.float32),
        "val_mAP": float(mAP),
        "threshold": float(f1_dict["threshold"]),
        "val_easy": float(difficulty["easy"]),
        "val_medium": float(difficulty["medium"]),
        "val_hard": float(difficulty["hard"]),
    }


def evaluate_fixed_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    use_inference_strategy: bool = False,
) -> Dict[str, Any]:
    micro, samples, macro, per_class = compute_f1(
        targets,
        scores,
        threshold=float(threshold),
        use_inference_strategy=use_inference_strategy,
    )
    difficulty = compute_difficulty_scores(per_class)
    return {
        "micro": float(micro),
        "samples": float(samples),
        "macro": float(macro),
        "per_class_f1": np.asarray(per_class, dtype=np.float32),
        "mAP": float(compute_mAP(scores, targets)),
        "threshold": float(threshold),
        "easy": float(difficulty["easy"]),
        "medium": float(difficulty["medium"]),
        "hard": float(difficulty["hard"]),
    }


def evaluate_with_validation_threshold(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_scores: np.ndarray | None = None,
    test_targets: np.ndarray | None = None,
    use_inference_strategy: bool = False,
) -> Dict[str, Dict[str, Any]]:
    val_metrics = eval_validation_set(
        val_scores,
        val_targets,
        use_inference_strategy=use_inference_strategy,
    )
    result: Dict[str, Dict[str, Any]] = {
        "val": {
            "micro": float(val_metrics["val_micro"]),
            "samples": float(val_metrics["val_samples"]),
            "macro": float(val_metrics["val_macro"]),
            "per_class_f1": np.asarray(val_metrics["val_none"], dtype=np.float32),
            "mAP": float(val_metrics["val_mAP"]),
            "threshold": float(val_metrics["threshold"]),
            "easy": float(val_metrics["val_easy"]),
            "medium": float(val_metrics["val_medium"]),
            "hard": float(val_metrics["val_hard"]),
        }
    }
    if test_scores is not None and test_targets is not None:
        result["test"] = evaluate_fixed_threshold(
            test_scores,
            test_targets,
            threshold=float(val_metrics["threshold"]),
            use_inference_strategy=use_inference_strategy,
        )
    return result


def search_classwise_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)
    if scores.shape != targets.shape:
        raise ValueError(f"scores shape {scores.shape} != targets shape {targets.shape}")

    grid = np.arange(0.05, 0.951, 0.01, dtype=np.float32)
    num_classes = scores.shape[1]
    thresholds = np.zeros(num_classes, dtype=np.float32)

    for class_idx in range(num_classes):
        y_true = targets[:, class_idx]
        best_thr = float(grid[0])
        best_f1 = -1.0
        for threshold in grid.tolist():
            y_pred = (scores[:, class_idx] > float(threshold)).astype(np.int32)
            f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
            if f1 > best_f1 + 1e-12:
                best_f1 = float(f1)
                best_thr = float(threshold)
        thresholds[class_idx] = best_thr
    return thresholds


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _logit_np(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.asarray(values, dtype=np.float32).clip(min=eps, max=1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _bernoulli_kl_per_class(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    scaled_logits = student_logits / float(temperature)
    student_probs = torch.sigmoid(scaled_logits).clamp(min=eps, max=1.0 - eps)
    teacher_probs = teacher_probs.clamp(min=eps, max=1.0 - eps)
    kl = (
        teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs))
        + (1.0 - teacher_probs)
        * (torch.log(1.0 - teacher_probs) - torch.log(1.0 - student_probs))
    )
    return kl * (float(temperature) ** 2)


def _compute_sample_agreement(
    labels: np.ndarray,
    soft_labels: np.ndarray,
    mode: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    soft_labels = np.asarray(soft_labels, dtype=np.float32)
    positive_mask = labels > 0.5
    if not np.any(positive_mask):
        return np.ones(labels.shape[0], dtype=np.float32)

    pooled = np.ones(labels.shape[0], dtype=np.float32)
    for sample_idx in range(labels.shape[0]):
        class_ids = np.where(positive_mask[sample_idx])[0]
        if len(class_ids) == 0:
            pooled[sample_idx] = 1.0
            continue
        values = soft_labels[sample_idx, class_ids]
        if mode == "mean":
            pooled[sample_idx] = float(values.mean())
        elif mode == "min":
            pooled[sample_idx] = float(values.min())
        else:
            raise ValueError(f"Unsupported agreement pooling mode: {mode}")
    return pooled.astype(np.float32)


def _has_non_binary_soft_labels(soft_labels: np.ndarray) -> bool:
    rounded = np.unique(np.round(np.asarray(soft_labels, dtype=np.float32), 6))
    return any(not np.isclose(value, 0.0) and not np.isclose(value, 1.0) for value in rounded.tolist())


def _assign_agreement_groups(
    labels: np.ndarray,
    soft_labels: np.ndarray,
    mode: str,
) -> np.ndarray:
    pooled = _compute_sample_agreement(labels=labels, soft_labels=soft_labels, mode=mode)
    groups = np.zeros(pooled.shape[0], dtype=np.float32)
    for idx, value in enumerate(pooled.tolist()):
        nearest = min(AGREEMENT_VALUES, key=lambda target: abs(float(target) - float(value)))
        groups[idx] = float(nearest)
    return groups


def _format_agreement(value: float) -> str:
    if math.isclose(float(value), 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-4):
        return "1/3"
    if math.isclose(float(value), 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-4):
        return "2/3"
    return "1"


def _slice_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    agreements: np.ndarray,
    class_thresholds: np.ndarray,
    method: str,
    split: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for agreement_value in AGREEMENT_VALUES:
        mask = np.isclose(agreements, float(agreement_value), atol=1e-6)
        count = int(mask.sum())
        if count == 0:
            continue
        micro, samples, macro, per_class = compute_f1(
            targets[mask],
            scores[mask],
            threshold=0.5,
            use_inference_strategy=False,
            class_thresholds=class_thresholds,
        )
        difficulty = compute_difficulty_scores(per_class)
        rows.append(
            {
                "split": split,
                "method": method,
                "agreement": _format_agreement(float(agreement_value)),
                "num_samples": count,
                "macro": float(macro) * 100.0,
                "micro": float(micro) * 100.0,
                "samples": float(samples) * 100.0,
                "mAP": float(compute_mAP(scores[mask], targets[mask])),
                "hard": float(difficulty["hard"]) * 100.0,
            }
        )
    return rows


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


def _load_train_metadata(annotation_file: Path, image_dir: Path) -> tuple[List[str], Dict[str, str], Dict[int, str]]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    class_names = [str(item["name"]) for item in categories]
    class_name_map = {idx: name for idx, name in enumerate(class_names)}

    image_lookup: Dict[str, str] = {}
    for item in data["images"]:
        filename = str(item["filename"])
        if filename.startswith("low/"):
            filename = filename[4:]
        image_lookup[str(item["id"])] = str((image_dir / filename).resolve())
    return class_names, image_lookup, class_name_map


def _labels_to_names(labels: np.ndarray, class_names: Sequence[str]) -> List[str]:
    label_ids = np.where(np.asarray(labels, dtype=np.float32) > 0.5)[0].tolist()
    return [str(class_names[idx]) for idx in label_ids]


class TeacherDataset(Dataset):
    def __init__(
        self,
        text_features: np.ndarray,
        labels: np.ndarray,
        image_features: np.ndarray | None = None,
    ) -> None:
        self.text_features = np.asarray(text_features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.image_features = None
        if image_features is not None:
            self.image_features = np.asarray(image_features, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "text_features": torch.from_numpy(self.text_features[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
        }
        if self.image_features is not None:
            sample["image_features"] = torch.from_numpy(self.image_features[idx])
        return sample


class StudentDataset(Dataset):
    def __init__(
        self,
        image_features: np.ndarray,
        labels: np.ndarray,
        agreement: np.ndarray,
        teacher_probs: np.ndarray,
    ) -> None:
        self.image_features = np.asarray(image_features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.agreement = np.asarray(agreement, dtype=np.float32)
        self.teacher_probs = np.asarray(teacher_probs, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image_features": torch.from_numpy(self.image_features[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "agreement": torch.tensor(float(self.agreement[idx]), dtype=torch.float32),
            "teacher_probs": torch.from_numpy(self.teacher_probs[idx]),
        }


class TeacherMLP(nn.Module):
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        input_mode: str = "text_only",
        image_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_mode not in {"text_only", "image_text"}:
            raise ValueError(f"Unsupported teacher input mode: {input_mode}")
        if input_mode == "image_text" and image_dim is None:
            raise ValueError("image_dim must be provided when teacher input mode is image_text.")

        self.input_mode = str(input_mode)
        input_dim = int(text_dim) if self.input_mode == "text_only" else int(text_dim) + int(image_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_mode == "text_only":
            inputs = text_features
        else:
            if image_features is None:
                raise ValueError("image_features is required when teacher input mode is image_text.")
            inputs = torch.cat([image_features, text_features], dim=1)
        return self.net(inputs)


class StudentMLP(nn.Module):
    def __init__(
        self,
        image_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.net(image_features)


@torch.inference_mode()
def _predict_teacher(
    model: TeacherMLP,
    text_features: np.ndarray,
    device: torch.device,
    batch_size: int,
    image_features: np.ndarray | None = None,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    for start in range(0, text_features.shape[0], int(batch_size)):
        text_batch = torch.as_tensor(
            text_features[start : start + int(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        image_batch = None
        if image_features is not None:
            image_batch = torch.as_tensor(
                image_features[start : start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
        logits = model(text_features=text_batch, image_features=image_batch)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


@torch.inference_mode()
def _predict_student(
    model: StudentMLP,
    image_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    for start in range(0, image_features.shape[0], int(batch_size)):
        image_batch = torch.as_tensor(
            image_features[start : start + int(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        logits = model(image_batch)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _train_teacher(
    model: TeacherMLP,
    train_dataset: TeacherDataset,
    val_text_features: np.ndarray,
    val_targets: np.ndarray,
    test_text_features: np.ndarray,
    test_targets: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    val_image_features: np.ndarray | None = None,
    test_image_features: np.ndarray | None = None,
) -> Dict[str, Any]:
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = AsymmetricLossOptimized(
        gamma_neg=2,
        gamma_pos=0,
        clip=0.05,
        eps=1e-5,
        disable_torch_grad_focal_loss=False,
    )

    best_bundle: Dict[str, Any] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_macro = float("-inf")
    stale_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            text_features = batch["text_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            image_features = None
            if "image_features" in batch:
                image_features = batch["image_features"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(text_features=text_features, image_features=image_features)
            loss = criterion(logits, labels, reduction="mean")
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batch_count += 1

        val_scores = _predict_teacher(
            model=model,
            text_features=val_text_features,
            device=device,
            batch_size=int(args.batch_size),
            image_features=val_image_features,
        )
        test_scores = _predict_teacher(
            model=model,
            text_features=test_text_features,
            device=device,
            batch_size=int(args.batch_size),
            image_features=test_image_features,
        )
        bundle = _evaluate_score_bundle(
            val_scores=val_scores,
            val_targets=val_targets,
            test_scores=test_scores,
            test_targets=test_targets,
        )
        val_macro = float(bundle["classwise"]["val"]["macro"])
        history.append(
            {
                "epoch": int(epoch),
                "loss": total_loss / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
            }
        )
        print(
            "[Distill][Teacher] epoch "
            f"{epoch:02d} | val_macro={val_macro*100.0:.2f} | "
            f"test_macro={float(bundle['classwise']['test']['macro'])*100.0:.2f} | "
            f"test_hard={float(bundle['classwise']['test']['hard'])*100.0:.2f}"
        )
        if val_macro > best_val_macro + 1e-9:
            best_val_macro = val_macro
            best_bundle = bundle
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    if best_bundle is None or best_state is None:
        raise RuntimeError("Teacher training did not produce a valid best checkpoint.")

    model.load_state_dict(best_state)
    return {
        "bundle": best_bundle,
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }


def _train_student(
    mode: str,
    model: StudentMLP,
    train_dataset: StudentDataset,
    val_image_features: np.ndarray,
    val_targets: np.ndarray,
    test_image_features: np.ndarray,
    test_targets: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if mode not in {"baseline", "standard_kd", "dynamic_kd"}:
        raise ValueError(f"Unsupported student mode: {mode}")

    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = AsymmetricLossOptimized(
        gamma_neg=2,
        gamma_pos=0,
        clip=0.05,
        eps=1e-5,
        disable_torch_grad_focal_loss=False,
    )

    best_bundle: Dict[str, Any] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_macro = float("-inf")
    stale_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_supervised = 0.0
        total_kd = 0.0
        total_gate = 0.0
        batch_count = 0
        for batch in loader:
            image_features = batch["image_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            agreement = batch["agreement"].to(device, non_blocking=True)
            teacher_probs = batch["teacher_probs"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image_features)
            supervised_per_class = criterion(logits, labels, reduction="none")
            supervised_per_sample = supervised_per_class.sum(dim=1)

            kd_per_class = _bernoulli_kl_per_class(
                student_logits=logits,
                teacher_probs=teacher_probs,
                temperature=float(args.temperature),
            )
            kd_per_sample = kd_per_class.sum(dim=1)

            if mode == "baseline":
                loss_per_sample = supervised_per_sample
                gate_value = torch.ones_like(agreement)
            elif mode == "standard_kd":
                loss_per_sample = supervised_per_sample + float(args.standard_kd_weight) * kd_per_sample
                gate_value = torch.full_like(agreement, fill_value=1.0)
            else:
                loss_per_sample = (
                    agreement * supervised_per_sample
                    + (1.0 - agreement) * float(args.dynamic_kd_weight) * kd_per_sample
                )
                gate_value = agreement

            loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_supervised += float(supervised_per_sample.mean().detach().cpu())
            total_kd += float(kd_per_sample.mean().detach().cpu())
            total_gate += float(gate_value.mean().detach().cpu())
            batch_count += 1

        val_scores = _predict_student(
            model=model,
            image_features=val_image_features,
            device=device,
            batch_size=int(args.batch_size),
        )
        test_scores = _predict_student(
            model=model,
            image_features=test_image_features,
            device=device,
            batch_size=int(args.batch_size),
        )
        bundle = _evaluate_score_bundle(
            val_scores=val_scores,
            val_targets=val_targets,
            test_scores=test_scores,
            test_targets=test_targets,
        )
        val_macro = float(bundle["classwise"]["val"]["macro"])
        history.append(
            {
                "epoch": int(epoch),
                "loss": total_loss / max(1, batch_count),
                "supervised_loss": total_supervised / max(1, batch_count),
                "kd_loss": total_kd / max(1, batch_count),
                "mean_gate": total_gate / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
            }
        )
        print(
            f"[Distill][{mode}] epoch "
            f"{epoch:02d} | val_macro={val_macro*100.0:.2f} | "
            f"test_macro={float(bundle['classwise']['test']['macro'])*100.0:.2f} | "
            f"test_hard={float(bundle['classwise']['test']['hard'])*100.0:.2f}"
        )
        if val_macro > best_val_macro + 1e-9:
            best_val_macro = val_macro
            best_bundle = bundle
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    if best_bundle is None or best_state is None:
        raise RuntimeError(f"{mode} training did not produce a valid best checkpoint.")

    model.load_state_dict(best_state)
    return {
        "bundle": best_bundle,
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }


def _build_teacher_candidates(
    train_image_ids: Sequence[str],
    train_labels: np.ndarray,
    train_soft_labels: np.ndarray,
    train_scores: np.ndarray,
    train_text_bundle: Mapping[str, Any],
    image_lookup: Mapping[str, str],
    class_names: Sequence[str],
    limit: int,
) -> Dict[str, List[Dict[str, Any]]]:
    labels = np.asarray(train_labels, dtype=np.float32)
    soft_labels = np.asarray(train_soft_labels, dtype=np.float32)
    scores = np.asarray(train_scores, dtype=np.float32)

    low_agreement_mask = np.isclose(
        _assign_agreement_groups(labels=labels, soft_labels=soft_labels, mode="min"),
        1.0 / 3.0,
        atol=1e-6,
    )
    low_rows = np.where(low_agreement_mask)[0].tolist()

    reject_rows: List[Dict[str, Any]] = []
    add_rows: List[Dict[str, Any]] = []
    for row_idx in low_rows:
        image_id = str(train_image_ids[row_idx])
        target_names = _labels_to_names(labels[row_idx], class_names)
        if not target_names:
            continue
        positive_ids = np.where(labels[row_idx] > 0.5)[0]
        sorted_positive_ids = sorted(
            positive_ids.tolist(),
            key=lambda idx: (float(soft_labels[row_idx, idx]), float(scores[row_idx, idx])),
        )
        weakest_id = int(sorted_positive_ids[0])
        teacher_reject_score = float(1.0 - scores[row_idx, weakest_id])
        reject_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_lookup.get(image_id, "")),
                "hard_labels": ", ".join(target_names),
                "agreement_values": ", ".join(
                    f"{class_names[idx]}={soft_labels[row_idx, idx]:.3f}" for idx in positive_ids.tolist()
                ),
                "teacher_reject_class": str(class_names[weakest_id]),
                "teacher_reject_prob": float(scores[row_idx, weakest_id]),
                "teacher_reject_priority": teacher_reject_score,
                "top_teacher_predictions": ", ".join(
                    f"{class_names[idx]}={scores[row_idx, idx]:.3f}"
                    for idx in np.argsort(-scores[row_idx])[:5].tolist()
                ),
                "rationale_text": str(train_text_bundle["texts"][row_idx]),
            }
        )

        candidate_ids = [idx for idx in np.argsort(-scores[row_idx]).tolist() if labels[row_idx, idx] <= 0.5]
        if not candidate_ids:
            continue
        best_new_id = int(candidate_ids[0])
        add_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_lookup.get(image_id, "")),
                "hard_labels": ", ".join(target_names),
                "agreement_values": ", ".join(
                    f"{class_names[idx]}={soft_labels[row_idx, idx]:.3f}" for idx in positive_ids.tolist()
                ),
                "teacher_add_class": str(class_names[best_new_id]),
                "teacher_add_prob": float(scores[row_idx, best_new_id]),
                "teacher_add_priority": float(scores[row_idx, best_new_id]),
                "top_teacher_predictions": ", ".join(
                    f"{class_names[idx]}={scores[row_idx, idx]:.3f}"
                    for idx in np.argsort(-scores[row_idx])[:5].tolist()
                ),
                "rationale_text": str(train_text_bundle["texts"][row_idx]),
            }
        )

    reject_rows = sorted(reject_rows, key=lambda item: float(item["teacher_reject_priority"]), reverse=True)
    add_rows = sorted(add_rows, key=lambda item: float(item["teacher_add_priority"]), reverse=True)
    return {
        "teacher_rejects_uncertain_positive": reject_rows[: int(limit)],
        "teacher_adds_new_label": add_rows[: int(limit)],
    }


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))
    print(f"[Distill] output_dir={output_dir} device={device}")

    cache_dir = Path(args.reuse_cache_dir)
    print(f"[Distill] loading base cache from {cache_dir}")
    train_clip = _maybe_limit_bundle(
        _load_cache_bundle(cache_dir / "train_clip.npz"),
        args.max_train_samples,
    )
    val_clip = _maybe_limit_bundle(
        _load_cache_bundle(cache_dir / "val_clip.npz"),
        args.max_val_samples,
    )
    test_clip = _maybe_limit_bundle(
        _load_cache_bundle(cache_dir / "test_clip.npz"),
        args.max_test_samples,
    )
    print(
        "[Distill] cache sizes "
        f"train={len(train_clip['image_ids'])} "
        f"val={len(val_clip['image_ids'])} "
        f"test={len(test_clip['image_ids'])}"
    )

    print("[Distill] loading rationale features")
    train_text = _align_text_bundle_to_clip(
        _load_text_bundle(Path(args.train_text_npz), required_keys=["image_ids", "features", "texts"]),
        train_clip,
    )
    val_text = _align_text_bundle_to_clip(
        _load_text_bundle(Path(args.val_text_npz), required_keys=["image_ids", "features"]),
        val_clip,
    )
    test_text = _align_text_bundle_to_clip(
        _load_text_bundle(Path(args.test_text_npz), required_keys=["image_ids", "features"]),
        test_clip,
    )

    if train_clip["image_ids"] != train_text["image_ids"]:
        raise RuntimeError("Train image order mismatch after text alignment.")
    if val_clip["image_ids"] != val_text["image_ids"]:
        raise RuntimeError("Validation image order mismatch after text alignment.")
    if test_clip["image_ids"] != test_text["image_ids"]:
        raise RuntimeError("Test image order mismatch after text alignment.")
    print("[Distill] feature alignment verified")

    class_names, image_lookup, _ = _load_train_metadata(
        annotation_file=Path(args.train_annotation_file),
        image_dir=Path(args.image_dir),
    )

    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    train_soft_labels = np.asarray(train_clip["soft_labels"], dtype=np.float32)
    val_labels = np.asarray(val_clip["labels"], dtype=np.float32)
    test_labels = np.asarray(test_clip["labels"], dtype=np.float32)

    teacher_seed = _set_component_seed(int(args.seed), offset=0)
    teacher_train_dataset = TeacherDataset(
        text_features=np.asarray(train_text["features"], dtype=np.float32),
        labels=train_labels,
        image_features=(
            np.asarray(train_clip["features"], dtype=np.float32)
            if str(args.teacher_input_mode) == "image_text"
            else None
        ),
    )
    teacher = TeacherMLP(
        text_dim=int(teacher_train_dataset.text_features.shape[1]),
        hidden_dim=int(args.teacher_hidden_dim),
        num_classes=int(teacher_train_dataset.labels.shape[1]),
        dropout=float(args.dropout),
        input_mode=str(args.teacher_input_mode),
        image_dim=(
            int(teacher_train_dataset.image_features.shape[1])
            if teacher_train_dataset.image_features is not None
            else None
        ),
    ).to(device)
    print(f"[Distill] training oracle teacher mode={args.teacher_input_mode} seed={teacher_seed}")
    teacher_result = _train_teacher(
        model=teacher,
        train_dataset=teacher_train_dataset,
        val_text_features=np.asarray(val_text["features"], dtype=np.float32),
        val_targets=val_labels,
        test_text_features=np.asarray(test_text["features"], dtype=np.float32),
        test_targets=test_labels,
        device=device,
        args=args,
        val_image_features=(
            np.asarray(val_clip["features"], dtype=np.float32)
            if str(args.teacher_input_mode) == "image_text"
            else None
        ),
        test_image_features=(
            np.asarray(test_clip["features"], dtype=np.float32)
            if str(args.teacher_input_mode) == "image_text"
            else None
        ),
    )

    teacher_train_scores = _predict_teacher(
        model=teacher,
        text_features=np.asarray(train_text["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
        image_features=(
            np.asarray(train_clip["features"], dtype=np.float32)
            if str(args.teacher_input_mode) == "image_text"
            else None
        ),
    )
    teacher_val_scores = _predict_teacher(
        model=teacher,
        text_features=np.asarray(val_text["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
        image_features=(
            np.asarray(val_clip["features"], dtype=np.float32)
            if str(args.teacher_input_mode) == "image_text"
            else None
        ),
    )
    teacher_test_scores = _predict_teacher(
        model=teacher,
        text_features=np.asarray(test_text["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
        image_features=(
            np.asarray(test_clip["features"], dtype=np.float32)
            if str(args.teacher_input_mode) == "image_text"
            else None
        ),
    )

    teacher_probs_train = _sigmoid_np(_logit_np(teacher_train_scores) / float(args.temperature))

    student_agreement = _compute_sample_agreement(
        labels=train_labels,
        soft_labels=train_soft_labels,
        mode=str(args.student_agreement_pool),
    )

    baseline_dataset = StudentDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        labels=train_labels,
        agreement=np.ones_like(student_agreement, dtype=np.float32),
        teacher_probs=np.zeros_like(train_labels, dtype=np.float32),
    )
    standard_kd_dataset = StudentDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        labels=train_labels,
        agreement=np.ones_like(student_agreement, dtype=np.float32),
        teacher_probs=teacher_probs_train,
    )
    dynamic_kd_dataset = StudentDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        labels=train_labels,
        agreement=student_agreement,
        teacher_probs=teacher_probs_train,
    )

    baseline_seed = _set_component_seed(int(args.seed), offset=100)
    baseline_model = StudentMLP(
        image_dim=int(baseline_dataset.image_features.shape[1]),
        hidden_dim=int(args.student_hidden_dim),
        num_classes=int(baseline_dataset.labels.shape[1]),
        dropout=float(args.dropout),
    ).to(device)
    print(f"[Distill] training baseline student seed={baseline_seed}")
    baseline_result = _train_student(
        mode="baseline",
        model=baseline_model,
        train_dataset=baseline_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_targets=val_labels,
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_targets=test_labels,
        device=device,
        args=args,
    )

    standard_kd_seed = _set_component_seed(int(args.seed), offset=200)
    standard_kd_model = StudentMLP(
        image_dim=int(standard_kd_dataset.image_features.shape[1]),
        hidden_dim=int(args.student_hidden_dim),
        num_classes=int(standard_kd_dataset.labels.shape[1]),
        dropout=float(args.dropout),
    ).to(device)
    print(f"[Distill] training standard KD student seed={standard_kd_seed}")
    standard_kd_result = _train_student(
        mode="standard_kd",
        model=standard_kd_model,
        train_dataset=standard_kd_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_targets=val_labels,
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_targets=test_labels,
        device=device,
        args=args,
    )

    dynamic_kd_seed = _set_component_seed(int(args.seed), offset=300)
    dynamic_kd_model = StudentMLP(
        image_dim=int(dynamic_kd_dataset.image_features.shape[1]),
        hidden_dim=int(args.student_hidden_dim),
        num_classes=int(dynamic_kd_dataset.labels.shape[1]),
        dropout=float(args.dropout),
    ).to(device)
    print(f"[Distill] training dynamic gated KD student seed={dynamic_kd_seed}")
    dynamic_kd_result = _train_student(
        mode="dynamic_kd",
        model=dynamic_kd_model,
        train_dataset=dynamic_kd_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_targets=val_labels,
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_targets=test_labels,
        device=device,
        args=args,
    )

    baseline_train_scores = _predict_student(
        model=baseline_model,
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )
    standard_kd_train_scores = _predict_student(
        model=standard_kd_model,
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )
    dynamic_kd_train_scores = _predict_student(
        model=dynamic_kd_model,
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )

    comparison_rows = [
        _comparison_row("oracle_teacher", teacher_result["bundle"], note=f"best_epoch={teacher_result['best_epoch']}"),
        _comparison_row("baseline", baseline_result["bundle"], note=f"best_epoch={baseline_result['best_epoch']}"),
        _comparison_row("standard_kd", standard_kd_result["bundle"], note=f"best_epoch={standard_kd_result['best_epoch']}"),
        _comparison_row("dynamic_gated_kd", dynamic_kd_result["bundle"], note=f"best_epoch={dynamic_kd_result['best_epoch']}"),
    ]
    _write_csv(output_dir / "main_comparison.csv", comparison_rows)

    teacher_candidates = _build_teacher_candidates(
        train_image_ids=train_clip["image_ids"],
        train_labels=train_labels,
        train_soft_labels=train_soft_labels,
        train_scores=teacher_train_scores,
        train_text_bundle=train_text,
        image_lookup=image_lookup,
        class_names=class_names,
        limit=int(args.teacher_candidate_limit),
    )
    for name, rows in teacher_candidates.items():
        _write_csv(output_dir / f"{name}.csv", rows)

    slice_rows: List[Dict[str, Any]] = []
    slice_note = ""
    slice_source_scores: Dict[str, np.ndarray] = {
        "baseline": baseline_train_scores,
        "standard_kd": standard_kd_train_scores,
        "dynamic_gated_kd": dynamic_kd_train_scores,
    }
    slice_targets = train_labels
    slice_soft_labels = train_soft_labels
    if args.slice_split == "train":
        slice_note = "train split used because public val/test annotations do not expose agreement soft labels."
    else:
        source_bundle = {"val": val_clip, "test": test_clip}[str(args.slice_split)]
        slice_targets = np.asarray(source_bundle["labels"], dtype=np.float32)
        slice_soft_labels = np.asarray(source_bundle["soft_labels"], dtype=np.float32)
        if not _has_non_binary_soft_labels(slice_soft_labels):
            slice_note = (
                f"{args.slice_split} split lacks agreement soft labels in the available cache; "
                "falling back to train slice analysis."
            )
            slice_targets = train_labels
            slice_soft_labels = train_soft_labels
        else:
            slice_source_scores = {
                "baseline": _predict_student(
                    model=baseline_model,
                    image_features=np.asarray(source_bundle["features"], dtype=np.float32),
                    device=device,
                    batch_size=int(args.batch_size),
                ),
                "standard_kd": _predict_student(
                    model=standard_kd_model,
                    image_features=np.asarray(source_bundle["features"], dtype=np.float32),
                    device=device,
                    batch_size=int(args.batch_size),
                ),
                "dynamic_gated_kd": _predict_student(
                    model=dynamic_kd_model,
                    image_features=np.asarray(source_bundle["features"], dtype=np.float32),
                    device=device,
                    batch_size=int(args.batch_size),
                ),
            }
            slice_note = f"{args.slice_split} split slice analysis."

    slice_agreements = _assign_agreement_groups(
        labels=slice_targets,
        soft_labels=slice_soft_labels,
        mode=str(args.slice_agreement_pool),
    )
    baseline_thresholds = np.asarray(
        baseline_result["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )
    standard_kd_thresholds = np.asarray(
        standard_kd_result["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )
    dynamic_kd_thresholds = np.asarray(
        dynamic_kd_result["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )
    slice_rows.extend(
        _slice_metrics(
            scores=slice_source_scores["baseline"],
            targets=slice_targets,
            agreements=slice_agreements,
            class_thresholds=baseline_thresholds,
            method="baseline",
            split="train" if "falling back" in slice_note or args.slice_split == "train" else str(args.slice_split),
        )
    )
    slice_rows.extend(
        _slice_metrics(
            scores=slice_source_scores["standard_kd"],
            targets=slice_targets,
            agreements=slice_agreements,
            class_thresholds=standard_kd_thresholds,
            method="standard_kd",
            split="train" if "falling back" in slice_note or args.slice_split == "train" else str(args.slice_split),
        )
    )
    slice_rows.extend(
        _slice_metrics(
            scores=slice_source_scores["dynamic_gated_kd"],
            targets=slice_targets,
            agreements=slice_agreements,
            class_thresholds=dynamic_kd_thresholds,
            method="dynamic_gated_kd",
            split="train" if "falling back" in slice_note or args.slice_split == "train" else str(args.slice_split),
        )
    )
    _write_csv(output_dir / "agreement_slice_analysis.csv", slice_rows)

    torch.save(teacher_result["state_dict"], output_dir / "teacher_best.pt")
    torch.save(baseline_result["state_dict"], output_dir / "baseline_best.pt")
    torch.save(standard_kd_result["state_dict"], output_dir / "standard_kd_best.pt")
    torch.save(dynamic_kd_result["state_dict"], output_dir / "dynamic_gated_kd_best.pt")

    teacher_summary = {key: value for key, value in teacher_result.items() if key != "state_dict"}
    baseline_summary = {key: value for key, value in baseline_result.items() if key != "state_dict"}
    standard_kd_summary = {key: value for key, value in standard_kd_result.items() if key != "state_dict"}
    dynamic_kd_summary = {key: value for key, value in dynamic_kd_result.items() if key != "state_dict"}

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "device": str(device),
                "comparison_rows": comparison_rows,
                "teacher": _json_ready(teacher_summary),
                "baseline": _json_ready(baseline_summary),
                "standard_kd": _json_ready(standard_kd_summary),
                "dynamic_gated_kd": _json_ready(dynamic_kd_summary),
                "teacher_candidates": {
                    key: len(value) for key, value in teacher_candidates.items()
                },
                "agreement_slice_note": slice_note,
                "agreement_slice_rows": slice_rows,
                "config": {
                    "max_train_samples": None if args.max_train_samples is None else int(args.max_train_samples),
                    "max_val_samples": None if args.max_val_samples is None else int(args.max_val_samples),
                    "max_test_samples": None if args.max_test_samples is None else int(args.max_test_samples),
                    "base_seed": int(args.seed),
                    "teacher_seed": int(args.seed) + 0,
                    "baseline_seed": int(args.seed) + 100,
                    "standard_kd_seed": int(args.seed) + 200,
                    "dynamic_kd_seed": int(args.seed) + 300,
                    "teacher_input_mode": str(args.teacher_input_mode),
                    "standard_kd_weight": float(args.standard_kd_weight),
                    "dynamic_kd_weight": float(args.dynamic_kd_weight),
                    "temperature": float(args.temperature),
                    "student_agreement_pool": str(args.student_agreement_pool),
                    "slice_agreement_pool": str(args.slice_agreement_pool),
                    "slice_split": str(args.slice_split),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Distill] finished. artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
