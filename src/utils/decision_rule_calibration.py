from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score


DEFAULT_THRESHOLD_GRID = np.arange(0.05, 0.951, 0.01, dtype=np.float32)


def search_classwise_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Search one threshold per class by maximizing that class F1 on validation."""
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)
    if scores.shape != targets.shape:
        raise ValueError(f"scores shape {scores.shape} != targets shape {targets.shape}")

    grid = DEFAULT_THRESHOLD_GRID if threshold_grid is None else np.asarray(threshold_grid, dtype=np.float32)
    num_classes = scores.shape[1]
    thresholds = np.zeros(num_classes, dtype=np.float32)

    for class_idx in range(num_classes):
        y_true = targets[:, class_idx]
        best_thr = float(grid[0])
        best_f1 = -1.0
        for thr in grid:
            y_pred = (scores[:, class_idx] > float(thr)).astype(np.int32)
            f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
            if f1 > best_f1 + 1e-12:
                best_f1 = float(f1)
                best_thr = float(thr)
        thresholds[class_idx] = best_thr

    return thresholds


def search_groupwise_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    groups: Dict[str, Sequence[int]],
    threshold_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Search one shared threshold per group by maximizing mean per-class F1 within that group."""
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)
    if scores.shape != targets.shape:
        raise ValueError(f"scores shape {scores.shape} != targets shape {targets.shape}")

    grid = DEFAULT_THRESHOLD_GRID if threshold_grid is None else np.asarray(threshold_grid, dtype=np.float32)
    num_classes = scores.shape[1]
    class_thresholds = np.zeros(num_classes, dtype=np.float32)
    group_thresholds: Dict[str, float] = {}

    for group_name, class_ids in groups.items():
        ids = [int(idx) for idx in class_ids]
        if not ids:
            continue
        best_thr = float(grid[0])
        best_score = -1.0
        for thr in grid:
            per_class_scores = []
            for class_idx in ids:
                y_true = targets[:, class_idx]
                y_pred = (scores[:, class_idx] > float(thr)).astype(np.int32)
                per_class_scores.append(
                    f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
                )
            mean_score = float(np.mean(per_class_scores)) if per_class_scores else 0.0
            if mean_score > best_score + 1e-12:
                best_score = mean_score
                best_thr = float(thr)
        for class_idx in ids:
            class_thresholds[class_idx] = best_thr
        group_thresholds[group_name] = best_thr

    return class_thresholds, group_thresholds


def build_prior_benefit_groups(class_gains: np.ndarray) -> Dict[str, list[int]]:
    """Split classes into prior-benefit vs prior-neutral/risk groups."""
    class_gains = np.asarray(class_gains, dtype=np.float32)
    benefit = [int(idx) for idx, gain in enumerate(class_gains.tolist()) if gain > 0.0]
    neutral_risk = [int(idx) for idx, gain in enumerate(class_gains.tolist()) if gain <= 0.0]
    return {
        "prior_benefit": benefit,
        "prior_neutral_or_risk": neutral_risk,
    }


def build_head_medium_tail_groups(
    positive_counts: np.ndarray,
) -> Dict[str, list[int]]:
    """Split classes into head / medium / tail by descending positive counts."""
    positive_counts = np.asarray(positive_counts, dtype=np.float32)
    order = np.argsort(-positive_counts)
    num_classes = positive_counts.shape[0]

    head_end = int(np.ceil(num_classes / 3.0))
    medium_end = int(np.ceil(2.0 * num_classes / 3.0))

    return {
        "head": [int(idx) for idx in order[:head_end].tolist()],
        "medium": [int(idx) for idx in order[head_end:medium_end].tolist()],
        "tail": [int(idx) for idx in order[medium_end:].tolist()],
    }
