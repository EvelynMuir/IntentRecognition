from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.utils.metrics import compute_difficulty_scores, compute_f1, compute_mAP, eval_validation_set


def normalize_scores_per_sample(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalize each sample across classes."""
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (scores - mean) / std


def aggregate_prompt_scores(
    prompt_scores: np.ndarray,
    mode: str = "average",
    top_n: int = 2,
) -> np.ndarray:
    """Aggregate multiple prompt scores for a class."""
    prompt_scores = np.asarray(prompt_scores, dtype=np.float32)
    if prompt_scores.ndim != 2:
        raise ValueError(f"prompt_scores must be 2D, got shape {prompt_scores.shape}")
    if prompt_scores.shape[1] == 0:
        raise ValueError("prompt_scores must contain at least one prompt per class.")

    if mode == "average":
        return prompt_scores.mean(axis=1)
    if mode == "max":
        return prompt_scores.max(axis=1)
    if mode == "top2_avg":
        top_n = max(1, min(int(top_n), prompt_scores.shape[1]))
        part = np.partition(prompt_scores, kth=prompt_scores.shape[1] - top_n, axis=1)
        return part[:, -top_n:].mean(axis=1)
    if mode == "logsumexp":
        row_max = prompt_scores.max(axis=1, keepdims=True)
        stabilized = prompt_scores - row_max
        return (
            np.log(np.exp(stabilized).sum(axis=1))
            + row_max.squeeze(axis=1)
            - np.log(float(prompt_scores.shape[1]))
        )

    raise ValueError(f"Unsupported prompt aggregation mode: {mode}")


def apply_topk_rerank_fusion(
    baseline_logits: np.ndarray,
    prior_scores: np.ndarray,
    topk: int,
    alpha: float,
    mode: str,
) -> np.ndarray:
    """Fuse prior scores into only the top-k baseline candidates."""
    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    prior_scores = np.asarray(prior_scores, dtype=np.float32)
    if baseline_logits.shape != prior_scores.shape:
        raise ValueError(
            f"baseline_logits shape {baseline_logits.shape} != prior_scores shape {prior_scores.shape}"
        )

    num_classes = baseline_logits.shape[1]
    topk = max(1, min(int(topk), num_classes))
    output = baseline_logits.copy()

    if mode == "add":
        fused_prior = prior_scores
    elif mode == "mix":
        fused_prior = prior_scores
    elif mode == "add_norm":
        fused_prior = normalize_scores_per_sample(prior_scores)
    else:
        raise ValueError(f"Unsupported rerank fusion mode: {mode}")

    topk_idx = np.argpartition(-baseline_logits, kth=topk - 1, axis=1)[:, :topk]
    row_idx = np.arange(baseline_logits.shape[0])[:, None]

    if mode == "mix":
        output[row_idx, topk_idx] = (
            (1.0 - float(alpha)) * baseline_logits[row_idx, topk_idx]
            + float(alpha) * fused_prior[row_idx, topk_idx]
        )
    else:
        output[row_idx, topk_idx] = (
            baseline_logits[row_idx, topk_idx] + float(alpha) * fused_prior[row_idx, topk_idx]
        )

    return output


def build_topk_comparative_prior(
    baseline_logits: np.ndarray,
    prior_scores: np.ndarray,
    topk: int,
    mode: str = "none",
) -> np.ndarray:
    """Build comparative prior scores inside the top-k candidate set."""
    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    prior_scores = np.asarray(prior_scores, dtype=np.float32)
    if baseline_logits.shape != prior_scores.shape:
        raise ValueError(
            f"baseline_logits shape {baseline_logits.shape} != prior_scores shape {prior_scores.shape}"
        )

    if mode == "none":
        return prior_scores.copy()

    num_classes = baseline_logits.shape[1]
    topk = max(1, min(int(topk), num_classes))
    output = np.zeros_like(prior_scores, dtype=np.float32)
    topk_idx = np.argpartition(-baseline_logits, kth=topk - 1, axis=1)[:, :topk]

    for row_idx in range(prior_scores.shape[0]):
        idx = topk_idx[row_idx]
        local_scores = prior_scores[row_idx, idx]
        if mode == "topk_center":
            output[row_idx, idx] = local_scores - float(local_scores.mean())
        elif mode == "topk_margin":
            for local_i, class_idx in enumerate(idx.tolist()):
                if topk == 1:
                    output[row_idx, class_idx] = local_scores[local_i]
                else:
                    other_max = np.max(np.delete(local_scores, local_i))
                    output[row_idx, class_idx] = local_scores[local_i] - float(other_max)
        else:
            raise ValueError(f"Unsupported comparative prior mode: {mode}")

    return output


def compute_class_gains(
    baseline_per_class_f1: np.ndarray,
    improved_per_class_f1: np.ndarray,
) -> np.ndarray:
    """Compute per-class F1 gains."""
    baseline_per_class_f1 = np.asarray(baseline_per_class_f1, dtype=np.float32)
    improved_per_class_f1 = np.asarray(improved_per_class_f1, dtype=np.float32)
    if baseline_per_class_f1.shape != improved_per_class_f1.shape:
        raise ValueError(
            "baseline_per_class_f1 and improved_per_class_f1 must share the same shape."
        )
    return improved_per_class_f1 - baseline_per_class_f1


def build_classwise_gate(
    class_gains: np.ndarray,
    mode: str = "binary",
    gamma: float = 10.0,
    gain_floor: float = 0.0,
) -> np.ndarray:
    """Build a static per-class gate from validation gains."""
    class_gains = np.asarray(class_gains, dtype=np.float32)

    if mode == "binary":
        return (class_gains > float(gain_floor)).astype(np.float32)
    if mode == "continuous":
        scaled = float(gamma) * (class_gains - float(gain_floor))
        return 1.0 / (1.0 + np.exp(-scaled))

    raise ValueError(f"Unsupported class gate mode: {mode}")


def build_uncertainty_gate(
    baseline_logits: np.ndarray,
    mode: str = "soft",
    delta: float = 0.3,
    tau: float = 0.5,
) -> np.ndarray:
    """Build a sample-class gate from baseline uncertainty."""
    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-baseline_logits))
    uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)

    if mode == "none":
        return np.ones_like(uncertainty, dtype=np.float32)
    if mode == "soft":
        return uncertainty.astype(np.float32)
    if mode == "binary":
        return (uncertainty > float(delta)).astype(np.float32)
    if mode == "rank_decay":
        order = np.argsort(-baseline_logits, axis=1)
        rank = np.empty_like(order, dtype=np.int64)
        row_ids = np.arange(order.shape[0])[:, None]
        rank[row_ids, order] = np.arange(order.shape[1], dtype=np.int64)[None, :]
        decay = np.exp(-float(tau) * rank.astype(np.float32))
        return (uncertainty * decay).astype(np.float32)

    raise ValueError(f"Unsupported uncertainty gate mode: {mode}")


def apply_selective_topk_rerank(
    baseline_logits: np.ndarray,
    prior_scores: np.ndarray,
    topk: int,
    alpha: float,
    prior_mode: str = "add_norm",
    class_gate: np.ndarray | None = None,
    uncertainty_gate: np.ndarray | None = None,
    positive_only: bool = False,
) -> np.ndarray:
    """Apply selective reranking inside the baseline top-k candidate set."""
    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    prior_scores = np.asarray(prior_scores, dtype=np.float32)
    if baseline_logits.shape != prior_scores.shape:
        raise ValueError(
            f"baseline_logits shape {baseline_logits.shape} != prior_scores shape {prior_scores.shape}"
        )

    if prior_mode == "add":
        fused_prior = prior_scores.copy()
    elif prior_mode == "add_norm":
        fused_prior = normalize_scores_per_sample(prior_scores)
    else:
        raise ValueError(f"Unsupported selective prior mode: {prior_mode}")

    if positive_only:
        fused_prior = np.maximum(fused_prior, 0.0)

    delta = float(alpha) * fused_prior

    if class_gate is not None:
        class_gate = np.asarray(class_gate, dtype=np.float32)
        if class_gate.ndim != 1 or class_gate.shape[0] != baseline_logits.shape[1]:
            raise ValueError(
                f"class_gate must have shape ({baseline_logits.shape[1]},), got {class_gate.shape}"
            )
        delta = delta * class_gate[None, :]

    if uncertainty_gate is not None:
        uncertainty_gate = np.asarray(uncertainty_gate, dtype=np.float32)
        if uncertainty_gate.shape != baseline_logits.shape:
            raise ValueError(
                "uncertainty_gate must match baseline_logits shape "
                f"{baseline_logits.shape}, got {uncertainty_gate.shape}"
            )
        delta = delta * uncertainty_gate

    num_classes = baseline_logits.shape[1]
    topk = max(1, min(int(topk), num_classes))
    topk_idx = np.argpartition(-baseline_logits, kth=topk - 1, axis=1)[:, :topk]
    output = baseline_logits.copy()
    row_idx = np.arange(baseline_logits.shape[0])[:, None]
    output[row_idx, topk_idx] = baseline_logits[row_idx, topk_idx] + delta[row_idx, topk_idx]
    return output


def mix_probabilities(
    baseline_scores: np.ndarray,
    prior_scores: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Convex combination between baseline probabilities and external priors."""
    baseline_scores = np.asarray(baseline_scores, dtype=np.float32)
    prior_scores = np.asarray(prior_scores, dtype=np.float32)
    if baseline_scores.shape != prior_scores.shape:
        raise ValueError(
            f"baseline_scores shape {baseline_scores.shape} != prior_scores shape {prior_scores.shape}"
        )

    beta = float(beta)
    mixed = (1.0 - beta) * baseline_scores + beta * prior_scores
    return np.clip(mixed, 0.0, 1.0)


def evaluate_fixed_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    use_inference_strategy: bool = False,
) -> Dict[str, float | np.ndarray]:
    """Evaluate a score matrix with a fixed decision threshold."""
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
        "per_class_f1": per_class.astype(np.float32),
        "mAP": float(compute_mAP(scores, targets)),
        "threshold": float(threshold),
        "easy": difficulty["easy"],
        "medium": difficulty["medium"],
        "hard": difficulty["hard"],
    }


def evaluate_with_validation_threshold(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_scores: np.ndarray | None = None,
    test_targets: np.ndarray | None = None,
    use_inference_strategy: bool = False,
) -> Dict[str, Dict[str, float | np.ndarray]]:
    """Tune the threshold on validation, then reuse it for optional test evaluation."""
    val_metrics = eval_validation_set(
        val_scores,
        val_targets,
        use_inference_strategy=use_inference_strategy,
    )
    result: Dict[str, Dict[str, float | np.ndarray]] = {
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


def threshold_predictions(
    scores: np.ndarray,
    threshold: float,
    use_inference_strategy: bool = False,
) -> np.ndarray:
    """Convert scores into binary predictions."""
    pred = np.asarray(scores, dtype=np.float32) >= float(threshold)
    if use_inference_strategy:
        zero_rows = np.where(pred.sum(axis=1) == 0)[0]
        if len(zero_rows) > 0:
            pred[zero_rows, np.argmax(scores[zero_rows], axis=1)] = True
    return pred.astype(np.int32)


def compute_sample_f1_scores(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Compute example-level F1 for each sample."""
    targets = np.asarray(targets, dtype=np.int32)
    predictions = np.asarray(predictions, dtype=np.int32)

    tp = np.logical_and(targets == 1, predictions == 1).sum(axis=1).astype(np.float32)
    fp = np.logical_and(targets == 0, predictions == 1).sum(axis=1).astype(np.float32)
    fn = np.logical_and(targets == 1, predictions == 0).sum(axis=1).astype(np.float32)

    denom = 2.0 * tp + fp + fn
    f1 = np.zeros(targets.shape[0], dtype=np.float32)
    valid = denom > 0.0
    f1[valid] = (2.0 * tp[valid]) / denom[valid]
    return f1


def build_confusion_pairs(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    top_n: int = 10,
    focus_class_ids: Iterable[int] | None = None,
) -> List[Dict[str, int | str]]:
    """Count directed FN->FP confusion pairs in multi-label predictions."""
    focus = None if focus_class_ids is None else {int(idx) for idx in focus_class_ids}
    counter: Counter[tuple[int, int]] = Counter()

    for target_row, pred_row in zip(targets, predictions, strict=True):
        fn_ids = np.where((target_row == 1) & (pred_row == 0))[0]
        fp_ids = np.where((target_row == 0) & (pred_row == 1))[0]
        if focus is not None:
            fn_ids = np.asarray([idx for idx in fn_ids if int(idx) in focus], dtype=np.int64)
            fp_ids = np.asarray([idx for idx in fp_ids if int(idx) in focus], dtype=np.int64)
        for fn_idx in fn_ids.tolist():
            for fp_idx in fp_ids.tolist():
                counter[(int(fn_idx), int(fp_idx))] += 1

    rows: List[Dict[str, int | str]] = []
    for (fn_idx, fp_idx), count in counter.most_common(top_n):
        rows.append(
            {
                "missed_class_id": fn_idx,
                "missed_class_name": class_names[fn_idx],
                "wrong_class_id": fp_idx,
                "wrong_class_name": class_names[fp_idx],
                "count": int(count),
            }
        )
    return rows


def class_gain_rows(
    baseline_per_class_f1: np.ndarray,
    improved_per_class_f1: np.ndarray,
    class_names: Sequence[str],
    top_n: int = 10,
) -> List[Dict[str, float | int | str]]:
    """Return the classes with the largest per-class F1 gains."""
    baseline_per_class_f1 = np.asarray(baseline_per_class_f1, dtype=np.float32)
    improved_per_class_f1 = np.asarray(improved_per_class_f1, dtype=np.float32)
    gains = improved_per_class_f1 - baseline_per_class_f1
    order = np.argsort(-gains)

    rows: List[Dict[str, float | int | str]] = []
    for idx in order[:top_n]:
        idx = int(idx)
        rows.append(
            {
                "class_id": idx,
                "class_name": class_names[idx],
                "baseline_f1": float(baseline_per_class_f1[idx]),
                "improved_f1": float(improved_per_class_f1[idx]),
                "gain": float(gains[idx]),
            }
        )
    return rows
