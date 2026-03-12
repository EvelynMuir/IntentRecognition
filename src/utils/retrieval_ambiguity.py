from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def build_retrieval_memory_indices(
    labels: np.ndarray,
    confusion_neighborhoods: Sequence[Sequence[int]] | None = None,
) -> Mapping[str, list[np.ndarray]]:
    binary_labels = (np.asarray(labels, dtype=np.float32) > 0.0)
    if binary_labels.ndim != 2:
        raise ValueError(f"Expected [num_samples, num_classes] labels, got shape {binary_labels.shape}")

    num_classes = binary_labels.shape[1]
    support_ids: list[np.ndarray] = []
    global_refute_ids: list[np.ndarray] = []
    confusion_refute_ids: list[np.ndarray] = []

    neighborhoods = (
        list(confusion_neighborhoods)
        if confusion_neighborhoods is not None
        else [[] for _ in range(num_classes)]
    )
    if len(neighborhoods) != num_classes:
        raise ValueError(
            f"confusion_neighborhoods length {len(neighborhoods)} != num_classes {num_classes}"
        )

    for class_idx in range(num_classes):
        positive_mask = binary_labels[:, class_idx]
        support_ids.append(np.flatnonzero(positive_mask).astype(np.int64))
        global_refute_ids.append(np.flatnonzero(~positive_mask).astype(np.int64))

        neighbor_ids = np.asarray(list(neighborhoods[class_idx]), dtype=np.int64)
        if neighbor_ids.size == 0:
            confusion_refute_ids.append(np.zeros((0,), dtype=np.int64))
            continue
        if np.any(neighbor_ids < 0) or np.any(neighbor_ids >= num_classes):
            raise ValueError(f"Invalid class ids in confusion_neighborhoods[{class_idx}]")

        confusion_mask = binary_labels[:, neighbor_ids].any(axis=1) & (~positive_mask)
        confusion_refute_ids.append(np.flatnonzero(confusion_mask).astype(np.int64))

    return {
        "support": support_ids,
        "global_refute": global_refute_ids,
        "confusion_refute": confusion_refute_ids,
    }


def compute_similarity_matrix(
    query_features: np.ndarray,
    memory_features: np.ndarray,
    chunk_size: int | None = None,
) -> np.ndarray:
    queries = np.asarray(query_features, dtype=np.float32)
    memory = np.asarray(memory_features, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError(f"Expected [num_queries, dim] query_features, got shape {queries.shape}")
    if memory.ndim != 2:
        raise ValueError(f"Expected [num_memory, dim] memory_features, got shape {memory.shape}")
    if queries.shape[1] != memory.shape[1]:
        raise ValueError(f"Feature dim mismatch: query dim {queries.shape[1]} vs memory dim {memory.shape[1]}")

    if chunk_size is None or int(chunk_size) >= queries.shape[0]:
        return np.asarray(queries @ memory.T, dtype=np.float32)

    step = max(1, int(chunk_size))
    outputs: list[np.ndarray] = []
    for start in range(0, queries.shape[0], step):
        end = min(queries.shape[0], start + step)
        outputs.append(np.asarray(queries[start:end] @ memory.T, dtype=np.float32))
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def compute_classwise_topk_mean_similarity(
    query_similarity: np.ndarray,
    class_memory_ids: Sequence[np.ndarray | Sequence[int]],
    k_values: Sequence[int],
) -> Mapping[int, np.ndarray]:
    similarity = np.asarray(query_similarity, dtype=np.float32)
    if similarity.ndim != 2:
        raise ValueError(f"Expected [num_queries, num_memory] similarity matrix, got shape {similarity.shape}")

    normalized_k_values = sorted({max(1, int(k)) for k in k_values})
    if not normalized_k_values:
        raise ValueError("k_values must contain at least one positive integer.")

    num_queries, num_memory = similarity.shape
    num_classes = len(class_memory_ids)
    outputs = {
        k: np.zeros((num_queries, num_classes), dtype=np.float32)
        for k in normalized_k_values
    }

    for class_idx, raw_ids in enumerate(class_memory_ids):
        memory_ids = np.asarray(raw_ids, dtype=np.int64)
        if memory_ids.ndim != 1:
            raise ValueError(f"class_memory_ids[{class_idx}] must be 1D, got shape {memory_ids.shape}")
        if memory_ids.size == 0:
            continue
        if np.any(memory_ids < 0) or np.any(memory_ids >= num_memory):
            raise ValueError(f"class_memory_ids[{class_idx}] contains out-of-range indices")

        class_similarity = similarity[:, memory_ids]
        max_k = min(normalized_k_values[-1], int(memory_ids.size))
        if max_k <= 0:
            continue

        if max_k >= class_similarity.shape[1]:
            top_values = np.sort(class_similarity, axis=1)[:, ::-1]
        else:
            part = np.partition(class_similarity, kth=class_similarity.shape[1] - max_k, axis=1)[:, -max_k:]
            top_values = np.sort(part, axis=1)[:, ::-1]

        cumulative = np.cumsum(top_values, axis=1, dtype=np.float32)
        for k in normalized_k_values:
            use_k = min(int(k), max_k)
            outputs[k][:, class_idx] = cumulative[:, use_k - 1] / float(use_k)

    return outputs


def build_retrieval_evidence_scores(
    support_scores: np.ndarray,
    refute_scores: np.ndarray | None = None,
) -> np.ndarray:
    support = np.asarray(support_scores, dtype=np.float32)
    if refute_scores is None:
        return support
    refute = np.asarray(refute_scores, dtype=np.float32)
    if support.shape != refute.shape:
        raise ValueError(f"support_scores shape {support.shape} != refute_scores shape {refute.shape}")
    return support - refute
