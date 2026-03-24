from __future__ import annotations

from typing import Mapping

import numpy as np


def row_topk_sparsify(matrix: np.ndarray, top_k: int | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if top_k is None:
        return values.copy()
    if values.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {values.shape}")
    if values.shape[1] == 0:
        return values.copy()

    keep_k = max(0, min(int(top_k), values.shape[1]))
    output = np.zeros_like(values, dtype=np.float32)
    if keep_k == 0:
        return output
    if keep_k >= values.shape[1]:
        return values.copy()

    row_ids = np.arange(values.shape[0])[:, None]
    top_ids = np.argpartition(-values, kth=keep_k - 1, axis=1)[:, :keep_k]
    output[row_ids, top_ids] = values[row_ids, top_ids]
    return output


def l2_normalize(matrix: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(denom, float(eps))


def gather_topk_values(matrix: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {values.shape}")
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D candidate index matrix, got shape {indices.shape}")
    if values.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Sample dimension mismatch: matrix rows {values.shape[0]} vs candidate rows {indices.shape[0]}"
        )
    row_ids = np.arange(values.shape[0])[:, None]
    return values[row_ids, indices]


def scatter_candidate_values(
    candidate_values: np.ndarray,
    candidate_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    values = np.asarray(candidate_values, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"Expected [num_samples, topk] values, got shape {values.shape}")
    if indices.shape != values.shape:
        raise ValueError(f"candidate_indices shape {indices.shape} != candidate_values shape {values.shape}")

    output = np.zeros((values.shape[0], int(num_classes)), dtype=np.float32)
    row_ids = np.arange(values.shape[0])[:, None]
    output[row_ids, indices] = values
    return output


def compute_candidate_region_summaries(
    patch_tokens: np.ndarray,
    candidate_indices: np.ndarray,
    query_embeddings: np.ndarray,
    attn_logit_scale: float = 10.0,
    return_attention: bool = False,
) -> Mapping[str, np.ndarray]:
    patches = np.asarray(patch_tokens, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    queries = np.asarray(query_embeddings, dtype=np.float32)

    if patches.ndim != 3:
        raise ValueError(f"Expected [num_samples, num_patches, dim] patch tokens, got shape {patches.shape}")
    if indices.ndim != 2:
        raise ValueError(f"Expected [num_samples, topk] candidate indices, got shape {indices.shape}")
    if queries.ndim != 2:
        raise ValueError(f"Expected [num_classes, dim] query embeddings, got shape {queries.shape}")
    if patches.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Patch sample dimension {patches.shape[0]} != candidate sample dimension {indices.shape[0]}"
        )
    if patches.shape[2] != queries.shape[1]:
        raise ValueError(f"Feature dim mismatch: patch dim {patches.shape[2]} vs query dim {queries.shape[1]}")

    selected_queries = queries[indices]
    patch_norm = l2_normalize(patches, axis=-1)
    query_norm = l2_normalize(selected_queries, axis=-1)
    logits = np.einsum("bkd,bnd->bkn", query_norm, patch_norm, optimize=True) * float(attn_logit_scale)
    logits = logits - logits.max(axis=-1, keepdims=True)
    attention = np.exp(logits)
    attention = attention / np.maximum(attention.sum(axis=-1, keepdims=True), 1e-8)
    summaries = np.einsum("bkn,bnd->bkd", attention, patches, optimize=True).astype(np.float32)

    output: dict[str, np.ndarray] = {
        "summaries": summaries,
        "attention_entropy": (
            -attention * np.log(np.maximum(attention, 1e-8))
        ).sum(axis=-1).astype(np.float32),
    }
    if return_attention:
        output["attention_weights"] = attention.astype(np.float32)
    return output


def compute_candidate_phrase_scores(
    region_summaries: np.ndarray,
    bank_embeddings: np.ndarray,
    logit_scale: float = 1.0,
) -> np.ndarray:
    summaries = np.asarray(region_summaries, dtype=np.float32)
    bank = np.asarray(bank_embeddings, dtype=np.float32)
    if summaries.ndim != 3:
        raise ValueError(f"Expected [num_samples, topk, dim] summaries, got shape {summaries.shape}")
    if bank.ndim != 2:
        raise ValueError(f"Expected [num_phrases, dim] bank embeddings, got shape {bank.shape}")
    if summaries.shape[2] != bank.shape[1]:
        raise ValueError(f"Feature dim mismatch: summary dim {summaries.shape[2]} vs bank dim {bank.shape[1]}")
    if bank.shape[0] == 0:
        return np.zeros((summaries.shape[0], summaries.shape[1], 0), dtype=np.float32)

    summary_norm = l2_normalize(summaries, axis=-1)
    bank_norm = l2_normalize(bank, axis=-1)
    return np.einsum("bkd,pd->bkp", summary_norm, bank_norm, optimize=True).astype(np.float32) * float(logit_scale)


def compute_class_evidence_scores(
    phrase_scores: np.ndarray,
    support_matrix: np.ndarray,
    contradiction_matrix: np.ndarray | None = None,
    topm: int | None = 5,
    positive_only: bool = True,
) -> np.ndarray:
    scores = np.asarray(phrase_scores, dtype=np.float32)
    support = np.asarray(support_matrix, dtype=np.float32)
    contradiction = (
        np.zeros_like(support, dtype=np.float32)
        if contradiction_matrix is None
        else np.asarray(contradiction_matrix, dtype=np.float32)
    )
    if scores.ndim != 2:
        raise ValueError(f"Expected [num_samples, num_phrases] phrase scores, got shape {scores.shape}")
    if support.ndim != 2:
        raise ValueError(f"Expected [num_classes, num_phrases] support matrix, got shape {support.shape}")
    if scores.shape[1] != support.shape[1]:
        raise ValueError(f"Phrase dim mismatch: scores {scores.shape[1]} vs support {support.shape[1]}")

    active = np.maximum(scores, 0.0) if positive_only else scores
    active = row_topk_sparsify(active, top_k=topm)
    relation = support - contradiction
    return active @ relation.T


def compute_candidate_class_evidence_scores(
    candidate_phrase_scores: np.ndarray,
    candidate_indices: np.ndarray,
    support_matrix: np.ndarray,
    contradiction_matrix: np.ndarray | None = None,
    topm: int | None = 5,
    positive_only: bool = True,
    num_classes: int | None = None,
) -> np.ndarray:
    scores = np.asarray(candidate_phrase_scores, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    support = np.asarray(support_matrix, dtype=np.float32)
    contradiction = (
        np.zeros_like(support, dtype=np.float32)
        if contradiction_matrix is None
        else np.asarray(contradiction_matrix, dtype=np.float32)
    )
    if scores.ndim != 3:
        raise ValueError(
            f"Expected [num_samples, topk, num_phrases] candidate phrase scores, got shape {scores.shape}"
        )
    if indices.ndim != 2:
        raise ValueError(f"Expected [num_samples, topk] candidate indices, got shape {indices.shape}")
    if scores.shape[:2] != indices.shape:
        raise ValueError(f"candidate_indices shape {indices.shape} != candidate_phrase_scores prefix {scores.shape[:2]}")
    if scores.shape[2] != support.shape[1]:
        raise ValueError(f"Phrase dim mismatch: scores {scores.shape[2]} vs support {support.shape[1]}")

    active = np.maximum(scores, 0.0) if positive_only else scores
    flat_active = row_topk_sparsify(active.reshape(-1, active.shape[-1]), top_k=topm).reshape(active.shape)
    relation = support - contradiction
    relation_rows = relation[indices]
    candidate_values = np.sum(flat_active * relation_rows, axis=-1, dtype=np.float32)
    class_count = int(num_classes) if num_classes is not None else int(support.shape[0])
    return scatter_candidate_values(candidate_values, indices, num_classes=class_count)


def normalize_topk_candidate_matrix(
    values: np.ndarray,
    candidate_indices: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if scores.ndim != 2:
        raise ValueError(f"Expected [num_samples, num_classes] values, got shape {scores.shape}")
    if indices.ndim != 2:
        raise ValueError(f"Expected [num_samples, topk] candidate indices, got shape {indices.shape}")
    if scores.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Sample dimension mismatch: scores rows {scores.shape[0]} vs candidate rows {indices.shape[0]}"
        )

    gathered = gather_topk_values(scores, indices)
    mean = gathered.mean(axis=1, keepdims=True)
    std = gathered.std(axis=1, keepdims=True)
    normalized = (gathered - mean) / np.maximum(std, float(eps))
    return scatter_candidate_values(normalized, indices, num_classes=scores.shape[1])


def build_expert_stack(
    expert_matrices: Mapping[str, np.ndarray],
    candidate_indices: np.ndarray,
    expert_order: list[str],
) -> np.ndarray:
    stacks = []
    for expert in expert_order:
        stacks.append(gather_topk_values(np.asarray(expert_matrices[expert], dtype=np.float32), candidate_indices))
    return np.stack(stacks, axis=-1).astype(np.float32)


def compute_soft_routing_weights(
    expert_stack: np.ndarray,
    candidate_logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    values = np.asarray(expert_stack, dtype=np.float32)
    logits = np.asarray(candidate_logits, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [num_samples, topk, num_experts] expert stack, got shape {values.shape}")
    if logits.ndim != 2:
        raise ValueError(f"Expected [num_samples, topk] candidate logits, got shape {logits.shape}")
    if values.shape[:2] != logits.shape:
        raise ValueError(f"candidate_logits shape {logits.shape} != expert stack prefix {values.shape[:2]}")

    belief_logits = logits - logits.max(axis=1, keepdims=True)
    belief_probs = np.exp(belief_logits)
    belief_probs = belief_probs / np.maximum(belief_probs.sum(axis=1, keepdims=True), 1e-8)
    candidate_uncertainty = 1.0 - belief_probs
    route_logits = values * candidate_uncertainty[:, :, None] / max(float(temperature), 1e-6)
    route_logits = route_logits - route_logits.max(axis=-1, keepdims=True)
    weights = np.exp(route_logits)
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)


def apply_soft_routing(expert_stack: np.ndarray, routing_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(expert_stack, dtype=np.float32)
    weights = np.asarray(routing_weights, dtype=np.float32)
    if values.shape != weights.shape:
        raise ValueError(f"expert_stack shape {values.shape} != routing_weights shape {weights.shape}")
    return np.sum(values * weights, axis=-1, dtype=np.float32)
