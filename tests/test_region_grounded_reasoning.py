import numpy as np

from src.utils.region_grounded_reasoning import (
    apply_soft_routing,
    compute_candidate_class_evidence_scores,
    compute_candidate_region_summaries,
    compute_soft_routing_weights,
    normalize_topk_candidate_matrix,
)


def test_compute_candidate_region_summaries_focuses_on_best_aligned_patch() -> None:
    patch_tokens = np.asarray(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    candidate_indices = np.asarray([[0, 1]], dtype=np.int64)
    query_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    outputs = compute_candidate_region_summaries(
        patch_tokens,
        candidate_indices,
        query_embeddings,
        attn_logit_scale=12.0,
        return_attention=True,
    )
    summaries = outputs["summaries"]
    attention = outputs["attention_weights"]

    np.testing.assert_allclose(summaries[0, 0], np.asarray([1.0, 0.0], dtype=np.float32), atol=1e-4)
    np.testing.assert_allclose(summaries[0, 1], np.asarray([0.0, 1.0], dtype=np.float32), atol=1e-4)
    assert attention[0, 0, 0] > attention[0, 0, 1]
    assert attention[0, 1, 1] > attention[0, 1, 0]


def test_compute_candidate_class_evidence_scores_scatter_to_topk_candidates() -> None:
    candidate_phrase_scores = np.asarray(
        [
            [
                [2.0, 0.5, 0.1],
                [0.2, 1.0, 0.3],
            ]
        ],
        dtype=np.float32,
    )
    candidate_indices = np.asarray([[2, 0]], dtype=np.int64)
    support = np.asarray(
        [
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    evidence = compute_candidate_class_evidence_scores(
        candidate_phrase_scores,
        candidate_indices,
        support,
        topm=1,
        positive_only=True,
        num_classes=3,
    )

    np.testing.assert_allclose(
        evidence,
        np.asarray([[1.5, 0.0, 4.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_normalize_topk_candidate_matrix_only_normalizes_selected_candidates() -> None:
    values = np.asarray([[5.0, 0.0, 1.0, 3.0]], dtype=np.float32)
    candidate_indices = np.asarray([[0, 3, 2]], dtype=np.int64)

    normalized = normalize_topk_candidate_matrix(values, candidate_indices)

    expected_topk = np.asarray([1.2247449, 0.0, -1.2247449], dtype=np.float32)
    np.testing.assert_allclose(normalized[0, [0, 3, 2]], expected_topk, atol=1e-5)
    assert float(normalized[0, 1]) == 0.0


def test_soft_routing_prefers_stronger_expert_when_candidate_is_uncertain() -> None:
    expert_stack = np.asarray(
        [
            [
                [1.2, 0.1],
                [1.2, 0.1],
            ]
        ],
        dtype=np.float32,
    )
    candidate_logits = np.asarray([[2.0, -2.0]], dtype=np.float32)

    weights = compute_soft_routing_weights(expert_stack, candidate_logits, temperature=0.5)
    routed = apply_soft_routing(expert_stack, weights)

    assert weights[0, 1, 0] > weights[0, 0, 0]
    assert weights[0, 1, 0] > 0.5
    assert routed[0, 1] > routed[0, 0]
