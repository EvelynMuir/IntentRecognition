import numpy as np

from src.utils.evidence_verification import (
    build_confusion_neighborhoods,
    build_margin_aware_gate,
    build_pairwise_relation_profiles,
    compute_pairwise_comparative_scores,
)


def test_build_confusion_neighborhoods_uses_topk_negative_candidates() -> None:
    candidate_logits = np.asarray(
        [
            [5.0, 4.0, 1.0],
            [4.5, 4.2, 0.5],
            [1.0, 4.8, 4.1],
            [0.8, 4.7, 4.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    neighborhoods = build_confusion_neighborhoods(candidate_logits, labels, topk=2, top_n=1)

    assert neighborhoods == [[1], [2], []]


def test_compute_pairwise_comparative_scores_uses_pair_specific_profiles() -> None:
    relation_bundle = {
        "selected_experts": ["object"],
        "support": {
            "object": np.asarray(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        },
        "contradiction": {
            "object": np.zeros((3, 3), dtype=np.float32)
        },
    }
    pairwise_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=["object"],
        pair_profile_topn=1,
        contradiction_lambda=1.0,
    )
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [0.9, 0.4, 0.2],
                [0.2, 0.8, 0.7],
            ],
            dtype=np.float32,
        )
    }
    candidate_logits = np.asarray(
        [
            [2.0, 1.5, 0.1],
            [1.8, 1.7, 1.6],
        ],
        dtype=np.float32,
    )

    scores = compute_pairwise_comparative_scores(
        expert_phrase_scores,
        pairwise_profiles,
        candidate_logits,
        selected_experts=["object"],
        candidate_topk=2,
        activation_topm=2,
        aggregate_mode="mean",
    )

    np.testing.assert_allclose(
        scores,
        np.asarray(
            [
                [1.2, -1.2, 0.0],
                [-1.2, 1.2, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_build_margin_aware_gate_shrinks_when_margin_is_large() -> None:
    candidate_logits = np.asarray(
        [
            [2.0, 1.0, 0.1],
            [2.0, 1.9, 0.1],
        ],
        dtype=np.float32,
    )

    exp_gate = build_margin_aware_gate(candidate_logits, mode="exp", gamma=1.0)
    binary_gate = build_margin_aware_gate(candidate_logits, mode="binary", tau=0.5)

    assert exp_gate[0] < exp_gate[1]
    np.testing.assert_allclose(
        binary_gate,
        np.asarray([0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
