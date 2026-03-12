import numpy as np

from src.utils.evidence_verification import (
    compute_data_driven_verification_scores,
    learn_data_driven_relations,
)


def test_learn_data_driven_relations_pos_neg_diff_keeps_top_abs_elements() -> None:
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [3.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 5.0, 0.0],
                [1.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
    }
    labels = np.asarray(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )

    bundle = learn_data_driven_relations(
        expert_phrase_scores,
        labels,
        selected_experts=["object"],
        relation_mode="pos_neg_diff",
        profile_topn=2,
        hard_negative_topn=1,
    )

    expected = np.asarray(
        [
            [2.0, -4.0, 0.0],
            [-2.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(bundle["support"]["object"], expected, atol=1e-6)
    np.testing.assert_allclose(
        bundle["contradiction"]["object"],
        np.zeros_like(expected, dtype=np.float32),
        atol=1e-6,
    )


def test_learn_data_driven_relations_support_contradiction_splits_signs() -> None:
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [3.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 5.0, 0.0],
                [1.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
    }
    labels = np.asarray(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )

    bundle = learn_data_driven_relations(
        expert_phrase_scores,
        labels,
        selected_experts=["object"],
        relation_mode="support_contradiction",
        profile_topn=None,
        hard_negative_topn=1,
    )

    np.testing.assert_allclose(
        bundle["support"]["object"],
        np.asarray(
            [
                [2.0, 0.0, 0.5],
                [0.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        bundle["contradiction"]["object"],
        np.asarray(
            [
                [0.0, 4.0, 0.0],
                [2.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_compute_data_driven_verification_scores_uses_topm_activations_and_contradiction() -> None:
    relation_bundle = {
        "selected_experts": ["object"],
        "support": {
            "object": np.asarray(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                ],
                dtype=np.float32,
            )
        },
        "contradiction": {
            "object": np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        },
    }
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [0.9, 0.4, 0.1],
                [0.2, 0.8, 0.5],
            ],
            dtype=np.float32,
        )
    }

    verification = compute_data_driven_verification_scores(
        expert_phrase_scores,
        relation_bundle,
        activation_topm=2,
        contradiction_lambda=0.5,
        activation_positive_only=True,
    )

    np.testing.assert_allclose(
        verification,
        np.asarray(
            [
                [1.6, -0.6],
                [-0.4, 2.4],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
