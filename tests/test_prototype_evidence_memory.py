import numpy as np

from src.utils.evidence_verification import (
    build_pairwise_relation_profiles,
    compute_pairwise_comparative_scores,
    learn_prototype_memory_relations,
    select_prototype_profile_ids,
)


def test_select_prototype_profile_ids_picks_closest_center() -> None:
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )
    }
    prototype_bundle = {
        "selected_experts": ["object"],
        "prototype_source": "full",
        "prototype_source_topm": None,
        "prototype_row_ids": np.asarray([[0, 1]], dtype=np.int64),
        "prototype_counts": np.asarray([2], dtype=np.int64),
        "prototype_centers": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    selected = select_prototype_profile_ids(
        expert_phrase_scores,
        prototype_bundle,
        selected_experts=["object"],
        positive_only_scores=True,
    )

    np.testing.assert_array_equal(selected[:, 0], np.asarray([0, 1, 0], dtype=np.int64))


def test_learn_prototype_memory_relations_falls_back_when_class_is_too_small() -> None:
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [1.0, 0.1],
                [0.9, 0.2],
                [0.8, 0.3],
                [0.7, 0.4],
            ],
            dtype=np.float32,
        )
    }
    labels = np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32)

    bundle = learn_prototype_memory_relations(
        expert_phrase_scores,
        labels,
        selected_experts=["object"],
        relation_mode="hard_negative_diff",
        profile_topn=None,
        hard_negative_topn=1,
        hard_negative_ids=[[]],
        positive_only_scores=True,
        prototype_k=2,
        prototype_source="full",
        prototype_source_topm=None,
        min_positive_samples=8,
        min_cluster_size=2,
        random_state=0,
    )

    np.testing.assert_array_equal(bundle["prototype_counts"], np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(bundle["fallback_mask"], np.asarray([True]))
    assert bundle["fallback_reasons"] == ["insufficient_positive_samples"]


def test_compute_pairwise_comparative_scores_uses_selected_prototype_rows() -> None:
    relation_bundle = {
        "selected_experts": ["object"],
        "support": {
            "object": np.asarray(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            )
        },
        "contradiction": {
            "object": np.zeros((4, 3), dtype=np.float32)
        },
    }
    pairwise_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=["object"],
        pair_profile_topn=None,
        contradiction_lambda=0.0,
    )
    expert_phrase_scores = {
        "object": np.asarray(
            [
                [1.0, 0.5, 0.2],
                [0.1, 1.0, 0.7],
            ],
            dtype=np.float32,
        )
    }
    candidate_logits = np.asarray(
        [
            [2.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidate_profile_ids = np.asarray(
        [
            [0, 2],
            [1, 3],
        ],
        dtype=np.int64,
    )

    scores = compute_pairwise_comparative_scores(
        expert_phrase_scores,
        pairwise_profiles,
        candidate_logits,
        selected_experts=["object"],
        candidate_profile_ids=candidate_profile_ids,
        candidate_topk=2,
        activation_topm=None,
        aggregate_mode="mean",
    )

    np.testing.assert_allclose(
        scores,
        np.asarray(
            [
                [1.8, -1.8],
                [0.6, -0.6],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
