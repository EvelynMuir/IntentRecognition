import numpy as np

from src.utils.evidence_verification import (
    build_confusion_aware_router,
    build_pairwise_relation_profiles,
    compute_specialist_pairwise_evidence,
    resolve_routed_specialist_evidence,
)


def test_build_confusion_aware_router_triggers_on_margin_or_confusion_hit() -> None:
    candidate_logits = np.asarray(
        [
            [2.0, 0.1, 1.9],
            [2.5, 2.0, 0.2],
        ],
        dtype=np.float32,
    )
    confusion_neighborhoods = [
        [1],
        [0],
        [],
    ]

    router = build_confusion_aware_router(
        candidate_logits,
        confusion_neighborhoods=confusion_neighborhoods,
        topk=2,
        margin_tau=0.2,
        trigger_mode="margin_confusion",
        dispatch_mode="all",
    )

    np.testing.assert_array_equal(router["margin_trigger_mask"], np.asarray([True, False]))
    np.testing.assert_array_equal(router["confusion_trigger_mask"], np.asarray([False, True]))
    np.testing.assert_array_equal(router["trigger_mask"], np.asarray([True, True]))
    assert router["selected_neighborhoods"] == [[0, 2], [0, 1]]
    assert router["selected_pairs"] == [[(0, 2)], [(0, 1)]]
    assert router["selected_experts"][0] == ["object", "scene", "style", "activity"]


def test_build_confusion_aware_router_routes_to_strongest_specialist() -> None:
    relation_bundle = {
        "selected_experts": ["object", "scene"],
        "support": {
            "object": np.asarray(
                [
                    [3.0, 0.0],
                    [0.0, 2.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "scene": np.asarray(
                [
                    [0.2, 0.0],
                    [0.0, 0.1],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        },
        "contradiction": {
            "object": np.zeros((3, 2), dtype=np.float32),
            "scene": np.zeros((3, 2), dtype=np.float32),
        },
    }
    pairwise_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=["object", "scene"],
        pair_profile_topn=None,
        contradiction_lambda=1.0,
    )
    candidate_logits = np.asarray([[2.5, 2.0, 0.1]], dtype=np.float32)

    router = build_confusion_aware_router(
        candidate_logits,
        confusion_neighborhoods=[[1], [0], []],
        pairwise_profiles=pairwise_profiles,
        selected_experts=["object", "scene"],
        topk=2,
        margin_tau=0.1,
        trigger_mode="margin_confusion",
        dispatch_mode="routed",
        max_routed_experts=1,
    )

    assert router["selected_neighborhoods"] == [[0, 1]]
    assert router["selected_pairs"] == [[(0, 1)]]
    assert router["selected_experts"] == [["object"]]
    assert router["expert_weights"] == [{"object": 1.0}]


def test_resolve_routed_specialist_evidence_only_updates_selected_neighborhood() -> None:
    relation_bundle = {
        "selected_experts": ["object", "scene"],
        "support": {
            "object": np.asarray(
                [
                    [2.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "scene": np.asarray(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        },
        "contradiction": {
            "object": np.zeros((3, 2), dtype=np.float32),
            "scene": np.zeros((3, 2), dtype=np.float32),
        },
    }
    pairwise_profiles = build_pairwise_relation_profiles(
        relation_bundle,
        selected_experts=["object", "scene"],
        pair_profile_topn=None,
        contradiction_lambda=1.0,
    )
    expert_phrase_scores = {
        "object": np.asarray([[1.0, 0.2]], dtype=np.float32),
        "scene": np.asarray([[0.3, 0.9]], dtype=np.float32),
    }
    router_outputs = {
        "trigger_mask": np.asarray([True]),
        "selected_pairs": [[(0, 1)]],
        "selected_experts": [["object"]],
        "expert_weights": [{"object": 1.0}],
    }

    specialist_evidence = compute_specialist_pairwise_evidence(
        expert_phrase_scores,
        pairwise_profiles,
        router_outputs,
        selected_experts=["object", "scene"],
        activation_topm=2,
    )
    resolved = resolve_routed_specialist_evidence(
        specialist_evidence,
        router_outputs,
        selected_experts=["object", "scene"],
        aggregate_mode="mean",
    )

    np.testing.assert_allclose(
        specialist_evidence["expert_scores"]["object"],
        np.asarray([[1.8, -1.8, 0.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        resolved["resolved_scores"],
        np.asarray([[1.8, -1.8, 0.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        resolved["expert_scores"]["scene"],
        np.zeros((1, 3), dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        resolved["pair_counts"],
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )
