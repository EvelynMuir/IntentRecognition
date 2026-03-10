import numpy as np

from src.utils.decision_rule_calibration import (
    build_head_medium_tail_groups,
    build_prior_benefit_groups,
    search_classwise_thresholds,
    search_groupwise_thresholds,
)


def test_search_classwise_thresholds_returns_best_per_class_thresholds() -> None:
    scores = np.asarray(
        [
            [0.9, 0.2],
            [0.8, 0.7],
            [0.1, 0.8],
            [0.2, 0.1],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
        ],
        dtype=np.int32,
    )
    grid = np.asarray([0.3, 0.5, 0.7], dtype=np.float32)

    thresholds = search_classwise_thresholds(scores, targets, threshold_grid=grid)

    np.testing.assert_allclose(thresholds, np.asarray([0.3, 0.3], dtype=np.float32), atol=1e-6)


def test_search_groupwise_thresholds_returns_one_threshold_per_group() -> None:
    scores = np.asarray(
        [
            [0.9, 0.8, 0.1],
            [0.7, 0.6, 0.9],
            [0.2, 0.3, 0.8],
            [0.1, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    groups = {
        "g1": [0, 1],
        "g2": [2],
    }
    grid = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)

    class_thresholds, group_thresholds = search_groupwise_thresholds(
        scores,
        targets,
        groups,
        threshold_grid=grid,
    )

    np.testing.assert_allclose(
        class_thresholds,
        np.asarray([0.5, 0.5, 0.25], dtype=np.float32),
        atol=1e-6,
    )
    assert group_thresholds == {"g1": 0.5, "g2": 0.25}


def test_build_prior_benefit_groups_splits_positive_vs_nonpositive() -> None:
    gains = np.asarray([0.1, 0.0, -0.2, 0.3], dtype=np.float32)

    groups = build_prior_benefit_groups(gains)

    assert groups == {
        "prior_benefit": [0, 3],
        "prior_neutral_or_risk": [1, 2],
    }


def test_build_head_medium_tail_groups_uses_descending_frequency() -> None:
    counts = np.asarray([100, 80, 60, 40, 20, 10], dtype=np.float32)

    groups = build_head_medium_tail_groups(counts)

    assert groups == {
        "head": [0, 1],
        "medium": [2, 3],
        "tail": [4, 5],
    }
