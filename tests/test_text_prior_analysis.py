import numpy as np

from src.utils.text_prior_analysis import (
    aggregate_prompt_scores,
    apply_topk_rerank_fusion,
    apply_selective_topk_rerank,
    build_topk_comparative_prior,
    build_classwise_gate,
    build_confusion_pairs,
    build_uncertainty_gate,
    compute_class_gains,
    compute_sample_f1_scores,
    mix_probabilities,
    normalize_scores_per_sample,
)


def test_normalize_scores_per_sample_rowwise_zero_mean() -> None:
    scores = np.asarray([[1.0, 2.0, 3.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    normalized = normalize_scores_per_sample(scores)

    np.testing.assert_allclose(normalized[0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[0].std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1], np.zeros(3, dtype=np.float32), atol=1e-6)


def test_aggregate_prompt_scores_supports_multiple_modes() -> None:
    scores = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)

    np.testing.assert_allclose(aggregate_prompt_scores(scores, mode="average"), np.asarray([2.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(aggregate_prompt_scores(scores, mode="max"), np.asarray([3.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(aggregate_prompt_scores(scores, mode="top2_avg"), np.asarray([2.5], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        aggregate_prompt_scores(scores, mode="logsumexp"),
        np.asarray([2.3089938], dtype=np.float32),
        atol=1e-6,
    )


def test_apply_topk_rerank_fusion_only_updates_topk_candidates() -> None:
    baseline = np.asarray([[0.9, 0.2, -0.1, 0.4]], dtype=np.float32)
    prior = np.asarray([[10.0, 9.0, 8.0, 7.0]], dtype=np.float32)

    fused = apply_topk_rerank_fusion(baseline, prior, topk=2, alpha=0.5, mode="add")

    expected = np.asarray([[5.9, 0.2, -0.1, 3.9]], dtype=np.float32)
    np.testing.assert_allclose(fused, expected, atol=1e-6)


def test_build_topk_comparative_prior_supports_center_and_margin() -> None:
    baseline = np.asarray([[0.9, 0.4, 0.1]], dtype=np.float32)
    prior = np.asarray([[2.0, 1.0, -3.0]], dtype=np.float32)

    centered = build_topk_comparative_prior(baseline, prior, topk=2, mode="topk_center")
    margin = build_topk_comparative_prior(baseline, prior, topk=2, mode="topk_margin")

    np.testing.assert_allclose(centered, np.asarray([[0.5, -0.5, 0.0]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(margin, np.asarray([[1.0, -1.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_mix_probabilities_is_convex_combination() -> None:
    baseline = np.asarray([[0.8, 0.2]], dtype=np.float32)
    prior = np.asarray([[0.4, 0.6]], dtype=np.float32)

    mixed = mix_probabilities(baseline, prior, beta=0.25)

    np.testing.assert_allclose(mixed, np.asarray([[0.7, 0.3]], dtype=np.float32), atol=1e-6)


def test_compute_sample_f1_scores_handles_partial_match() -> None:
    targets = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    predictions = np.asarray([[1, 1, 0], [0, 1, 0]], dtype=np.int32)

    scores = compute_sample_f1_scores(targets, predictions)

    np.testing.assert_allclose(scores, np.asarray([0.5, 1.0], dtype=np.float32), atol=1e-6)


def test_build_confusion_pairs_counts_fn_to_fp_pairs() -> None:
    class_names = ["a", "b", "c"]
    targets = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
    predictions = np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.int32)

    rows = build_confusion_pairs(targets, predictions, class_names, top_n=5)

    assert rows == [
        {
            "missed_class_id": 0,
            "missed_class_name": "a",
            "wrong_class_id": 1,
            "wrong_class_name": "b",
            "count": 1,
        },
        {
            "missed_class_id": 1,
            "missed_class_name": "b",
            "wrong_class_id": 2,
            "wrong_class_name": "c",
            "count": 1,
        },
    ]


def test_compute_class_gains_and_binary_gate() -> None:
    baseline = np.asarray([0.2, 0.4, 0.8], dtype=np.float32)
    improved = np.asarray([0.3, 0.1, 0.8], dtype=np.float32)

    gains = compute_class_gains(baseline, improved)
    gate = build_classwise_gate(gains, mode="binary")

    np.testing.assert_allclose(gains, np.asarray([0.1, -0.3, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(gate, np.asarray([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_build_uncertainty_gate_binary_and_rank_decay() -> None:
    logits = np.asarray([[0.0, 4.0, -4.0]], dtype=np.float32)

    binary_gate = build_uncertainty_gate(logits, mode="binary", delta=0.5)
    rank_decay_gate = build_uncertainty_gate(logits, mode="rank_decay", tau=1.0)

    np.testing.assert_allclose(binary_gate, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), atol=1e-6)
    assert rank_decay_gate[0, 0] > rank_decay_gate[0, 1]
    assert rank_decay_gate[0, 1] > rank_decay_gate[0, 2]


def test_apply_selective_topk_rerank_respects_gates_and_positive_only() -> None:
    baseline = np.asarray([[0.9, 0.4, 0.1]], dtype=np.float32)
    prior = np.asarray([[2.0, -3.0, 10.0]], dtype=np.float32)
    class_gate = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    uncertainty_gate = np.asarray([[1.0, 0.5, 1.0]], dtype=np.float32)

    fused = apply_selective_topk_rerank(
        baseline,
        prior,
        topk=2,
        alpha=1.0,
        prior_mode="add",
        class_gate=class_gate,
        uncertainty_gate=uncertainty_gate,
        positive_only=True,
    )

    expected = np.asarray([[2.9, 0.4, 0.1]], dtype=np.float32)
    np.testing.assert_allclose(fused, expected, atol=1e-6)
