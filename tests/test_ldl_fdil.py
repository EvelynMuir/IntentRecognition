import numpy as np

from scripts.run_ldl_fdil import apply_local_reranking, distribution_metrics, normalized_entropy, uncertainty_gate


def test_distribution_metrics_identity() -> None:
    targets = np.asarray(
        [[0.02, 0.08, 0.04, 0.16, 0.10, 0.32, 0.20, 0.08],
         [0.25, 0.05, 0.10, 0.20, 0.05, 0.15, 0.08, 0.12]], dtype=np.float32
    )
    metrics = distribution_metrics(targets, targets)
    assert metrics["chebyshev"] == 0.0
    assert metrics["clark"] == 0.0
    assert abs(metrics["kl"]) < 1e-12
    assert abs(metrics["cosine"] - 1.0) < 1e-12
    assert abs(metrics["spearman"] - 1.0) < 1e-12
    assert abs(metrics["mu"] - 100.0) < 1e-9
    assert abs(metrics["dvse"]) < 1e-12


def test_entropy_gate_extremes() -> None:
    values = normalized_entropy(np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))
    assert np.allclose(values, [0.0, 1.0])
    assert np.allclose(uncertainty_gate(np.asarray([[1.0, 0.0], [0.5, 0.5]]), "max_mass"), [0.0, 0.5])


def test_local_reranking_only_changes_topk() -> None:
    base = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    prior = np.asarray([[1.0, -1.0, 10.0]], dtype=np.float32)
    output = apply_local_reranking(base, prior, topk=2, alpha=0.3)
    assert np.allclose(output, [[3.3, 1.7, 1.0]])
