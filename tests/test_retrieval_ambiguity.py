import numpy as np

from src.utils.retrieval_ambiguity import (
    build_retrieval_evidence_scores,
    build_retrieval_memory_indices,
    compute_classwise_topk_mean_similarity,
    compute_similarity_matrix,
)


def test_build_retrieval_memory_indices_constructs_support_and_refute_pools() -> None:
    labels = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    memory = build_retrieval_memory_indices(labels, confusion_neighborhoods=[[1], [0, 2], [1]])

    np.testing.assert_array_equal(memory["support"][0], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(memory["support"][1], np.asarray([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(memory["support"][2], np.asarray([3], dtype=np.int64))

    np.testing.assert_array_equal(memory["global_refute"][0], np.asarray([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(memory["global_refute"][1], np.asarray([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(memory["global_refute"][2], np.asarray([0, 1, 2], dtype=np.int64))

    np.testing.assert_array_equal(memory["confusion_refute"][0], np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(memory["confusion_refute"][1], np.asarray([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(memory["confusion_refute"][2], np.asarray([1, 2], dtype=np.int64))


def test_compute_similarity_matrix_matches_chunked_dot_products() -> None:
    query_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    memory_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    similarity = compute_similarity_matrix(query_features, memory_features, chunk_size=1)

    np.testing.assert_allclose(
        similarity,
        np.asarray(
            [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_compute_classwise_topk_mean_similarity_reuses_available_neighbors() -> None:
    similarity = np.asarray(
        [
            [0.1, 0.8, 0.4, 0.6],
            [0.9, 0.2, 0.7, 0.3],
        ],
        dtype=np.float32,
    )
    class_memory_ids = [
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        np.asarray([], dtype=np.int64),
    ]

    means = compute_classwise_topk_mean_similarity(similarity, class_memory_ids, k_values=[1, 2, 4])

    np.testing.assert_allclose(
        means[1],
        np.asarray(
            [
                [0.8, 0.8, 0.0],
                [0.9, 0.3, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        means[2],
        np.asarray(
            [
                [0.6, 0.7, 0.0],
                [0.8, 0.25, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        means[4],
        np.asarray(
            [
                [(0.8 + 0.4 + 0.1) / 3.0, 0.7, 0.0],
                [(0.9 + 0.7 + 0.2) / 3.0, 0.25, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )

    evidence = build_retrieval_evidence_scores(means[2], refute_scores=np.asarray([[0.2, 0.1, 0.0], [0.4, 0.2, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        evidence,
        np.asarray(
            [
                [0.4, 0.6, 0.0],
                [0.4, 0.05, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
