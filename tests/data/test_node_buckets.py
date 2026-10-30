"""Tests for per-bucket link scoring."""

from __future__ import annotations

import numpy as np

from tnbbeta_vae.data.node_buckets import bucketed_link_scores, quantile_buckets


def test_quantile_buckets_are_balanced_even_with_ties() -> None:
    values = np.array([1.0] * 6 + [5.0] * 2 + [9.0] * 4)

    buckets = quantile_buckets(values, 3)

    assert np.bincount(buckets).tolist() == [4, 4, 4]
    assert buckets[-1] == 2 and buckets[0] == 0


def test_bucketed_scores_separate_a_perfect_bucket_from_a_random_one() -> None:
    # Nodes 0-3 (bucket 0) embed so positives score high and negatives low; nodes
    # 4-7 (bucket 1) embed so positives score LOW and negatives high.
    embeddings = np.zeros((8, 2))
    embeddings[0], embeddings[1] = [1, 0], [1, 0]
    embeddings[2], embeddings[3] = [1, 0], [0, 1]
    embeddings[4], embeddings[5] = [1, 0], [0, 1]
    embeddings[6], embeddings[7] = [1, 0], [1, 0]
    positive = np.array([[0, 4], [1, 5]])  # (0,1) scores 1; (4,5) scores 0
    negative = np.array([[2, 6], [3, 7]])  # (2,3) scores 0; (6,7) scores 1
    buckets = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    scores = bucketed_link_scores(embeddings, positive, negative, buckets)

    assert scores[0]["auc"] == 1.0
    assert scores[1]["auc"] == 0.0
    assert scores[0]["n_pos"] == 1.0 and scores[1]["n_neg"] == 1.0


def test_buckets_without_both_edge_types_are_dropped() -> None:
    embeddings = np.eye(4)
    positive = np.array([[0], [1]])
    negative = np.array([[0], [2]])
    buckets = np.array([0, 0, 1, -1])

    scores = bucketed_link_scores(embeddings, positive, negative, buckets)

    assert 1 not in scores  # bucket 1 has a negative (node 2) but no positive
    assert -1 not in scores
