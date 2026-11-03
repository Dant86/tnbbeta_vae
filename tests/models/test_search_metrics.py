"""Tests for tnbbeta_vae.models.search_metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tnbbeta_vae.data.graph_space import GraphSpec
from tnbbeta_vae.models.search_metrics import (
    block_labels,
    graph_statistic_distances,
    hard_negative_ranking,
    kmeans_labels,
    latent_recovery,
    onmi_lfk,
)


def _cover(labels: np.ndarray, k: int) -> list[np.ndarray]:
    return [labels == j for j in range(k)]


def test_onmi_is_one_for_identical_covers_and_low_for_independent_ones() -> None:
    labels = np.repeat(np.arange(4), 25)
    same = onmi_lfk(_cover(labels, 4), _cover(labels, 4))
    shuffled = np.random.default_rng(0).permutation(labels)
    other = onmi_lfk(_cover(labels, 4), _cover(shuffled, 4))

    assert same == pytest.approx(1.0)
    assert other < 0.2
    # relabelling the communities does not change the score
    relabelled = (labels + 1) % 4
    assert onmi_lfk(_cover(labels, 4), _cover(relabelled, 4)) == pytest.approx(1.0)


def test_onmi_handles_overlapping_covers() -> None:
    n = 60
    first = [np.arange(n) < 40, np.arange(n) >= 20]
    value = onmi_lfk(first, first)

    assert value == pytest.approx(1.0)
    assert 0.0 <= onmi_lfk(first, [np.arange(n) < 30, np.arange(n) >= 30]) <= 1.0


def test_kmeans_recovers_separated_clusters() -> None:
    rng = np.random.default_rng(1)
    centres = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    labels = np.repeat(np.arange(3), 30)
    points = centres[labels] + 0.1 * rng.standard_normal((90, 3))

    found = kmeans_labels(points, 3, np.random.default_rng(2))

    assert onmi_lfk(_cover(found, 3), _cover(labels, 3)) > 0.99


def test_hard_negative_ranking_is_perfect_for_the_oracle_and_chance_for_constants() -> (
    None
):
    n = 40
    rng = np.random.default_rng(0)
    full = np.zeros((n, n), dtype=bool)
    positives = np.array([[0, 1, 2, 3], [10, 11, 12, 13]])
    for a, b in positives.T:
        full[a, b] = full[b, a] = True
    common = rng.integers(0, 3, (n, n)).astype(np.float32)
    ppr = rng.random((n, n))
    oracle = full.astype(np.float64)
    constant = np.zeros((n, n))

    best = hard_negative_ranking(oracle, positives, full, common, ppr, rng)
    flat = hard_negative_ranking(constant, positives, full, common, ppr, rng)

    assert best["hits10"] == 1.0 and best["mrr"] == pytest.approx(1.0)
    # tied with every one of its 38 negatives: average rank 1 + 19
    assert flat["mrr"] == pytest.approx(1.0 / 20.0)
    assert flat["hits10"] == 0.0 and flat["hits50"] == 1.0


def test_hard_negatives_exclude_every_neighbour_in_the_full_graph() -> None:
    n = 30
    rng = np.random.default_rng(3)
    full = np.zeros((n, n), dtype=bool)
    full[0, 1:20] = full[1:20, 0] = True  # node 0 is linked to 1..19
    positives = np.array([[0], [1]])
    score = np.zeros((n, n))
    score[0, 1:20] = 1.0  # neighbours score high: they must not be negatives
    result = hard_negative_ranking(
        score, positives, full, np.ones((n, n)), np.ones((n, n)), rng, num_negatives=100
    )

    assert result["hits10"] == 1.0


def test_graph_statistics_vanish_for_identical_probabilities() -> None:
    p = torch.full((60, 60), 0.2)
    p.fill_diagonal_(0.0)
    generator = torch.Generator().manual_seed(0)

    same = graph_statistic_distances(p, p, samples=20, generator=generator)
    denser = graph_statistic_distances(
        torch.clamp(p * 2, max=1.0), p, samples=20, generator=generator
    )

    assert same["gs_degree_w1"] < 0.05
    assert same["gs_clustering_err"] < 0.05
    assert denser["gs_degree_w1"] > 0.5
    assert denser["gs_triangle_err"] > same["gs_triangle_err"]


def test_block_labels_exist_only_for_block_families() -> None:
    spec = GraphSpec("assortative", 100, 8.0, 5, "none", "none")

    labels, count = block_labels(spec) or (np.zeros(0), 0)
    assert count == 5 and labels.max() == 4
    assert block_labels(GraphSpec("ring", 100, 8.0, 5, "none", "none")) is None


def test_latent_recovery_scores_the_true_latent_highly_and_noise_low() -> None:
    n, k, seed = 120, 4, 11
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    spec = GraphSpec("latent_kernel", n, 8.0, k, "none", "none")

    exact = latent_recovery(spec, n, z, seed)
    noise = latent_recovery(
        spec, n, np.random.default_rng(5).standard_normal((n, 8)), seed
    )

    assert exact["latent_gram_rho"] > 0.99
    assert exact["latent_procrustes"] < 1e-8
    assert noise["latent_gram_rho"] < 0.2
    assert math.isnan(exact["latent_nmi"])


def test_latent_recovery_nan_for_undefined_families() -> None:
    spec = GraphSpec("scale_free", 50, 8.0, 3, "none", "none")
    out = latent_recovery(
        spec, 50, np.random.default_rng(0).standard_normal((50, 4)), 1
    )

    assert all(math.isnan(v) for v in out.values())


def test_latent_nmi_for_blocks_and_mixed_membership() -> None:
    n = 100
    spec = GraphSpec("assortative", n, 8.0, 4, "none", "none")
    labels, _ = block_labels(spec) or (np.zeros(n, dtype=int), 0)
    centres = np.eye(4)[labels] + 0.01 * np.random.default_rng(0).standard_normal(
        (n, 4)
    )
    mixed = GraphSpec("mixed_membership", n, 8.0, 4, "none", "none")

    assert latent_recovery(spec, n, centres, 3)["latent_nmi"] > 0.95
    assert 0.0 <= latent_recovery(mixed, n, centres, 3)["latent_nmi"] <= 1.0
