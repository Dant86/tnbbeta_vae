"""Tests for tnbbeta_vae.data.structured_graph."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from tnbbeta_vae.data.block_graph import BlockGraph
from tnbbeta_vae.data.structured_graph import (
    latent_sphere_graph,
    mixed_membership_graph,
    ring_graph,
    scale_free_graph,
)

_GENERATORS: dict[str, Callable[..., BlockGraph]] = {
    "sphere": latent_sphere_graph,
    "mixed": mixed_membership_graph,
    "scale_free": scale_free_graph,
    "ring": ring_graph,
}


@pytest.mark.parametrize("name", sorted(_GENERATORS))
def test_graph_is_a_simple_symmetric_sample_of_its_symmetric_truth(name: str) -> None:
    sample = _GENERATORS[name](num_nodes=120, seed=3)
    dense = sample.adjacency.toarray()

    assert np.array_equal(dense, dense.T)
    assert np.all(np.diag(dense) == 0)
    assert np.allclose(sample.probability, sample.probability.T)
    assert np.all(np.diag(sample.probability) == 0)
    assert sample.probability.min() >= 0.0 and sample.probability.max() <= 1.0
    assert sample.kind.shape == sample.probability.shape
    assert sample.block.shape == (120,)


@pytest.mark.parametrize("name", sorted(_GENERATORS))
def test_realized_density_tracks_the_true_probabilities(name: str) -> None:
    sample = _GENERATORS[name](num_nodes=300, seed=0)

    expected = sample.probability.sum() / 2
    assert abs(sample.adjacency.sum() / 2 - expected) < 0.15 * expected


def test_sampling_is_deterministic_in_the_seed() -> None:
    a = scale_free_graph(num_nodes=80, seed=5)
    b = scale_free_graph(num_nodes=80, seed=5)

    assert np.array_equal(a.adjacency.toarray(), b.adjacency.toarray())


def test_latent_sphere_probability_is_monotone_in_cosine_and_kinds_follow_it() -> None:
    sample = latent_sphere_graph(num_nodes=100, seed=0)
    off_diagonal = ~np.eye(100, dtype=bool)
    near = sample.probability[(sample.kind == "near") & off_diagonal]
    far = sample.probability[sample.kind == "far"]

    assert near.min() > far.max()
    assert {"near", "far"} <= set(sample.kind.ravel())


def test_ring_lattice_neighbours_are_denser_than_shortcuts() -> None:
    sample = ring_graph(num_nodes=100, half_width=3, p_near=0.5, p_shortcut=0.02)

    assert sample.probability[0, 3] == 0.5
    assert sample.probability[0, 97] == 0.5  # wraps around the cycle
    assert sample.probability[0, 50] == 0.02
    assert sample.kind[0, 3] == "lattice" and sample.kind[0, 50] == "shortcut"


def test_scale_free_has_hubs_far_above_the_typical_degree() -> None:
    sample = scale_free_graph(num_nodes=400, seed=1)
    degree = np.asarray(sample.adjacency.sum(axis=1)).ravel()

    assert degree.max() > 5 * np.median(degree)


def test_mixed_membership_overlap_lowers_the_within_community_probability() -> None:
    sample = mixed_membership_graph(num_nodes=300, seed=0)
    pure_same = sample.probability[sample.kind == "pure-same"].mean()
    pure_diff = sample.probability[sample.kind == "pure-diff"].mean()

    assert pure_same > 3 * pure_diff
