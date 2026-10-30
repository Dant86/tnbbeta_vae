"""Tests for the shared per-node neighbor-heterogeneity function."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from tnbbeta_vae.data.neighbor_heterogeneity import (
    labels_to_communities,
    neighbor_heterogeneity,
)


def _adjacency(edges: list[tuple[int, int]], num_nodes: int) -> sp.csr_matrix:
    """Builds a symmetric binary adjacency matrix from an undirected edge list."""
    rows = [a for a, b in edges] + [b for a, b in edges]
    cols = [b for a, b in edges] + [a for a, b in edges]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def test_all_same_community_neighborhood_has_zero_entropy() -> None:
    # Node 0's neighbors (1, 2, 3) are all community "A".
    adjacency = _adjacency([(0, 1), (0, 2), (0, 3)], num_nodes=4)
    communities = {0: {9}, 1: {0}, 2: {0}, 3: {0}}

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert heterogeneity[0] == pytest.approx(0.0, abs=1e-9)


def test_evenly_mixed_neighborhood_has_higher_entropy_than_skewed_one() -> None:
    # Node 0 has two neighbors in community 0 and two in community 1 (even split);
    # node 5 has three neighbors in community 0 and one in community 1 (skewed).
    adjacency = _adjacency(
        [(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (5, 8), (5, 9)],
        num_nodes=10,
    )
    communities = {
        1: {0},
        2: {0},
        3: {1},
        4: {1},
        6: {0},
        7: {0},
        8: {0},
        9: {1},
    }

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert heterogeneity[0] == pytest.approx(math.log(2), abs=1e-9)
    assert heterogeneity[5] < heterogeneity[0]
    assert heterogeneity[5] > 0.0


def test_isolated_node_gets_nan_not_a_crash() -> None:
    adjacency = _adjacency([(0, 1)], num_nodes=3)
    communities = {0: {0}, 1: {0}}

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert math.isnan(heterogeneity[2])


def test_node_whose_neighbors_have_no_community_labels_gets_nan() -> None:
    adjacency = _adjacency([(0, 1)], num_nodes=2)
    communities: dict[int, set[int]] = {}

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert math.isnan(heterogeneity[0])
    assert math.isnan(heterogeneity[1])


def test_neighbor_in_multiple_communities_contributes_to_each() -> None:
    # Node 0's only neighbor (1) belongs to two communities at once; node 0's
    # neighborhood distribution should then show up as split between them, exactly
    # as if it had two neighbors, one in each.
    adjacency = _adjacency([(0, 1)], num_nodes=2)
    communities = {1: {0, 1}}

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert heterogeneity[0] == pytest.approx(math.log(2), abs=1e-9)


def test_a_nodes_own_community_label_is_irrelevant_only_neighbors_count() -> None:
    adjacency = _adjacency([(0, 1), (0, 2)], num_nodes=3)
    communities = {0: {5}, 1: {0}, 2: {0}}

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert heterogeneity[0] == pytest.approx(0.0, abs=1e-9)


def test_labels_to_communities_maps_each_node_to_a_singleton_set() -> None:
    labels = np.array([0, 0, 1, 2])

    communities = labels_to_communities(labels)

    assert communities == {0: {0}, 1: {0}, 2: {1}, 3: {2}}


def test_labels_to_communities_round_trips_through_neighbor_heterogeneity() -> None:
    # Single-label adapter feeds the same function with no special case: an SBM-style
    # all-same-community neighborhood still gets zero entropy.
    labels = np.array([0, 0, 0, 1])
    communities = labels_to_communities(labels)
    adjacency = _adjacency([(0, 1), (0, 2)], num_nodes=4)

    heterogeneity = neighbor_heterogeneity(adjacency, communities)

    assert heterogeneity[0] == pytest.approx(0.0, abs=1e-9)
