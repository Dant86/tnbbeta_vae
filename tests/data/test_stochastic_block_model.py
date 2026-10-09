"""Tests for the synthetic stochastic-block-model graph generator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tnbbeta_vae.data.planetoid import normalized_adjacency, split_edges
from tnbbeta_vae.data.snap_community import Graph
from tnbbeta_vae.data.stochastic_block_model import (
    sbm_community_labels,
    stochastic_block_model,
)


def test_returns_a_snap_community_graph_with_expected_shape() -> None:
    graph = stochastic_block_model(
        num_communities=4, nodes_per_community=10, p_in=0.3, p_out=0.01, seed=0
    )

    assert isinstance(graph, Graph)
    assert graph.adjacency.shape == (40, 40)
    assert graph.features.shape == (40, 40)


def test_node_id_map_is_the_trivial_identity_map() -> None:
    graph = stochastic_block_model(
        num_communities=3, nodes_per_community=5, p_in=0.3, p_out=0.01, seed=0
    )

    assert graph.node_id_map == {i: i for i in range(15)}


def test_features_are_a_sparse_identity_matrix() -> None:
    graph = stochastic_block_model(
        num_communities=3, nodes_per_community=5, p_in=0.3, p_out=0.01, seed=0
    )

    assert isinstance(graph.features, torch.Tensor)
    assert graph.features.is_sparse
    dense = graph.features.to_dense().numpy()
    np.testing.assert_array_equal(dense, np.eye(15, dtype=np.float32))


def test_adjacency_is_symmetric_binary_with_no_self_loops() -> None:
    graph = stochastic_block_model(
        num_communities=4, nodes_per_community=10, p_in=0.3, p_out=0.01, seed=0
    )

    assert (graph.adjacency != graph.adjacency.T).nnz == 0
    assert graph.adjacency.diagonal().sum() == 0
    values = np.unique(graph.adjacency.data)
    assert set(values.tolist()) <= {1.0}


def test_reproducible_given_the_same_seed() -> None:
    first = stochastic_block_model(
        num_communities=5, nodes_per_community=20, p_in=0.3, p_out=0.01, seed=42
    )
    second = stochastic_block_model(
        num_communities=5, nodes_per_community=20, p_in=0.3, p_out=0.01, seed=42
    )

    assert (first.adjacency != second.adjacency).nnz == 0


def test_different_seeds_give_different_graphs() -> None:
    first = stochastic_block_model(
        num_communities=5, nodes_per_community=20, p_in=0.3, p_out=0.01, seed=0
    )
    second = stochastic_block_model(
        num_communities=5, nodes_per_community=20, p_in=0.3, p_out=0.01, seed=1
    )

    assert (first.adjacency != second.adjacency).nnz > 0


def test_within_and_across_community_edge_density_match_requested_probabilities() -> (
    None
):
    # Large enough and far enough apart (p_in vs p_out) that the empirical edge
    # density lands within a generous tolerance of the requested probability, not
    # just "trust the parameters" -- a real measurement against the sampled
    # adjacency.
    num_communities, nodes_per_community = 10, 50
    p_in, p_out = 0.3, 0.01
    graph = stochastic_block_model(
        num_communities=num_communities,
        nodes_per_community=nodes_per_community,
        p_in=p_in,
        p_out=p_out,
        seed=0,
    )
    labels = sbm_community_labels(num_communities, nodes_per_community)
    adjacency = graph.adjacency.tocoo()
    same_community = labels[adjacency.row] == labels[adjacency.col]

    num_nodes = num_communities * nodes_per_community
    within_pairs = num_communities * nodes_per_community * (nodes_per_community - 1)
    total_pairs = num_nodes * (num_nodes - 1)
    across_pairs = total_pairs - within_pairs

    # adjacency counts each undirected edge twice (both directions).
    within_edges = int(same_community.sum())
    across_edges = int((~same_community).sum())

    within_density = within_edges / within_pairs
    across_density = across_edges / across_pairs

    assert within_density == pytest.approx(p_in, abs=0.03)
    assert across_density == pytest.approx(p_out, abs=0.01)
    # Sanity: within-community pairs should be connected far more often.
    assert within_density > 10 * across_density


def test_community_labels_shape_and_values() -> None:
    labels = sbm_community_labels(num_communities=4, nodes_per_community=3)

    assert labels.shape == (12,)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


def test_plugs_directly_into_split_edges_and_normalized_adjacency() -> None:
    graph = stochastic_block_model(
        num_communities=4, nodes_per_community=10, p_in=0.3, p_out=0.01, seed=0
    )

    split = split_edges(graph.adjacency, seed=0)
    norm = normalized_adjacency(split.train_adjacency)

    assert norm.shape == (40, 40)
    assert split.val_positive.shape[0] == 2
