"""Tests for tnbbeta_vae.data.block_graph."""

from __future__ import annotations

import numpy as np

from tnbbeta_vae.data.block_graph import (
    assortative_spec,
    bipartite_spec,
    block_graph,
    hub_spoke_spec,
)


def test_probability_is_symmetric_with_zero_diagonal_and_matches_the_affinity() -> None:
    sample = block_graph(assortative_spec(3, 10, p_in=0.4, p_out=0.05), seed=0)

    assert np.allclose(sample.probability, sample.probability.T)
    assert np.all(np.diag(sample.probability) == 0)
    assert sample.probability[0, 1] == 0.4
    assert sample.probability[0, 15] == 0.05
    assert sample.kind[0, 1] == "within"
    assert sample.kind[0, 15] == "between"


def test_adjacency_is_symmetric_binary_and_has_the_right_density() -> None:
    sample = block_graph(bipartite_spec(150, p_cross=0.3, p_within=0.0), seed=1)
    dense = sample.adjacency.toarray()

    assert np.array_equal(dense, dense.T)
    assert set(np.unique(dense)) <= {0.0, 1.0}
    assert np.all(np.diag(dense) == 0)
    assert dense[:150, :150].sum() == 0  # no within-side edges
    assert abs(dense[:150, 150:].mean() - 0.3) < 0.03


def test_hub_spoke_links_hubs_only_to_their_own_leaves() -> None:
    spec = hub_spoke_spec(num_hubs=3, leaves_per_hub=4, p_hub_leaf=1.0)
    sample = block_graph(spec, seed=0)
    hub_0, leaf_of_hub_0, leaf_of_hub_1 = 0, 3, 7

    assert sample.probability[hub_0, leaf_of_hub_0] == 1.0
    assert sample.probability[hub_0, leaf_of_hub_1] == 0.0
    assert sample.kind[hub_0, leaf_of_hub_0] == "hub-own-leaf"
    assert sample.kind[hub_0, leaf_of_hub_1] == "hub-other-leaf"
    assert sample.kind[leaf_of_hub_0, leaf_of_hub_0 + 1] == "leaf-leaf-same-hub"
    assert sample.kind[leaf_of_hub_0, leaf_of_hub_1] == "leaf-leaf-diff-hub"
    assert sample.kind[0, 1] == "hub-hub"
    assert sample.adjacency[hub_0, leaf_of_hub_0] == 1.0


def test_degree_exponent_skews_degrees_and_is_reproducible() -> None:
    plain = block_graph(assortative_spec(2, 150, 0.1, 0.1), seed=0)
    skewed = block_graph(
        assortative_spec(2, 150, 0.1, 0.1), seed=0, degree_exponent=2.5
    )
    again = block_graph(assortative_spec(2, 150, 0.1, 0.1), seed=0, degree_exponent=2.5)

    degrees = np.asarray(skewed.adjacency.sum(1)).ravel()
    flat = np.asarray(plain.adjacency.sum(1)).ravel()
    assert degrees.std() / degrees.mean() > 2 * flat.std() / flat.mean()
    assert np.array_equal(skewed.adjacency.toarray(), again.adjacency.toarray())
