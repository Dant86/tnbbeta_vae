"""Tests for tnbbeta_vae.data.block_graph."""

from __future__ import annotations

import numpy as np

from tnbbeta_vae.data.block_graph import (
    assortative_spec,
    bipartite_spec,
    block_graph,
    core_periphery_spec,
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


def test_core_periphery_links_cores_densely_and_peripheries_only_to_their_core() -> (
    None
):
    spec = core_periphery_spec(
        num_groups=2,
        core_size=3,
        periphery_size=4,
        p_core_core=1.0,
        p_core_periphery=1.0,
        p_other=0.0,
    )
    sample = block_graph(spec, seed=0)
    core_0, core_0_other, core_1, periphery_0, periphery_0_other, periphery_1 = (
        0,
        1,
        3,
        6,
        7,
        10,
    )

    assert sample.probability[core_0, core_0_other] == 1.0
    assert sample.probability[core_0, periphery_0] == 1.0
    assert sample.probability[core_0, core_1] == 0.0
    assert sample.probability[core_0, periphery_1] == 0.0
    assert sample.probability[periphery_0, periphery_0_other] == 0.0
    assert sample.kind[core_0, core_0_other] == "core-core-same"
    assert sample.kind[core_0, core_1] == "core-core-diff"
    assert sample.kind[core_0, periphery_0] == "core-own-periphery"
    assert sample.kind[core_0, periphery_1] == "core-other-periphery"
    assert sample.kind[periphery_0, periphery_0_other] == "periphery-periphery-same"
    assert sample.kind[periphery_0, periphery_1] == "periphery-periphery-diff"
    assert sample.adjacency[core_0, periphery_0] == 1.0


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
