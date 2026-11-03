"""Tests for tnbbeta_vae.data.graph_covariates."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from tnbbeta_vae.data.graph_covariates import (
    COVARIATE_NAMES,
    config_axes,
    draw_covariates,
    draw_statistics,
)
from tnbbeta_vae.data.graph_space import GraphSpec, SyntheticGraph
from tnbbeta_vae.data.planetoid import split_edges


def test_draw_covariates_are_finite_reproducible_and_complete() -> None:
    spec = GraphSpec("assortative", 90, 8.0, 3, "mild", "none", 5.0, 0.8, 0.8)

    first = draw_covariates(spec, 3)

    assert set(COVARIATE_NAMES) <= set(first)
    assert all(math.isfinite(first[k]) for k in COVARIATE_NAMES)
    assert first == draw_covariates(spec, 3)
    assert first != draw_covariates(spec, 4)
    assert 0.0 <= first["clustering"] <= 1.0
    assert 0.0 <= first["giant_fraction"] <= 1.0
    assert first["kl0"] > 0


def test_statistics_on_an_exactly_known_graph() -> None:
    clique = np.ones((6, 6)) - np.eye(6)
    block = np.kron(np.eye(2), clique).astype(
        np.float64
    )  # two disjoint 6-cliques, P = adjacency
    adjacency = sp.csr_matrix(block)
    sample = SyntheticGraph(adjacency, block, sp.csr_matrix(sp.identity(12)))
    split = split_edges(adjacency, seed=0)

    stats = draw_statistics(sample, split)

    assert stats["clustering"] == 1.0
    assert stats["realised_degree"] == 5.0
    assert stats["expected_degree_cv"] == 0.0
    assert stats["assortativity"] == 0.0  # every degree is equal
    assert math.isclose(stats["clip_fraction"], 30 / 66)
    assert math.isclose(stats["high_prob_fraction"], 30 / 66)
    assert math.isclose(stats["mass_top1pct"], 1 / 30)
    # |eigenvalues| are 5, 5 and ten 1s
    shares = np.array([5, 5] + [1] * 10) / 20
    assert math.isclose(
        stats["affinity_erank"], float(np.exp(-(shares * np.log(shares)).sum()))
    )
    assert 0.0 < stats["giant_fraction"] <= 0.5 + 1e-9


def test_config_axes_code_group_free_families() -> None:
    ring = config_axes(
        GraphSpec("ring", 300, 10.0, 7, "none", "informative", 1, 0.7, 0.7)
    )
    block = config_axes(
        GraphSpec("assortative", 300, 10.0, 7, "heavy", "none", 1, 0.7, 0.7)
    )

    assert ring["uses_groups"] == 0.0 and ring["log_groups"] == 0.0
    assert block["uses_groups"] == 1.0 and math.isclose(
        block["log_groups"], math.log(7)
    )
    assert block["hetero_level"] == 2.0 and ring["informative_features"] == 1.0
