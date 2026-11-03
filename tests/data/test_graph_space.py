"""Tests for tnbbeta_vae.data.graph_space."""

from __future__ import annotations

import numpy as np
import pytest

from tnbbeta_vae.data.graph_space import (
    FAMILIES,
    GraphSpec,
    calibrate_contrast,
    expected_oracle_auc,
    sample_graph,
    true_probability,
)


def _spec(family: str, **overrides: object) -> GraphSpec:
    fields = {
        "family": family,
        "num_nodes": 200,
        "mean_degree": 10.0,
        "num_groups": 4,
        "heterogeneity": "none",
        "features": "none",
        **overrides,
    }
    return GraphSpec(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_gives_a_symmetric_truth_with_the_requested_mean_degree(
    family: str,
) -> None:
    spec = _spec(family, contrast=3.0)
    probability = true_probability(spec, np.random.default_rng(0))

    assert np.allclose(probability, probability.T)
    assert np.all(np.diag(probability) == 0)
    assert probability.min() >= 0.0 and probability.max() <= 1.0
    assert abs(probability.sum() / 200 - 10.0) < 0.5


def test_draws_with_different_seeds_differ_and_same_seed_repeats() -> None:
    spec = _spec("latent_kernel", contrast=3.0)
    a, b, c = sample_graph(spec, 1), sample_graph(spec, 1), sample_graph(spec, 2)

    assert np.array_equal(a.adjacency.toarray(), b.adjacency.toarray())
    assert not np.array_equal(a.adjacency.toarray(), c.adjacency.toarray())
    dense = a.adjacency.toarray()
    assert np.array_equal(dense, dense.T) and np.all(np.diag(dense) == 0)


def test_oracle_auc_is_one_half_without_structure_and_one_with_perfect_structure() -> (
    None
):
    flat = np.full((50, 50), 0.1)
    np.fill_diagonal(flat, 0.0)
    assert abs(expected_oracle_auc(flat) - 0.5) < 1e-9

    spec = _spec("assortative", contrast=1e-3)
    weak = expected_oracle_auc(true_probability(spec, np.random.default_rng(0)))
    strong = expected_oracle_auc(
        true_probability(_spec("assortative", contrast=1e4), np.random.default_rng(0))
    )
    assert weak < 0.52 and strong > 0.85


@pytest.mark.parametrize(
    "family", ["assortative", "latent_kernel", "ring", "geometric"]
)
def test_calibration_hits_the_target_auc(family: str) -> None:
    calibrated = calibrate_contrast(_spec(family), 0.8)

    assert abs(calibrated.achieved_auc - 0.8) < 0.01
    assert calibrated.target_auc == 0.8


def test_a_target_the_family_cannot_reach_is_reported_not_forced() -> None:
    calibrated = calibrate_contrast(_spec("bipartite"), 0.95)

    assert calibrated.achieved_auc < 0.8  # two sides cap the AUC near 0.75


def test_informative_features_have_the_projection_width_and_identity_otherwise() -> (
    None
):
    informative = sample_graph(
        _spec("assortative", features="informative", contrast=5.0), 0
    )
    plain = sample_graph(_spec("assortative", contrast=5.0), 0)

    assert informative.features.shape == (200, 32)
    assert plain.features.shape == (200, 200) and plain.features.nnz == 200


def test_calibration_survives_a_contrast_that_underflows_the_affinity() -> None:
    spec = _spec("latent_kernel", num_groups=16)

    calibrated = calibrate_contrast(spec, 0.95)

    assert np.isfinite(calibrated.achieved_auc)
    assert calibrated.contrast > 0
