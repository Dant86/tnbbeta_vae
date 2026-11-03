"""Tests for the numeric helpers behind apps.eval.param_probe."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from apps.eval.param_probe import (
    clustering_coefficient,
    cv_r2,
    effective_rank,
    parameter_map,
    partial_spearman,
)


def test_effective_rank_is_one_for_copies_of_a_scalar_and_k_for_independent() -> None:
    rng = np.random.default_rng(0)
    scalar = rng.standard_normal(2000)
    copies = np.column_stack([scalar, 2 * scalar + 1, -scalar])
    independent = rng.standard_normal((2000, 3))

    assert effective_rank(copies) < 1.01
    assert 2.8 < effective_rank(independent) <= 3.0


def test_partial_spearman_removes_a_shared_driver() -> None:
    rng = np.random.default_rng(1)
    driver = rng.standard_normal(3000)
    a = driver + 0.1 * rng.standard_normal(3000)
    b = driver + 0.1 * rng.standard_normal(3000)

    assert np.corrcoef(a, b)[0, 1] > 0.9
    assert abs(partial_spearman(a, b, driver)) < 0.1


def test_cv_r2_is_high_for_a_monotone_relation_and_near_zero_for_noise() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1500, 1))

    assert cv_r2(x, np.exp(x[:, 0])) > 0.95
    assert abs(cv_r2(x, rng.standard_normal(1500))) < 0.05


def test_clustering_coefficient_of_a_triangle_with_a_pendant() -> None:
    # Triangle 0-1-2 plus pendant 3 attached to 2.
    rows = [0, 1, 2, 0, 1, 2, 2, 3]
    cols = [1, 2, 0, 2, 0, 1, 3, 2]
    adjacency = sp.csr_matrix((np.ones(8), (rows, cols)), shape=(4, 4))

    coefficients = clustering_coefficient(adjacency)

    np.testing.assert_allclose(coefficients, [1.0, 1.0, 1 / 3, 0.0])


def test_parameter_map_drops_constant_parameters_and_reports_degree_gain() -> None:
    rng = np.random.default_rng(3)
    degree = rng.standard_normal(2000)
    other = rng.standard_normal(2000)
    covariates = {"log_degree": degree, "other": other + degree}
    parameters = {
        "kappa": degree + 0.05 * rng.standard_normal(2000),
        "fixed": np.full(2000, 3.0),
    }

    result = parameter_map(parameters, covariates)

    assert result["parameters"] == ["kappa"]
    assert result["spearman"]["log_degree"]["kappa"] > 0.95
    assert result["r2_over_degree"]["other"]["degree_only"] > 0
    assert "effective_rank" not in result
