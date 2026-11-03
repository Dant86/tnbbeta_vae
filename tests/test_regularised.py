"""Tests for tnbbeta_vae.regularised."""

from __future__ import annotations

import numpy as np
import pytest

from tnbbeta_vae.regularised import (
    block_permutation,
    fit_elastic_net,
    lambda_grid,
    leave_family_out,
    select_lambda,
)


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(6), 10)
    x = rng.standard_normal((60, 5))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.1 * rng.standard_normal(60)
    return x, y, groups


def test_path_recovers_sparse_signal_and_shrinks_to_zero_at_the_top() -> None:
    x, y, _ = _data()
    lams = lambda_grid(x, y, 0.5)
    fits = fit_elastic_net(x, y, lams)

    assert np.allclose(fits[0].coefficients, 0.0)
    last = fits[-1].coefficients
    assert last[0] > 1.5 and last[1] < -0.7
    assert np.all(np.abs(last[2:]) < 0.3)
    assert fits[-1].predict(x).shape == (60,)


def test_leave_family_out_r2_is_high_on_signal_and_low_on_noise() -> None:
    x, y, groups = _data()
    assert select_lambda(x, y, groups) > 0
    r2, predictions, fits = leave_family_out(x, y, groups)
    assert r2 > 0.9 and predictions.shape == y.shape and len(fits) == 6

    noise = np.random.default_rng(4).standard_normal(60)
    assert leave_family_out(x, noise, groups)[0] < 0.1


def test_block_permutation_moves_whole_groups_and_requires_equal_sizes() -> None:
    groups = np.repeat(np.arange(4), 3)
    permuted = block_permutation(groups, np.random.default_rng(0))

    assert sorted(permuted) == list(range(12))
    for g in range(4):
        sources = {int(i) // 3 for i in permuted[groups == g]}
        assert len(sources) == 1  # each block receives one source group's rows
    with pytest.raises(ValueError):
        block_permutation(np.array([0, 0, 1]), np.random.default_rng(0))
