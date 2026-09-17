"""Tests for tnbbeta_vae.models.diagnostics."""

from __future__ import annotations

import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.diagnostics import tnbbeta_spherical_posterior_diagnostics


def test_diagnostics_contain_expected_keys_and_are_finite() -> None:
    posterior = _random_posterior(batch_size=6, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    expected_keys = {
        "posterior_p_mean",
        "posterior_p_min",
        "posterior_p_max",
        "posterior_q_mean",
        "posterior_q_min",
        "posterior_q_max",
        "posterior_epsilon_mean",
        "posterior_direction_pairwise_cosine_mean",
    }
    assert set(diagnostics) == expected_keys
    for value in diagnostics.values():
        assert value.dim() == 0
        assert torch.isfinite(value)


def test_min_mean_max_are_ordered() -> None:
    posterior = _random_posterior(batch_size=10, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert diagnostics["posterior_p_min"] <= diagnostics["posterior_p_mean"]
    assert diagnostics["posterior_p_mean"] <= diagnostics["posterior_p_max"]
    assert diagnostics["posterior_q_min"] <= diagnostics["posterior_q_mean"]
    assert diagnostics["posterior_q_mean"] <= diagnostics["posterior_q_max"]


def test_pairwise_cosine_key_absent_for_batch_size_one() -> None:
    posterior = _random_posterior(batch_size=1, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert "posterior_direction_pairwise_cosine_mean" not in diagnostics


def test_pairwise_cosine_is_one_for_a_collapsed_posterior() -> None:
    """A posterior collapsed onto a single direction should read cosine ~= 1."""
    mean_direction = torch.zeros(6, 4)
    mean_direction[:, 0] = 1.0
    posterior = TNBBetaSpherical(
        mean_direction, p=torch.full((6,), 0.01), q=torch.full((6,), 0.999), epsilon=1.0
    )

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert diagnostics["posterior_direction_pairwise_cosine_mean"] > 0.999


def test_pairwise_cosine_is_near_zero_for_diverse_directions() -> None:
    """Independent random directions in dim=4 should read cosine near 0, not 1."""
    torch.manual_seed(0)
    posterior = _random_posterior(batch_size=200, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert diagnostics["posterior_direction_pairwise_cosine_mean"].abs() < 0.2


def _random_posterior(batch_size: int, dim: int) -> TNBBetaSpherical:
    mean_direction = torch.randn(batch_size, dim)
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)
    p = torch.rand(batch_size) * 0.4 + 0.3
    q = torch.rand(batch_size) * 0.4 + 0.3
    return TNBBetaSpherical(mean_direction, p=p, q=q, epsilon=1.0)
