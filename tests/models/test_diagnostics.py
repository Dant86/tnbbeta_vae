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


def test_values_match_the_posterior_parameters() -> None:
    mean_direction = torch.eye(3)[:2]
    posterior = TNBBetaSpherical(
        mean_direction,
        p=torch.tensor([0.2, 0.6]),
        q=torch.tensor([0.1, 0.3]),
        epsilon=torch.tensor([2.0, 4.0]),
    )

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert torch.isclose(diagnostics["posterior_p_mean"], torch.tensor(0.4))
    assert diagnostics["posterior_q_max"] == 0.3
    assert diagnostics["posterior_epsilon_mean"] == 3.0


def _random_posterior(batch_size: int, dim: int) -> TNBBetaSpherical:
    mean_direction = torch.randn(batch_size, dim)
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)
    p = torch.rand(batch_size) * 0.4 + 0.3
    q = torch.rand(batch_size) * 0.4 + 0.3
    return TNBBetaSpherical(mean_direction, p=p, q=q, epsilon=1.0)
