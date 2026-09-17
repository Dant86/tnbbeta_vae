"""Tests for tnbbeta_vae.models.priors.tnbbeta_spherical."""

from __future__ import annotations

import torch

from tnbbeta_vae.models.priors.tnbbeta_spherical import FixedTNBBetaSphericalPrior


def test_prior_mean_direction_is_the_canonical_pole() -> None:
    prior = FixedTNBBetaSphericalPrior(dim=5, p=0.9, q=0.8, epsilon=1.0)

    dist = prior()

    expected = torch.zeros(5)
    expected[0] = 1.0
    assert torch.allclose(dist.mean_direction, expected)


def test_prior_log_prob_is_finite() -> None:
    prior = FixedTNBBetaSphericalPrior(dim=5, p=0.9, q=0.8, epsilon=1.0)
    dist = prior()

    z = torch.randn(10, 5)
    z = z / z.norm(dim=-1, keepdim=True)

    log_probs = dist.log_prob(z)

    assert torch.isfinite(log_probs).all()


def test_prior_buffer_moves_with_module() -> None:
    """register_buffer should make mean_direction follow .to(dtype)."""
    prior = FixedTNBBetaSphericalPrior(dim=5, p=0.9, q=0.8, epsilon=1.0)

    prior = prior.to(torch.float64)

    assert prior.mean_direction.dtype == torch.float64
