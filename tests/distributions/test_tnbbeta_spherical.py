"""Tests for tnbbeta_vae.distributions.tnbbeta_spherical."""

from __future__ import annotations

import math

import pytest
import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.distributions.tnbbeta_spherical import _householder_reflect


def test_rejects_dim_less_than_two() -> None:
    """A 1-dimensional ambient space isn't a hypersphere (need dim >= 2)."""
    with pytest.raises(ValueError, match="dim >= 2"):
        TNBBetaSpherical(mean_direction=torch.tensor([1.0]), p=0.5, q=0.5, epsilon=1.0)


def test_normalizes_mean_direction() -> None:
    """A non-unit mean_direction is normalized, not rejected."""
    dist = TNBBetaSpherical(
        mean_direction=torch.tensor([3.0, 4.0, 0.0]), p=0.5, q=0.5, epsilon=1.0
    )

    assert torch.allclose(dist.mean_direction.norm(), torch.tensor(1.0))
    assert torch.allclose(dist.mean_direction, torch.tensor([0.6, 0.8, 0.0]))


def test_rsample_produces_unit_norm_points() -> None:
    """Every draw must lie exactly on the sphere."""
    torch.manual_seed(0)
    dist = TNBBetaSpherical(
        mean_direction=torch.tensor([0.0, 0.0, 1.0]), p=0.7, q=0.6, epsilon=1.0
    )

    samples = dist.rsample((2_000,))

    assert samples.shape == (2_000, 3)
    assert torch.allclose(samples.norm(dim=-1), torch.ones(2_000), atol=1e-4)


def test_rsample_concentrates_near_mean_direction() -> None:
    """High p and q should pull mass toward mean_direction."""
    torch.manual_seed(1)
    mu = torch.tensor([0.0, 0.0, 1.0])
    dist = TNBBetaSpherical(mean_direction=mu, p=0.95, q=0.9, epsilon=1.0)

    samples = dist.rsample((5_000,))
    cosine_similarity = (samples * mu).sum(-1)

    assert cosine_similarity.mean() > 0.5


def test_log_prob_matches_monte_carlo_integral_over_sphere() -> None:
    """The density should integrate to ~1 over S^2 (Monte Carlo, uniform proposal)."""
    torch.manual_seed(2)
    dist = TNBBetaSpherical(
        mean_direction=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        p=torch.tensor(0.7, dtype=torch.float64),
        q=torch.tensor(0.6, dtype=torch.float64),
        epsilon=torch.tensor(1.0, dtype=torch.float64),
    )

    g = torch.randn(200_000, 3, dtype=torch.float64)
    uniform_points = g / g.norm(dim=-1, keepdim=True)
    surface_area_s2 = 4 * math.pi

    integral_estimate = surface_area_s2 * dist.log_prob(uniform_points).exp().mean()

    assert torch.abs(integral_estimate - 1.0) < 0.01


def test_log_prob_matches_direct_circle_parameterization() -> None:
    """dim=2 cross-check: log_prob should integrate to ~1 over the circle directly.

    This is an independent check from the Monte Carlo one above: it
    parameterizes S^1 directly by angle (no sampling noise) rather than
    relying on uniform importance sampling.
    """
    dist = TNBBetaSpherical(
        mean_direction=torch.tensor([1.0, 0.0], dtype=torch.float64),
        p=torch.tensor(0.6, dtype=torch.float64),
        q=torch.tensor(0.5, dtype=torch.float64),
        epsilon=torch.tensor(1.0, dtype=torch.float64),
    )

    theta = torch.linspace(1e-4, 2 * math.pi - 1e-4, 400_000, dtype=torch.float64)
    points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    integral = torch.trapz(dist.log_prob(points).exp(), theta)

    assert torch.abs(integral - 1.0) < 1e-3


def test_householder_reflect_maps_pole_to_mean_direction() -> None:
    """Sanity-checks the private reflection helper directly."""
    torch.manual_seed(3)
    mu = torch.randn(5)
    mu = mu / mu.norm()
    pole = torch.zeros(5)
    pole[0] = 1.0

    reflected = _householder_reflect(pole, mu)

    assert torch.allclose(reflected, mu, atol=1e-6)


def test_householder_reflect_is_an_involution() -> None:
    """Applying the reflection twice must recover the original point."""
    torch.manual_seed(4)
    mu = torch.randn(5)
    mu = mu / mu.norm()
    z = torch.randn(5)
    z = z / z.norm()

    twice_reflected = _householder_reflect(_householder_reflect(z, mu), mu)

    assert torch.allclose(twice_reflected, z, atol=1e-5)


def test_householder_reflect_degenerate_case_is_identity() -> None:
    """When mean_direction == pole, no reflection is needed."""
    pole = torch.zeros(4)
    pole[0] = 1.0
    z = torch.randn(4)
    z = z / z.norm()

    assert torch.allclose(_householder_reflect(z, pole), z, atol=1e-6)


def test_rsample_is_differentiable_wrt_all_parameters() -> None:
    """rsample() must be reparameterized: gradients should flow to every parameter."""
    torch.manual_seed(5)
    raw_mu = torch.randn(4, requires_grad=True)
    mu = (raw_mu / raw_mu.norm()).detach().requires_grad_(True)
    p = torch.tensor(0.5, requires_grad=True)
    q = torch.tensor(0.5, requires_grad=True)
    epsilon = torch.tensor(1.0, requires_grad=True)
    dist = TNBBetaSpherical(mean_direction=mu, p=p, q=q, epsilon=epsilon)

    samples = dist.rsample((500,))
    samples.sum().backward()

    assert dist.has_rsample
    for param, grad in (
        (mu, mu.grad),
        (p, p.grad),
        (q, q.grad),
        (epsilon, epsilon.grad),
    ):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, param
