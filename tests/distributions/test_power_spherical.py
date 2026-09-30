"""Tests for tnbbeta_vae.distributions.power_spherical."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import kl_divergence

from tnbbeta_vae.distributions import HypersphericalUniform, PowerSpherical


def test_rejects_dim_less_than_two() -> None:
    """A 1-dimensional ambient space isn't a hypersphere (need dim >= 2)."""
    with pytest.raises(ValueError, match="dim >= 2"):
        PowerSpherical(mean_direction=torch.tensor([1.0]), kappa=1.0)


def test_normalizes_mean_direction() -> None:
    """A non-unit mean_direction is normalized, not rejected."""
    dist = PowerSpherical(mean_direction=torch.tensor([3.0, 4.0, 0.0]), kappa=1.0)

    assert torch.allclose(dist.mean_direction.norm(), torch.tensor(1.0))
    assert torch.allclose(dist.mean_direction, torch.tensor([0.6, 0.8, 0.0]))


def test_float_kappa_follows_the_mean_directions_device() -> None:
    """Float kappa must not stay on the CPU when mean_direction is elsewhere.

    Uses the ``meta`` device as a stand-in for CUDA, so this runs without a GPU.
    """
    mean_direction = torch.empty(3, device="meta")

    dist = PowerSpherical(mean_direction, kappa=2.0, validate_args=False)

    assert dist.kappa.device.type == "meta"


def test_rsample_produces_unit_norm_points() -> None:
    """Every draw must lie exactly on the sphere."""
    torch.manual_seed(0)
    dist = PowerSpherical(mean_direction=torch.tensor([0.0, 0.0, 1.0]), kappa=5.0)

    samples = dist.rsample((2_000,))

    assert samples.shape == (2_000, 3)
    assert torch.allclose(samples.norm(dim=-1), torch.ones(2_000), atol=1e-4)


def test_rsample_concentrates_near_mean_direction() -> None:
    """High kappa should pull mass toward mean_direction."""
    torch.manual_seed(1)
    mu = torch.tensor([0.0, 0.0, 1.0])
    dist = PowerSpherical(mean_direction=mu, kappa=50.0)

    samples = dist.rsample((5_000,))
    cosine_similarity = (samples * mu).sum(-1)

    assert cosine_similarity.mean() > 0.8


def test_log_prob_matches_monte_carlo_integral_over_sphere() -> None:
    """The density should integrate to ~1 over S^2 (Monte Carlo, uniform proposal)."""
    torch.manual_seed(2)
    dist = PowerSpherical(
        mean_direction=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        kappa=torch.tensor(4.0, dtype=torch.float64),
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
    dist = PowerSpherical(
        mean_direction=torch.tensor([1.0, 0.0], dtype=torch.float64),
        kappa=torch.tensor(3.0, dtype=torch.float64),
    )

    theta = torch.linspace(1e-4, 2 * math.pi - 1e-4, 400_000, dtype=torch.float64)
    points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    integral = torch.trapz(dist.log_prob(points).exp(), theta)

    assert torch.abs(integral - 1.0) < 1e-3


def test_log_prob_matches_reference_normalized_density() -> None:
    """Cross-checks log_prob against a direct evaluation of Theorem 13's formula."""
    dim, kappa = 5, 3.5
    mu = torch.randn(dim, dtype=torch.float64)
    mu = mu / mu.norm()
    dist = PowerSpherical(mu, torch.tensor(kappa, dtype=torch.float64))

    x = torch.randn(7, dim, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)

    alpha = (dim - 1) / 2 + kappa
    beta = (dim - 1) / 2
    log_normalizer = (
        (alpha + beta) * math.log(2)
        + beta * math.log(math.pi)
        + math.lgamma(alpha)
        - math.lgamma(alpha + beta)
    )
    expected = kappa * torch.log1p((mu * x).sum(-1)) - log_normalizer

    assert torch.allclose(dist.log_prob(x), expected, atol=1e-8)


def test_kappa_zero_is_numerically_uniform() -> None:
    """kappa=0 must reduce exactly to Uniform(S^(dim-1)), not merely approximate it."""
    torch.manual_seed(3)
    dim = 4
    mu = torch.randn(dim, dtype=torch.float64)
    mu = mu / mu.norm()
    dist = PowerSpherical(mu, torch.tensor(0.0, dtype=torch.float64))
    prior = HypersphericalUniform(dim - 1)

    x = torch.randn(11, dim, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)

    assert torch.allclose(
        dist.log_prob(x), prior.log_prob(x.float()).double(), atol=1e-6
    )
    assert torch.allclose(
        dist.kl_to_uniform(), torch.zeros((), dtype=torch.float64), atol=1e-6
    )


@pytest.mark.parametrize(("dim", "kappa"), [(3, 5.0), (5, 20.0), (2, 0.5), (8, 1.0)])
def test_kl_to_uniform_matches_monte_carlo(dim: int, kappa: float) -> None:
    torch.manual_seed(4)
    mu = torch.randn(dim)
    mu = mu / mu.norm()
    dist = PowerSpherical(mu, torch.tensor(kappa))
    prior = HypersphericalUniform(dim - 1)

    z = dist.rsample((200_000,))
    monte_carlo = (dist.log_prob(z) - prior.log_prob(z)).mean()

    analytic = dist.kl_to_uniform()

    assert analytic.item() >= 0
    assert analytic.item() == pytest.approx(monte_carlo.item(), abs=0.05)


def test_registered_kl_matches_kl_to_uniform() -> None:
    """`kl_divergence` (used by `monte_carlo_elbo(analytic_kl=True)`) must agree."""
    mu = torch.randn(6)
    mu = mu / mu.norm()
    dist = PowerSpherical(mu.expand(10, 6), torch.full((10,), 7.0))
    prior = HypersphericalUniform(5)

    registered = kl_divergence(dist, prior)

    assert torch.allclose(registered, dist.kl_to_uniform())


def test_rsample_is_differentiable_wrt_mean_direction_and_kappa() -> None:
    """rsample() must be reparameterized: gradients should flow to every parameter."""
    torch.manual_seed(5)
    raw_mu = torch.randn(4, requires_grad=True)
    mu = (raw_mu / raw_mu.norm()).detach().requires_grad_(True)
    kappa = torch.tensor(3.0, requires_grad=True)
    dist = PowerSpherical(mean_direction=mu, kappa=kappa)

    samples = dist.rsample((500,))
    samples.sum().backward()

    assert dist.has_rsample
    for param, grad in ((mu, mu.grad), (kappa, kappa.grad)):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, param


def test_entropy_and_kl_are_differentiable_wrt_kappa() -> None:
    """The analytic KL must pass a gradient back to kappa (used for VAE training)."""
    mu = torch.tensor([0.0, 0.0, 1.0])
    kappa = torch.tensor(4.0, requires_grad=True)
    dist = PowerSpherical(mu, kappa)

    dist.kl_to_uniform().backward()

    assert kappa.grad is not None
    assert torch.isfinite(kappa.grad).all()
