"""Tests for the ported S-VAE distributions (vMF, hyperspherical uniform)."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import kl_divergence

from tnbbeta_vae.distributions import HypersphericalUniform, VonMisesFisher
from tnbbeta_vae.distributions._ive import ive


def _log_bessel_i(nu: float, x: float) -> float:
    """log I_nu(x) from its power series (a log-sum-exp over terms)."""
    logs = [
        (2 * k + nu) * math.log(x / 2) - math.lgamma(k + 1) - math.lgamma(k + nu + 1)
        for k in range(2000)
    ]
    peak = max(logs)
    return peak + math.log(sum(math.exp(value - peak) for value in logs))


def _batch(n: int, dim: int, kappa: float, seed: int = 0) -> VonMisesFisher:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=generator)
    loc = (direction / direction.norm()).expand(n, dim).contiguous()
    return VonMisesFisher(loc, torch.full((n, 1), kappa))


@pytest.mark.parametrize("nu", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("kappa", [0.5, 5.0, 40.0])
def test_ive_matches_power_series(nu: float, kappa: float) -> None:
    expected = math.exp(_log_bessel_i(nu, kappa) - kappa)

    assert ive(nu, torch.tensor([kappa], dtype=torch.float64)).item() == pytest.approx(
        expected, rel=1e-5
    )


def test_ive_gradient_matches_finite_difference() -> None:
    # The original returns float32, so use a step well above its resolution.
    z = torch.tensor([3.0], requires_grad=True)
    ive(3.0, z).sum().backward()

    step = 1e-2
    plus = ive(3.0, torch.tensor([3.0 + step]))
    minus = ive(3.0, torch.tensor([3.0 - step]))
    assert z.grad is not None
    assert z.grad.item() == pytest.approx(
        ((plus - minus) / (2 * step)).item(), rel=1e-2
    )


@pytest.mark.parametrize("dim", [2, 3, 8])
def test_samples_are_unit_norm_and_match_mean_resultant_length(dim: int) -> None:
    torch.manual_seed(0)
    kappa = 12.0
    distribution = _batch(20_000, dim, kappa)

    z = distribution.sample()

    assert z.shape == (20_000, dim)
    assert torch.allclose(z.norm(dim=-1), torch.ones(20_000), atol=1e-4)
    empirical = (z * distribution.loc).sum(-1).mean()
    expected = math.exp(
        _log_bessel_i(dim / 2, kappa) - _log_bessel_i(dim / 2 - 1, kappa)
    )
    assert empirical.item() == pytest.approx(expected, abs=0.01)


def test_analytic_kl_to_uniform_matches_monte_carlo() -> None:
    torch.manual_seed(1)
    posterior = _batch(100_000, 8, 7.0)
    prior = HypersphericalUniform(7)
    z = posterior.sample()
    monte_carlo = (posterior.log_prob(z) - prior.log_prob(z)).mean()

    analytic = kl_divergence(posterior, prior)

    assert analytic.shape == (100_000,)
    assert analytic.mean().item() == pytest.approx(monte_carlo.item(), abs=0.03)


def test_rsample_gives_gradients_to_loc_and_scale() -> None:
    torch.manual_seed(0)
    loc = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1).requires_grad_()
    scale = torch.full((16, 1), 6.0, requires_grad=True)

    z = VonMisesFisher(loc, scale).rsample()
    (z[:, 0] * 3 + z[:, 1]).sum().backward()

    assert loc.grad is not None and loc.grad.abs().sum() > 0
    assert scale.grad is not None and scale.grad.abs().sum() > 0


def test_hyperspherical_uniform_samples_and_log_prob() -> None:
    torch.manual_seed(0)
    prior = HypersphericalUniform(7)

    z = prior.sample(torch.Size([5]))

    assert z.shape == (5, 8)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    expected = -(math.log(2) + 4 * math.log(math.pi) - math.lgamma(4))
    assert torch.allclose(prior.log_prob(z), torch.full((5,), expected))


def test_samples_stay_on_the_circle_when_loc_is_near_the_pole() -> None:
    torch.manual_seed(0)
    angle = torch.linspace(0, 1e-3, 500)
    loc = torch.stack([angle.cos(), angle.sin()], dim=-1)
    distribution = VonMisesFisher(loc, torch.full((500, 1), 1.0))

    z = distribution.sample()

    assert torch.allclose(z.norm(dim=-1), torch.ones(500), atol=1e-5)
