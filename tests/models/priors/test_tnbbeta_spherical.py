"""Tests for tnbbeta_vae.models.priors.tnbbeta_spherical."""

from __future__ import annotations

import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.priors.tnbbeta_spherical import (
    FixedTNBBetaSphericalPrior,
    uniform_prior_params,
)


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


def test_uniform_prior_params_gives_constant_log_prob() -> None:
    """The whole point: log_prob must not depend on the point, for any dim."""
    torch.manual_seed(0)
    for dim in (3, 4, 8, 16):
        p, q, epsilon = uniform_prior_params(dim)
        mean_direction = torch.zeros(dim, dtype=torch.float64)
        mean_direction[0] = 1.0
        dist = TNBBetaSpherical(
            mean_direction,
            p=torch.tensor(p, dtype=torch.float64),
            q=torch.tensor(q, dtype=torch.float64),
            epsilon=torch.tensor(epsilon, dtype=torch.float64),
        )

        points = torch.randn(200, dim, dtype=torch.float64)
        points = points / points.norm(dim=-1, keepdim=True)
        log_probs = dist.log_prob(points)

        assert torch.allclose(
            log_probs, log_probs[0].expand_as(log_probs), atol=1e-6
        ), f"dim={dim}"


def test_uniform_prior_params_independent_of_mean_direction() -> None:
    """q=0 means the direction shouldn't matter -- different poles, same density."""
    torch.manual_seed(1)
    dim = 6
    p, q, epsilon = uniform_prior_params(dim)
    point = torch.randn(dim, dtype=torch.float64)
    point = point / point.norm()

    log_probs = []
    for _ in range(5):
        direction = torch.randn(dim, dtype=torch.float64)
        direction = direction / direction.norm()
        dist = TNBBetaSpherical(
            direction,
            p=torch.tensor(p, dtype=torch.float64),
            q=torch.tensor(q, dtype=torch.float64),
            epsilon=torch.tensor(epsilon, dtype=torch.float64),
        )
        log_probs.append(dist.log_prob(point))

    stacked = torch.stack(log_probs)
    assert torch.allclose(stacked, stacked[0].expand_as(stacked), atol=1e-6)
