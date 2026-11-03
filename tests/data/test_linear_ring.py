"""Tests for the scalar-observation sphere-PCA exact quantities."""

from __future__ import annotations

import math

import pytest
from scipy import integrate
import torch

from tnbbeta_vae.data.linear_ring import (
    information_nats,
    kl_uniform_to_posterior,
    latitude_std,
    log_evidence,
    sample_observations,
)


@pytest.mark.parametrize("dim", [3, 5, 17])
def test_evidence_agrees_with_finer_grid(dim: int) -> None:
    y = torch.linspace(-1.2, 1.2, 9, dtype=torch.float64)
    coarse = log_evidence(y, dim, 0.1)
    fine = log_evidence(y, dim, 0.1, grid_size=64001)
    assert torch.allclose(coarse, fine, atol=1e-7)


def test_evidence_d3_matches_direct_integral() -> None:
    """For d = 3 the latitude is uniform on [-1, 1]."""
    sigma, y = 0.3, 0.4

    def density(w: float) -> float:
        return (
            math.exp(-0.5 * ((y - w) / sigma) ** 2)
            / (sigma * math.sqrt(2 * math.pi))
            / 2
        )

    expected = math.log(integrate.quad(density, -1, 1)[0])
    got = float(log_evidence(torch.tensor([y], dtype=torch.float64), 3, sigma)[0])
    assert got == pytest.approx(expected, abs=1e-7)


def test_evidence_matches_monte_carlo() -> None:
    dim, sigma = 9, 0.2
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(400_000, dim, generator=generator, dtype=torch.float64)
    w = (z / z.norm(dim=-1, keepdim=True))[:, 0]
    y = torch.tensor([0.1, 0.5], dtype=torch.float64)
    density = torch.exp(-0.5 * ((y[:, None] - w[None]) / sigma) ** 2) / (
        sigma * math.sqrt(2 * math.pi)
    )
    assert torch.allclose(
        torch.log(density.mean(-1)), log_evidence(y, dim, sigma), atol=0.02
    )


def test_custom_mean_function_is_used() -> None:
    y = torch.tensor([0.3], dtype=torch.float64)
    shifted = log_evidence(y, 5, 0.2, lambda w: w + 0.5)
    plain = log_evidence(y - 0.5, 5, 0.2)
    assert torch.allclose(shifted, plain, atol=1e-9)


def test_latitude_std_shrinks_with_sigma() -> None:
    y = torch.tensor([0.3], dtype=torch.float64)
    thin = latitude_std(y, 9, 0.05)
    thick = latitude_std(y, 9, 0.4)
    assert thin < thick
    assert float(thin) == pytest.approx(0.05, rel=0.2)


def test_information_is_positive_and_falls_with_noise() -> None:
    generator = torch.Generator().manual_seed(1)
    y_thin = sample_observations(5, 0.1, 4000, generator).double()
    y_thick = sample_observations(5, 0.4, 4000, generator).double()
    assert information_nats(y_thin, 5, 0.1) > information_nats(y_thick, 5, 0.4) > 0


def test_sample_observations_validates_dim() -> None:
    with pytest.raises(ValueError, match="dim >= 3"):
        sample_observations(2, 0.1, 5)
    y = sample_observations(5, 0.2, 10, torch.Generator().manual_seed(0))
    assert y.shape == (10,)


def test_kl_uniform_to_posterior_is_nonnegative_and_grows_as_sigma_shrinks() -> None:
    y = torch.tensor([0.0, 0.3], dtype=torch.float64)
    thin = kl_uniform_to_posterior(y, 9, 0.05)
    thick = kl_uniform_to_posterior(y, 9, 0.4)
    assert (thick >= 0).all()
    assert (thin > thick).all()
