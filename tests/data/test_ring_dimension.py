"""Tests for the single-axis latitude-shell task at any ambient dimension."""

from __future__ import annotations

import math

import pytest
import torch

from tnbbeta_vae.data.ring_classes import ring_class_task
from tnbbeta_vae.data.ring_dimension import (
    DIMENSIONS,
    dimension_ring_task,
    information_summary,
)
from tnbbeta_vae.data.ring_posterior import (
    exact_kl,
    kl_uniform_to_posterior,
    latitude_grid,
    sphere_log_area,
)
from tnbbeta_vae.distributions import VonMisesFisher


def _data(dim: int, num: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    task = dimension_ring_task(dim)
    x, z = task.sample(num, torch.Generator().manual_seed(3))
    return x.double(), z.double()


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_latitude_density_is_normalised(dim: int) -> None:
    task = dimension_ring_task(dim)
    x, _ = _data(dim)
    theta, density = task.latitude_density(x)
    assert theta.shape == (task.grid_size,)
    mass = (density * (math.pi / task.grid_size)).sum(-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-10)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_log_evidence_is_converged_in_the_grid(dim: int) -> None:
    x, _ = _data(dim)
    coarse = dimension_ring_task(dim).log_evidence(x)
    fine = dimension_ring_task(dim, grid_size=40_001).log_evidence(x)
    assert (coarse - fine).abs().max() < 1e-7


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_log_evidence_matches_monte_carlo(dim: int) -> None:
    task = dimension_ring_task(dim)
    x, _ = _data(dim, 8)
    generator = torch.Generator().manual_seed(11)
    z = torch.randn(400_000, dim, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    likelihood = task.log_likelihood(x[:, None, :], z[None]).exp().mean(-1).log()
    # The evidence is dominated by the bulk of the prior, so MC is accurate to ~1%.
    assert (likelihood - task.log_evidence(x)).abs().max() < 0.03


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_posterior_integrates_to_one_over_the_sphere(dim: int) -> None:
    """``E_U[p(z|x) / (1 / area)]`` is 1, i.e. the surface density is normalised."""
    task = dimension_ring_task(dim)
    x, _ = _data(dim, 4)
    generator = torch.Generator().manual_seed(2)
    z = torch.randn(200_000, dim, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    log_p = task.log_posterior(x[:, None, :], z[None])
    ratio = (log_p + sphere_log_area(dim)).exp().mean(-1)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=0.03)


def test_dim3_reproduces_the_existing_sphere_code() -> None:
    task = dimension_ring_task(3, grid_size=4001)  # the existing code's grid
    old = ring_class_task(1)
    x, _ = _data(3)
    assert torch.allclose(task.log_evidence(x), old.log_evidence(x), atol=1e-9)
    assert torch.allclose(
        kl_uniform_to_posterior(task, x), old.kl_uniform_to_posterior(x), atol=1e-9
    )
    w = torch.linspace(-0.9, 0.9, 7, dtype=torch.float64)
    assert torch.allclose(
        task.log_posterior_at_latitude(x[:, None, :], w),
        old.log_posterior_at_latitude(x[:, None, :], w),
        atol=1e-9,
    )
    assert torch.allclose(task.features(w), old.features(w))


@pytest.mark.parametrize("dim", [3, 9])
def test_exact_kl_of_the_exact_posterior_family_is_nonnegative(dim: int) -> None:
    """A vMF centred on ``a`` has a KL to the exact posterior >= 0 (and finite)."""
    task = dimension_ring_task(dim)
    x, _ = _data(dim, 4)
    posterior = VonMisesFisher(
        task.axis.double().expand(4, dim), torch.full((4, 1), 3.0, dtype=torch.float64)
    )
    kl, se = exact_kl(posterior, task, x, num_samples=4000)
    assert torch.isfinite(kl).all()
    assert (kl > -4 * se).all()


def test_information_decreases_with_dimension_and_is_finite() -> None:
    summaries = {d: information_summary(dimension_ring_task(d), 4000) for d in (3, 17)}
    assert all(math.isfinite(v) for s in summaries.values() for v in s.values())
    assert summaries[3]["mutual_information"] > summaries[17]["mutual_information"]
    assert summaries[17]["prior_w_std"] == pytest.approx(1 / math.sqrt(17))


def test_latitude_grid_midpoints_stay_off_the_poles() -> None:
    theta, _ = latitude_grid(4001, torch.float64)
    assert theta.min() > 0 and theta.max() < math.pi
