"""Tests for the single-beacon ranging task's exact posterior and KL."""

from __future__ import annotations

import math

from scipy import integrate
import torch

from tnbbeta_vae.data.ring_posterior import (
    exact_kl,
    kl_uniform_to_posterior,
    ring_task,
    sphere_log_area,
)
from tnbbeta_vae.distributions import TNBBetaSpherical, VonMisesFisher


def _x_star(task, latitude_degrees: float) -> torch.Tensor:
    w = torch.tensor([math.cos(math.radians(latitude_degrees))], dtype=torch.float64)
    return task.features(w)


def test_latitude_density_integrates_to_one_and_matches_quad() -> None:
    task = ring_task(5, 0.2)
    x = _x_star(task, 60.0)
    theta, density = task.latitude_density(x)
    delta = math.pi / task.grid_size
    assert (
        float((density * delta).sum()) == torch.tensor(1.0).item()
        or abs(float((density * delta).sum()) - 1.0) < 1e-9
    )

    def unnormalised(t: float) -> float:
        w = torch.tensor([math.cos(t)], dtype=torch.float64)
        energy = ((x[0] - task.features(w)[0]) / task.sigma).pow(2).sum() / 2
        return math.exp(-float(energy)) * math.sin(t) ** (task.dim - 2)

    total, _ = integrate.quad(unnormalised, 0, math.pi, points=[math.radians(60)])
    j = int(torch.argmin((theta - math.radians(60)).abs()))
    expected = unnormalised(float(theta[j])) / total
    assert abs(float(density[0, j]) - expected) / expected < 1e-3


def test_surface_density_integrates_to_one_by_monte_carlo() -> None:
    # E_uniform[p_true(z) * Area] = 1, with an independent uniform sampler.
    task = ring_task(3, 0.4)
    x = _x_star(task, 60.0)
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(400_000, 3, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    log_p = task.log_posterior(x, z[:, None, :])[:, 0]
    estimate = log_p.exp().mean() * math.exp(sphere_log_area(3))
    assert abs(float(estimate) - 1.0) < 0.02


def test_evidence_equals_elbo_of_exact_posterior() -> None:
    task = ring_task(4, 0.3)
    generator = torch.Generator().manual_seed(1)
    x, _ = task.sample(3, generator)
    x = x.double()
    # z ~ p_true by importance resampling from a fine uniform pool, per x.
    pool = torch.randn(300_000, 4, generator=generator, dtype=torch.float64)
    pool = pool / pool.norm(dim=-1, keepdim=True)
    for i in range(3):
        weights = task.log_posterior(x[i : i + 1], pool[:, None, :])[:, 0].exp()
        weights = weights / weights.sum()
        # E_{p_true}[log lik + log prior - log p_true] = log p(x).
        log_prior = -sphere_log_area(4)
        integrand = (
            task.log_likelihood(x[i : i + 1], pool[:, None, :])[:, 0]
            + log_prior
            - task.log_posterior(x[i : i + 1], pool[:, None, :])[:, 0]
        )
        estimate = (weights * integrand).sum()
        assert abs(float(estimate) - float(task.log_evidence(x[i : i + 1])[0])) < 0.02


def test_kl_of_uniform_matches_monte_carlo_with_tnbbeta_uniform() -> None:
    for dim in (3, 8):
        task = ring_task(dim, 0.4)
        x = torch.cat([_x_star(task, 30.0), _x_star(task, 80.0)])
        mu = torch.zeros(2, dim, dtype=torch.float64)
        mu[:, 0] = 1.0
        half = torch.full((2,), 0.5, dtype=torch.float64)
        uniform = TNBBetaSpherical(
            mu,
            half,
            torch.zeros(2, dtype=torch.float64),
            torch.full((2,), (dim - 1) / 2, dtype=torch.float64),
        )
        torch.manual_seed(0)
        mc, se = exact_kl(uniform, task, x, num_samples=200_000)
        exact = kl_uniform_to_posterior(task, x)
        assert torch.all((mc - exact).abs() < 5 * se + 1e-3), (mc, exact, se)
        assert torch.all(exact > 0)


def test_vmf_kl_matches_latitude_quadrature() -> None:
    dim, kappa = 5, 4.0
    task = ring_task(dim, 0.4)
    x = _x_star(task, 60.0)
    # vMF centred on the axis: its latitude density is proportional to
    # exp(kappa cos t) sin^(d-2) t, normalised numerically (no Bessel function).
    theta, density_p = task.latitude_density(x)
    delta = math.pi / task.grid_size
    log_q = kappa * torch.cos(theta) + (dim - 2) * torch.log(torch.sin(theta))
    q_density = torch.exp(log_q - torch.logsumexp(log_q + math.log(delta), 0))
    expected = (q_density * (q_density.log() - density_p[0].log()) * delta).sum()
    vmf = VonMisesFisher(
        task.axis.double()[None], torch.full((1, 1), kappa, dtype=torch.float64)
    )
    torch.manual_seed(0)
    mc, se = exact_kl(vmf, task, x, num_samples=300_000)
    assert abs(float(mc[0]) - float(expected)) < 5 * float(se[0]) + 2e-3
