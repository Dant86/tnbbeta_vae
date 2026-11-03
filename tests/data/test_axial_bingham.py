"""Tests for the axial Bingham task's exact quantities."""

from __future__ import annotations

import math

import pytest
import torch

from tnbbeta_vae.data.axial_bingham import (
    AxialBinghamTask,
    bingham_log_normaliser,
    num_sym,
    sphere_quadrature,
    vec_sym,
)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_normaliser_matches_monte_carlo(dim: int) -> None:
    torch.manual_seed(0)
    eigenvalues = torch.tensor([[-1.0, 0.0, 1.0, 2.5][-dim:]], dtype=torch.float64)
    z = torch.randn(400_000, dim, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    mc = torch.logsumexp((z**2 * eigenvalues).sum(-1), 0) - math.log(len(z))
    exact = bingham_log_normaliser(eigenvalues)[0]
    assert abs(float(exact - mc)) < 5e-3


@pytest.mark.parametrize("dim", [2, 3])
def test_normaliser_matches_sphere_grid_when_peaked(dim: int) -> None:
    eigenvalues = torch.tensor([[0.0, 3.0, 60.0][-dim:]], dtype=torch.float64)
    z, log_w = sphere_quadrature(dim)
    grid = torch.logsumexp((z**2 * eigenvalues).sum(-1) + log_w, 0)
    assert abs(float(bingham_log_normaliser(eigenvalues)[0] - grid)) < 1e-8


def test_quadrature_weights_sum_to_one() -> None:
    for dim in (2, 3):
        _, log_w = sphere_quadrature(dim)
        assert abs(float(log_w.exp().sum()) - 1) < 1e-12


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_evidence_and_posterior_are_consistent(dim: int) -> None:
    task = AxialBinghamTask(dim, 0.4)
    x, z = task.sample(5, torch.Generator().manual_seed(1))
    x = x.double()
    assert x.shape == (5, dim * dim)
    # log p(x) = log p(x | z) + log prior(z) - log p(z | x) for any z (Bayes), uniform
    # prior.
    z = z.double()
    log_area = math.log(2) + (dim / 2) * math.log(math.pi) - math.lgamma(dim / 2)
    lhs = task.log_evidence(x)
    rhs = task.log_likelihood(x, z) - log_area - task.log_posterior(x, z)
    assert torch.allclose(lhs, rhs, atol=1e-7)


def test_posterior_is_antipodal() -> None:
    task = AxialBinghamTask(3, 0.3)
    x, z = task.sample(4, torch.Generator().manual_seed(2))
    x, z = x.double(), z.double()
    assert torch.allclose(task.log_posterior(x, z), task.log_posterior(x, -z))


def test_vec_sym_norm_and_size() -> None:
    z = torch.randn(7, 4)
    z = z / z.norm(dim=-1, keepdim=True)
    v = vec_sym(z)
    assert v.shape == (7, num_sym(4))
    assert torch.allclose(v.norm(dim=-1), torch.ones(7), atol=1e-6)


def test_unsupported_dimension_raises() -> None:
    with pytest.raises(ValueError):
        bingham_log_normaliser(torch.zeros(1, 5, dtype=torch.float64))
