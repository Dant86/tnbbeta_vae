"""Tests for the posterior heads in tnbbeta_vae.models.heads."""

from __future__ import annotations

import pytest
import torch

from tnbbeta_vae.models.heads import tnbbeta_posterior, vmf_posterior


def _raw_with_zero_rows(dim: int) -> torch.Tensor:
    torch.manual_seed(0)
    raw = torch.randn(5, dim)
    raw[1] = 0.0  # e.g. an isolated, featureless graph node
    raw[3] = 1e-7
    return raw.requires_grad_()


def test_vmf_head_gives_unit_directions_even_for_zero_rows() -> None:
    raw_mean = _raw_with_zero_rows(6)

    posterior = vmf_posterior(raw_mean, torch.zeros(5, 1))

    assert torch.isfinite(posterior.loc).all()
    assert torch.allclose(posterior.loc.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.equal(posterior.loc[1], torch.tensor([1.0, 0, 0, 0, 0, 0]))
    assert torch.allclose(posterior.loc[0], raw_mean[0] / raw_mean[0].norm(), atol=1e-6)
    z = posterior.rsample()
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-4)
    z.sum().backward()
    assert raw_mean.grad is not None and torch.isfinite(raw_mean.grad).all()


def test_tnbbeta_head_gives_unit_directions_even_for_zero_rows() -> None:
    dim = 6
    raw = torch.cat([_raw_with_zero_rows(dim), torch.randn(5, 3)], dim=-1)

    posterior = tnbbeta_posterior(raw.detach(), dim)

    assert torch.allclose(
        posterior.mean_direction.norm(dim=-1), torch.ones(5), atol=1e-6
    )
    assert torch.equal(posterior.mean_direction[1], torch.tensor([1.0, 0, 0, 0, 0, 0]))
    assert torch.isfinite(posterior.rsample()).all()


@pytest.mark.parametrize("scale", [1e-3, 1.0, 50.0])
def test_regular_rows_are_unchanged_by_the_fallback(scale: float) -> None:
    raw_mean = scale * torch.randn(4, 5)

    loc = vmf_posterior(raw_mean, torch.zeros(4, 1)).loc

    assert torch.allclose(
        loc, raw_mean / raw_mean.norm(dim=-1, keepdim=True), atol=1e-6
    )
