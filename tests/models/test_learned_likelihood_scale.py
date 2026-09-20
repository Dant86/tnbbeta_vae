"""Tests for the learned Gaussian-likelihood scale."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.distributions import Normal

from tnbbeta_vae.models.losses import LearnedLikelihoodScale
from tnbbeta_vae.registry import build_model

_MODELS = ["conv_gaussian_vae", "conv_vmf_vae", "conv_tnbbeta_spherical_vae"]


def _build(name: str, **overrides: object) -> Any:
    return build_model(name, latent_dim=4, hidden_channels=8, **overrides)


def test_initial_scale_matches_argument() -> None:
    assert LearnedLikelihoodScale(0.3)().item() == pytest.approx(0.3)


def test_optimum_is_the_root_mean_squared_error() -> None:
    torch.manual_seed(0)
    residuals = torch.randn(4000) * 0.3
    scale = LearnedLikelihoodScale(1.0)
    optimizer = torch.optim.Adam(scale.parameters(), lr=0.05)

    for _ in range(600):
        optimizer.zero_grad()
        loss = -Normal(0.0, scale()).log_prob(residuals).mean()
        loss.backward()
        optimizer.step()

    assert scale().item() ** 2 == pytest.approx(
        residuals.pow(2).mean().item(), rel=0.05
    )


@pytest.mark.parametrize("name", _MODELS)
def test_scale_starts_at_the_configured_value_and_receives_gradients(name: str) -> None:
    torch.manual_seed(0)
    model = _build(name, likelihood_scale=0.5)

    out = model.training_step(torch.rand(4, 3, 32, 32))
    out["loss"].backward()

    assert out["likelihood_scale"].item() == pytest.approx(0.5)
    parameter = model.learned_scale.log_variance
    assert parameter.grad is not None and parameter.grad.abs() > 0


@pytest.mark.parametrize("name", _MODELS)
def test_scale_is_a_model_parameter(name: str) -> None:
    model = _build(name, likelihood_scale=0.5)

    assert "learned_scale.log_variance" in model.state_dict()
