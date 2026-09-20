"""Tests for the Bernoulli pixel likelihood on 28x28 single-channel images."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Independent, Normal

from tnbbeta_vae.models import (
    ConvGaussianVAE,
    ConvGaussianVAEConfig,
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
)
from tnbbeta_vae.models.losses import monte_carlo_elbo


def _models() -> list[torch.nn.Module]:
    common = {
        "image_channels": 1,
        "image_size": 28,
        "hidden_channels": 8,
        "latent_dim": 5,
        "likelihood": "bernoulli",
    }
    return [
        ConvGaussianVAE(ConvGaussianVAEConfig(**common)),
        ConvTNBBetaSphericalVAE(ConvTNBBetaSphericalVAEConfig(**common)),
    ]


def test_bernoulli_log_likelihood_matches_the_closed_form() -> None:
    torch.manual_seed(0)
    x = torch.bernoulli(torch.rand(3, 1, 4, 4))
    logits = torch.randn(3, 1, 4, 4)
    posterior = Independent(Normal(torch.zeros(3, 2), torch.ones(3, 2)), 1)

    terms = monte_carlo_elbo(
        x, posterior, posterior, lambda _z: logits, likelihood="bernoulli"
    )

    probability = torch.sigmoid(logits)
    expected = (x * probability.log() + (1 - x) * (1 - probability).log()).sum(
        (1, 2, 3)
    )
    assert torch.allclose(terms["log_likelihood"], expected, atol=1e-5)


@pytest.mark.parametrize("model", _models())
def test_step_shapes_kl_weight_and_generation(model: torch.nn.Module) -> None:
    batch = torch.bernoulli(torch.rand(4, 1, 28, 28))
    torch.manual_seed(0)
    full = model.training_step(batch)  # pyright: ignore[reportCallIssue]
    torch.manual_seed(0)
    likelihood_only = model.training_step(batch, kl_weight=0.0)  # pyright: ignore[reportCallIssue]

    assert torch.isfinite(full["loss"])
    assert torch.allclose(full["loss"], -(full["log_likelihood"] - full["kl"]))
    assert torch.allclose(likelihood_only["loss"], -likelihood_only["log_likelihood"])
    samples = model.generate(3)  # pyright: ignore[reportCallIssue]
    assert samples.shape == (3, 1, 28, 28)
    assert samples.min() >= 0 and samples.max() <= 1
    reconstruction = model(batch)[0]
    assert reconstruction.shape == batch.shape
