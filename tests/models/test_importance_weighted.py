"""Tests for tnbbeta_vae.models.losses.importance_weighted."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Independent, Normal

from tnbbeta_vae.models import (
    ConvGaussianVAE,
    ConvGaussianVAEConfig,
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
    ConvVonMisesFisherVAE,
    ConvVonMisesFisherVAEConfig,
)
from tnbbeta_vae.models.losses import importance_weighted_metrics, pixel_log_likelihood

_NOISE = 0.7


class _LinearGaussian:
    """z ~ N(0, 1), x | z ~ N(z, NOISE^2), with the exact posterior as the proposal."""

    def posterior_and_prior(self, x: torch.Tensor) -> tuple[Independent, Independent]:
        scale = _NOISE**2 / (1 + _NOISE**2)
        posterior = Independent(
            Normal(x / (1 + _NOISE**2), torch.full_like(x, math.sqrt(scale))), 1
        )
        prior = Independent(Normal(torch.zeros_like(x), torch.ones_like(x)), 1)
        return posterior, prior

    def log_likelihood(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return Normal(z, _NOISE).log_prob(x).sum(-1)


def test_exact_posterior_makes_every_importance_weight_the_marginal() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 1) * 1.5

    metrics = importance_weighted_metrics(
        _LinearGaussian(), x, num_samples=7, chunk_size=3
    )

    marginal = Normal(0.0, math.sqrt(1 + _NOISE**2)).log_prob(x).squeeze(-1)
    assert torch.allclose(metrics["ll"], marginal, atol=1e-4)
    # The ELBO's reconstruction term is a sample mean, so it is only close.
    many = importance_weighted_metrics(
        _LinearGaussian(), x, num_samples=4000, chunk_size=1000
    )
    assert torch.allclose(many["elbo"], marginal, atol=0.05)
    assert torch.allclose(metrics["elbo"], metrics["re"] - metrics["kl"])
    assert metrics["kl"].shape == (6,)
    assert (metrics["kl"] >= 0).all()


def _small_models() -> list[torch.nn.Module]:
    common = {
        "image_channels": 1,
        "image_size": 28,
        "hidden_channels": 8,
        "latent_dim": 4,
        "likelihood": "bernoulli",
    }
    return [
        ConvGaussianVAE(ConvGaussianVAEConfig(**common)),
        ConvVonMisesFisherVAE(ConvVonMisesFisherVAEConfig(**common)),
        ConvTNBBetaSphericalVAE(ConvTNBBetaSphericalVAEConfig(**common)),
    ]


@pytest.mark.parametrize("model", _small_models())
def test_models_expose_the_evaluation_interface(model: torch.nn.Module) -> None:
    torch.manual_seed(0)
    x = torch.bernoulli(torch.rand(3, 1, 28, 28))

    metrics = importance_weighted_metrics(model, x, num_samples=20, chunk_size=8)

    for name in ("ll", "elbo", "re", "kl"):
        assert metrics[name].shape == (3,)
        assert torch.isfinite(metrics[name]).all()
    assert torch.allclose(metrics["elbo"], metrics["re"] - metrics["kl"])
    assert metrics["ll"].mean() >= metrics["elbo"].mean() - 1.0


def test_pixel_log_likelihood_broadcasts_over_leading_sample_dimensions() -> None:
    x = torch.bernoulli(torch.rand(3, 1, 4, 4))
    logits = torch.randn(5, 3, 1, 4, 4)

    bernoulli = pixel_log_likelihood(x, logits, "bernoulli")
    gaussian = pixel_log_likelihood(x, torch.sigmoid(logits), "gaussian", 0.5)

    assert bernoulli.shape == gaussian.shape == (5, 3)
    single = pixel_log_likelihood(x, logits[2], "bernoulli")
    assert torch.allclose(bernoulli[2], single)
