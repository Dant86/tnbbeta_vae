"""Tests for tnbbeta_vae.models.conv_vmf_vae."""

from __future__ import annotations

import torch

from tnbbeta_vae.distributions import VonMisesFisher
from tnbbeta_vae.models import ConvVonMisesFisherVAE, ConvVonMisesFisherVAEConfig
from tnbbeta_vae.registry import build_model, list_registered_models


def _small_model() -> ConvVonMisesFisherVAE:
    return ConvVonMisesFisherVAE(
        ConvVonMisesFisherVAEConfig(latent_dim=4, hidden_channels=8)
    )


def test_registered_and_buildable() -> None:
    assert "conv_vmf_vae" in list_registered_models()
    model = build_model("conv_vmf_vae", latent_dim=4, hidden_channels=8)

    assert isinstance(model, ConvVonMisesFisherVAE)


def test_forward_shapes_and_unit_norm_latents() -> None:
    torch.manual_seed(0)
    x = torch.rand(6, 3, 32, 32)

    reconstruction, posterior, z = _small_model()(x)

    assert reconstruction.shape == x.shape
    assert isinstance(posterior, VonMisesFisher)
    assert z.shape == (6, 4)
    assert torch.allclose(z.norm(dim=-1), torch.ones(6), atol=1e-4)


def test_training_step_has_nonnegative_analytic_kl_and_gradients() -> None:
    torch.manual_seed(0)
    model = _small_model()

    out = model.training_step(torch.rand(4, 3, 32, 32))
    out["loss"].backward()

    assert out["kl"].item() >= 0
    assert "posterior_kappa_mean" in out
    assert all(p.grad is not None for p in model.fc_var.parameters())


def test_generate_returns_valid_images() -> None:
    images = _small_model().generate(5)

    assert images.shape == (5, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 1
