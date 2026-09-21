"""Tests for tnbbeta_vae.models.conv_vmf_vae."""

from __future__ import annotations

import math
from typing import Any

import pytest
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
    assert all(p.grad is not None for p in model.learned_scale.parameters())


def test_generate_returns_valid_images() -> None:
    images = _small_model().generate(5)

    assert images.shape == (5, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 1


def _kappas(**config: Any) -> torch.Tensor:
    torch.manual_seed(0)
    model = ConvVonMisesFisherVAE(
        ConvVonMisesFisherVAEConfig(
            image_channels=1, image_size=28, hidden_channels=8, latent_dim=6, **config
        )
    )
    with torch.no_grad():
        posterior = model._encode(torch.rand(32, 1, 28, 28))
    return posterior.scale.flatten()


def test_default_initialization_starts_at_a_low_concentration() -> None:
    assert _kappas().median() < 3.0


def test_initial_kappa_sets_where_every_posterior_starts() -> None:
    kappas = _kappas(initial_kappa=40.0)

    # The random weights add spread around the bias, so only check the middle.
    assert 30 < kappas.median() < 55
    assert (kappas > 10).all()


@pytest.mark.parametrize("bad", [1.0, 0.5, 2e6])
def test_initial_kappa_must_be_between_one_and_the_float32_cap(bad: float) -> None:
    with pytest.raises(ValueError, match=r"kappa must be in \(1, 1e\+06\)"):
        _kappas(initial_kappa=bad)


@pytest.mark.parametrize("kappa", [1.5, 12.0, 4000.0, 250_000.0])
def test_kappa_inverse_round_trips(kappa: float) -> None:
    from tnbbeta_vae.models.heads import vmf_kappa, vmf_kappa_inverse

    raw = torch.tensor([vmf_kappa_inverse(kappa)])

    assert vmf_kappa(raw).item() == pytest.approx(kappa, rel=1e-4)


def test_kappa_is_capped_where_float32_stops_being_meaningful() -> None:
    from tnbbeta_vae.models.heads import vmf_kappa

    kappa = vmf_kappa(torch.tensor([50.0, 1e9]))

    assert kappa.max().item() == pytest.approx(1e6)
    assert torch.isfinite(kappa).all()


def test_the_analytic_kl_is_accurate_up_to_the_cap() -> None:
    from torch.distributions import kl_divergence

    from tnbbeta_vae.distributions import HypersphericalUniform, VonMisesFisher
    from tnbbeta_vae.models.heads import vmf_kappa

    loc = torch.zeros(4, 2)
    loc[:, 0] = 1.0
    kappa = vmf_kappa(torch.tensor([[1e9]] * 4))  # clamped to the cap

    kl = kl_divergence(VonMisesFisher(loc, kappa), HypersphericalUniform(1))

    # d=2: KL = 0.5 ln(2 pi kappa / e) for large kappa.
    expected = 0.5 * math.log(2 * math.pi * 1e6 / math.e)
    assert kl.mean().item() == pytest.approx(expected, abs=0.05)


def test_a_training_step_from_a_large_initial_kappa_has_finite_gradients() -> None:
    torch.manual_seed(0)
    model = ConvVonMisesFisherVAE(
        ConvVonMisesFisherVAEConfig(
            image_channels=1,
            image_size=28,
            hidden_channels=8,
            latent_dim=3,
            likelihood="bernoulli",
            initial_kappa=200.0,
        )
    )

    out = model.training_step(torch.bernoulli(torch.rand(6, 1, 28, 28)))
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
