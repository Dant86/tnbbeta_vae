"""Tests for tnbbeta_vae.models.conv_vmf_vae."""

from __future__ import annotations

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


@pytest.mark.parametrize("parameterization", ["softplus", "exp"])
def test_initial_kappa_sets_where_every_posterior_starts(parameterization: str) -> None:
    kappas = _kappas(initial_kappa=40.0, kappa_parameterization=parameterization)

    # The random weights add spread around the bias (multiplicative for "exp"), so
    # only check that the middle is within a factor of two of the request.
    assert 20 < kappas.median() < 80
    assert (kappas > 10).all()


def test_initial_kappa_must_exceed_one() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        _kappas(initial_kappa=1.0)


@pytest.mark.parametrize("parameterization", ["softplus", "exp"])
@pytest.mark.parametrize("kappa", [1.5, 12.0, 4000.0])
def test_kappa_inverse_round_trips(parameterization: str, kappa: float) -> None:
    from tnbbeta_vae.models.heads import vmf_kappa, vmf_kappa_inverse

    param: Any = parameterization
    raw = torch.tensor([vmf_kappa_inverse(kappa, param)])

    assert vmf_kappa(raw, param).item() == pytest.approx(kappa, rel=1e-4)


def test_exp_parameterization_grows_multiplicatively_and_is_capped() -> None:
    from tnbbeta_vae.models.heads import vmf_kappa

    raw = torch.tensor([0.0, 5.0, 10.0, 100.0])

    exp = vmf_kappa(raw, "exp")
    softplus = vmf_kappa(raw, "softplus")

    assert exp[2] > 20_000 > softplus[2]
    assert torch.isfinite(exp).all() and exp[3] == exp[3].clamp(max=1e9)


def test_a_training_step_with_the_exp_parameterization_has_finite_gradients() -> None:
    torch.manual_seed(0)
    model = ConvVonMisesFisherVAE(
        ConvVonMisesFisherVAEConfig(
            image_channels=1,
            image_size=28,
            hidden_channels=8,
            latent_dim=3,
            likelihood="bernoulli",
            initial_kappa=200.0,
            kappa_parameterization="exp",
        )
    )

    out = model.training_step(torch.bernoulli(torch.rand(6, 1, 28, 28)))
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
