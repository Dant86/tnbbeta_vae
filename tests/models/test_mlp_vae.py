"""Tests for tnbbeta_vae.models.mlp_vae."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.losses import importance_weighted_metrics
from tnbbeta_vae.registry import build_model, list_registered_models

_FAMILIES = ["gaussian", "vmf", "tnbbeta", "power_spherical"]


def _model(family: Any, latent_dim: int = 3) -> MlpVAE:
    return MlpVAE(
        MlpVAEConfig(
            family=family, latent_dim=latent_dim, input_dim=12, hidden_dims=[16, 8]
        )
    )


def test_registered_and_buildable() -> None:
    assert "mlp_vae" in list_registered_models()
    model = build_model("mlp_vae", family="vmf", input_dim=12, latent_dim=3)

    assert isinstance(model, MlpVAE)


@pytest.mark.parametrize("family", _FAMILIES)
def test_forward_step_and_generation(family: str) -> None:
    torch.manual_seed(0)
    model = _model(family)
    x = torch.randn(5, 12)

    reconstruction, _, z = model(x)
    out = model.training_step(x)
    out["loss"].backward()

    assert reconstruction.shape == x.shape and z.shape == (5, 3)
    assert torch.isfinite(out["loss"])
    assert all(p.grad is not None for p in model.posterior_head.parameters())
    assert model.generate(4).shape == (4, 12)
    if family != "gaussian":
        assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-4)


@pytest.mark.parametrize("family", _FAMILIES)
def test_kl_weight_zero_leaves_only_the_likelihood(family: str) -> None:
    x = torch.randn(4, 12)
    model = _model(family)

    torch.manual_seed(0)
    full = model.training_step(x)
    torch.manual_seed(0)
    likelihood_only = model.training_step(x, kl_weight=0.0)

    assert torch.allclose(full["loss"], -(full["log_likelihood"] - full["kl"]))
    assert torch.allclose(likelihood_only["loss"], -likelihood_only["log_likelihood"])


@pytest.mark.parametrize("family", _FAMILIES)
def test_supports_the_importance_weighted_metrics(family: str) -> None:
    torch.manual_seed(0)
    metrics = importance_weighted_metrics(
        _model(family, latent_dim=2), torch.randn(4, 12), num_samples=10, chunk_size=4
    )

    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert metrics["ll"].shape == (4,)


def _tnbbeta_model(**overrides: object) -> MlpVAE:
    return MlpVAE(
        MlpVAEConfig(
            family="tnbbeta",
            latent_dim=4,
            input_dim=12,
            hidden_dims=[16, 8],
            **overrides,  # pyright: ignore[reportArgumentType]
        )
    )


def test_fixed_epsilon_shrinks_the_posterior_head_and_holds_epsilon_constant() -> None:
    torch.manual_seed(0)
    model = _tnbbeta_model(fixed_epsilon=0.5)
    default_model = _tnbbeta_model()

    assert (
        model.posterior_head.out_features
        == default_model.posterior_head.out_features - 1
    )
    x = torch.randn(5, 12)
    posterior = model._encode(x)

    assert torch.equal(posterior.epsilon, torch.full((5,), 0.5))  # pyright: ignore[reportAttributeAccessIssue]
    assert not posterior.epsilon.requires_grad  # pyright: ignore[reportAttributeAccessIssue]


def test_fixed_epsilon_trains_with_finite_gradients_everywhere() -> None:
    torch.manual_seed(0)
    model = _tnbbeta_model(fixed_epsilon=1.0)
    x = torch.randn(6, 12)

    out = model.training_step(x)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert all(p.grad is not None for p in model.posterior_head.parameters())


def test_default_config_keeps_epsilon_learned() -> None:
    torch.manual_seed(0)
    model = _tnbbeta_model()
    x = torch.randn(5, 12)

    posterior = model._encode(x)

    assert model.config.fixed_epsilon is None
    assert len(posterior.epsilon.unique()) > 1  # pyright: ignore[reportAttributeAccessIssue]


def test_fixed_epsilon_is_ignored_for_non_tnbbeta_families() -> None:
    """``fixed_epsilon`` only matters for ``tnbbeta``; elsewhere it's a no-op."""
    torch.manual_seed(0)
    plain_model = _model("vmf", latent_dim=4)
    model_with_fixed_epsilon = MlpVAE(
        MlpVAEConfig(
            family="vmf",
            latent_dim=4,
            input_dim=12,
            hidden_dims=[16, 8],
            fixed_epsilon=0.5,
        )
    )
    x = torch.randn(5, 12)

    assert (
        model_with_fixed_epsilon.posterior_head.out_features
        == plain_model.posterior_head.out_features
    )
    out = model_with_fixed_epsilon.training_step(x)
    assert torch.isfinite(out["loss"])
