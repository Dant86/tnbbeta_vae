"""Tests for the optional TNBBeta epsilon_floor (default = old head)."""

from __future__ import annotations

import pytest
import torch
from torch.nn.functional import softplus

from tnbbeta_vae.models import (
    ConvTNBBetaSphericalVAEConfig,
    GraphVAEConfig,
    M1M2Config,
    MlpVAE,
    MlpVAEConfig,
    PairwiseMlpVAEConfig,
)
from tnbbeta_vae.models.heads import posterior_from_raw, tnbbeta_posterior

_CONFIGS = [
    ConvTNBBetaSphericalVAEConfig,
    GraphVAEConfig,
    M1M2Config,
    MlpVAEConfig,
    PairwiseMlpVAEConfig,
]


@pytest.mark.parametrize("config_cls", _CONFIGS)
def test_default_floor_is_the_historical_one(config_cls: type) -> None:
    assert config_cls().epsilon_floor == 1e-4


def test_default_head_reproduces_the_old_formula_exactly() -> None:
    raw = torch.randn(32, 6)
    raw[:, -1] = torch.linspace(-30.0, 5.0, 32)
    old_epsilon = softplus(raw[:, -1]) + 1e-4
    assert torch.equal(tnbbeta_posterior(raw, 3).epsilon, old_epsilon)
    assert torch.equal(
        tnbbeta_posterior(raw, 3, epsilon_floor=1e-4).epsilon, old_epsilon
    )
    via = posterior_from_raw("tnbbeta", raw, 3)
    assert torch.equal(via.epsilon, old_epsilon)  # pyright: ignore[reportAttributeAccessIssue]


def test_floor_is_additive_and_leaves_p_q_direction_alone() -> None:
    raw = torch.randn(8, 6)
    raw[:, -1] = -40.0
    base = tnbbeta_posterior(raw, 3)
    floored = tnbbeta_posterior(raw, 3, epsilon_floor=0.3)
    assert float(floored.epsilon.min()) >= 0.3
    assert torch.allclose(floored.epsilon - base.epsilon, torch.tensor(0.3 - 1e-4))
    assert torch.equal(floored.p, base.p)
    assert torch.equal(floored.q, base.q)
    assert torch.equal(floored.mean_direction, base.mean_direction)
    via = posterior_from_raw("tnbbeta", raw, 3, epsilon_floor=0.3)
    assert torch.equal(via.epsilon, floored.epsilon)  # pyright: ignore[reportAttributeAccessIssue]


def test_floor_is_ignored_with_fixed_epsilon() -> None:
    raw = torch.randn(4, 5)
    posterior = tnbbeta_posterior(raw, 3, fixed_epsilon=2.0, epsilon_floor=0.3)
    assert torch.equal(posterior.epsilon, torch.full((4,), 2.0))


def test_model_passes_the_floor_to_its_posterior() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 12)
    default = MlpVAE(MlpVAEConfig(family="tnbbeta", latent_dim=3, input_dim=12))
    floored = MlpVAE(
        MlpVAEConfig(family="tnbbeta", latent_dim=3, input_dim=12, epsilon_floor=0.3)
    )
    floored.load_state_dict(default.state_dict())  # no parameter changes
    floored_eps = floored.posterior_and_prior(x)[0].epsilon  # pyright: ignore[reportAttributeAccessIssue]
    default_eps = default.posterior_and_prior(x)[0].epsilon  # pyright: ignore[reportAttributeAccessIssue]
    assert (floored_eps >= 0.3).all()
    delta = floored_eps - default_eps
    assert torch.allclose(delta, torch.tensor(0.3 - 1e-4), atol=1e-6)
