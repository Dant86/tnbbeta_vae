"""Tests for the TNBBeta raw-p bias initialisation (``p_init_offset``)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import nn

from tnbbeta_vae.models import (
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
    MlpVAE,
    MlpVAEConfig,
    PairwiseMlpVAE,
    PairwiseMlpVAEConfig,
)
from tnbbeta_vae.models.heads import P_RAW_INIT, init_tnbbeta_p_bias
from tnbbeta_vae.models.semi_supervised import M1M2VAE, M1M2Config


def _mlp(offset: float | None = None, **kwargs: Any) -> MlpVAE:
    extra = {} if offset is None else {"p_init_offset": offset}
    torch.manual_seed(0)
    return MlpVAE(
        MlpVAEConfig(
            input_dim=12,
            hidden_dims=[16, 8],
            latent_dim=3,
            **extra,
            **kwargs,
        )
    )


def test_constant_is_log_three() -> None:
    assert math.log(3) == P_RAW_INIT
    assert torch.sigmoid(torch.tensor(P_RAW_INIT)).item() == pytest.approx(0.75)


def test_helper_sets_only_the_p_entry() -> None:
    torch.manual_seed(1)
    head = nn.Linear(5, 3 + 3)
    before_w, before_b = head.weight.clone(), head.bias.detach().clone()
    init_tnbbeta_p_bias(head, 3)
    assert head.bias[3].item() == pytest.approx(math.log(3))
    mask = torch.ones(6, dtype=torch.bool)
    mask[3] = False
    assert torch.equal(head.bias[mask], before_b[mask])
    assert torch.equal(head.weight, before_w)


def test_helper_index_with_fixed_mean_direction() -> None:
    head = nn.Linear(5, 2)
    init_tnbbeta_p_bias(head, 3, fixed_mean_direction=True, offset=1.5)
    assert head.bias[0].item() == pytest.approx(1.5)


def test_helper_zero_offset_is_a_noop() -> None:
    torch.manual_seed(1)
    head = nn.Linear(5, 6)
    before = head.bias.detach().clone()
    init_tnbbeta_p_bias(head, 3, offset=0.0)
    assert torch.equal(head.bias, before)


def test_mlp_default_sets_p_bias_and_posterior_p() -> None:
    model = _mlp()
    assert model.posterior_head.bias[3].item() == pytest.approx(math.log(3))


def test_mlp_zero_offset_reproduces_previous_init_exactly() -> None:
    old = _mlp(0.0)
    new = _mlp()
    for (name, a), (_, b) in zip(
        old.state_dict().items(), new.state_dict().items(), strict=True
    ):
        if name == "posterior_head.bias":
            mask = torch.ones_like(a, dtype=torch.bool)
            mask[3] = False
            assert torch.equal(a[mask], b[mask])
            assert a[3] != b[3]
        else:
            assert torch.equal(a, b), name


def test_mlp_state_dict_keys_and_shapes_unchanged() -> None:
    old, new = _mlp(0.0), _mlp()
    assert {k: v.shape for k, v in old.state_dict().items()} == {
        k: v.shape for k, v in new.state_dict().items()
    }


def test_mlp_fixed_epsilon_layout() -> None:
    model = _mlp(fixed_epsilon=1.0)
    assert model.posterior_head.bias.shape == (3 + 2,)
    assert model.posterior_head.bias[3].item() == pytest.approx(math.log(3))


@pytest.mark.parametrize("family", ["gaussian", "vmf", "power_spherical"])
def test_other_families_are_untouched(family: Any) -> None:
    torch.manual_seed(0)
    a = MlpVAE(
        MlpVAEConfig(family=family, input_dim=12, hidden_dims=[16, 8], latent_dim=3)
    )
    torch.manual_seed(0)
    b = MlpVAE(
        MlpVAEConfig(
            family=family,
            input_dim=12,
            hidden_dims=[16, 8],
            latent_dim=3,
            p_init_offset=0.0,
        )
    )
    for (name, x), (_, y) in zip(
        a.state_dict().items(), b.state_dict().items(), strict=True
    ):
        assert torch.equal(x, y), name


def test_pairwise_mlp_vae() -> None:
    kwargs: dict[str, Any] = {"input_dim": 6, "hidden_dims": [8], "latent_dim": 4}
    torch.manual_seed(0)
    old = PairwiseMlpVAE(PairwiseMlpVAEConfig(p_init_offset=0.0, **kwargs))
    torch.manual_seed(0)
    new = PairwiseMlpVAE(PairwiseMlpVAEConfig(**kwargs))
    assert new.posterior_head.bias[4].item() == pytest.approx(math.log(3))
    assert torch.equal(old.posterior_head.weight, new.posterior_head.weight)
    assert old.state_dict().keys() == new.state_dict().keys()
    torch.manual_seed(0)
    vmf = PairwiseMlpVAE(PairwiseMlpVAEConfig(family="vmf", **kwargs))
    torch.manual_seed(0)
    vmf0 = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(family="vmf", p_init_offset=0.0, **kwargs)
    )
    assert torch.equal(vmf.posterior_head.bias, vmf0.posterior_head.bias)


@pytest.mark.parametrize("fixed_mean_direction", [False, True])
def test_conv_vae(fixed_mean_direction: bool) -> None:
    config: dict[str, Any] = {
        "image_size": 8,
        "hidden_channels": 8,
        "latent_dim": 3,
        "fixed_mean_direction": fixed_mean_direction,
    }
    model = ConvTNBBetaSphericalVAE(ConvTNBBetaSphericalVAEConfig(**config))
    index = 0 if fixed_mean_direction else 3
    assert model.posterior_head.bias[index].item() == pytest.approx(math.log(3))
    old = ConvTNBBetaSphericalVAE(
        ConvTNBBetaSphericalVAEConfig(p_init_offset=0.0, **config)
    )
    assert old.state_dict().keys() == model.state_dict().keys()


def test_semi_supervised_only_tnbbeta_heads() -> None:
    config: dict[str, Any] = {
        "input_dim": 10,
        "z1_dim": 4,
        "z2_dim": 3,
        "hidden_dim": 8,
    }
    model: Any = M1M2VAE(
        M1M2Config(z1_family="tnbbeta", z2_family="gaussian", **config)
    )
    assert model.encoder1[-1].bias[4].item() == pytest.approx(math.log(3))
    assert model.decoder2[-1].bias[4].item() == pytest.approx(math.log(3))
    torch.manual_seed(0)
    gaussian = M1M2VAE(M1M2Config(z1_family="gaussian", z2_family="gaussian", **config))
    torch.manual_seed(0)
    zero = M1M2VAE(
        M1M2Config(
            z1_family="gaussian", z2_family="gaussian", p_init_offset=0.0, **config
        )
    )
    for (name, x), (_, y) in zip(
        gaussian.state_dict().items(), zero.state_dict().items(), strict=True
    ):
        assert torch.equal(x, y), name
