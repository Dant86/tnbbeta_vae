"""Tests for tnbbeta_vae.models.pairwise_mlp_vae."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tnbbeta_vae.data.cluster_mixture import cluster_mixture_data, positive_pairs
from tnbbeta_vae.models.heads import head_size
from tnbbeta_vae.models.pairwise_mlp_vae import (
    PairwiseBatch,
    PairwiseMlpVAE,
    PairwiseMlpVAEConfig,
)
from tnbbeta_vae.registry import list_registered_models

_FAMILIES = ["gaussian", "vmf", "tnbbeta", "power_spherical"]


def _batch(seed: int = 0, num_pairs: int = 50) -> PairwiseBatch:
    generator = torch.Generator().manual_seed(seed)
    features, labels = cluster_mixture_data(
        num_clusters=5, points_per_cluster=10, dim=8, generator=generator
    )
    pairs = positive_pairs(labels, num_pairs, generator=generator)
    return PairwiseBatch(features=features, positive_pairs=pairs)


def test_registered() -> None:
    assert "pairwise_mlp_vae" in list_registered_models()


def test_config_defaults() -> None:
    config = PairwiseMlpVAEConfig()

    assert config.family == "tnbbeta"
    assert config.fixed_temperature is None


@pytest.mark.parametrize("family", _FAMILIES)
def test_posterior_head_size_matches_heads_dispatch(family: Any) -> None:
    model = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(
            family=family, input_dim=8, hidden_dims=[16, 8], latent_dim=4
        )
    )

    assert model.posterior_head.out_features == head_size(family, 4)


def test_num_items_property() -> None:
    batch = _batch()

    assert batch.num_items == 50


def test_batch_to_moves_device() -> None:
    batch = _batch()

    moved = batch.to(torch.device("cpu"))

    assert moved.features.device == torch.device("cpu")


@pytest.mark.parametrize("family", _FAMILIES)
def test_training_step_produces_finite_loss_and_gradients(family: Any) -> None:
    torch.manual_seed(0)
    batch = _batch()
    model = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(
            family=family, input_dim=8, hidden_dims=[16, 8], latent_dim=4
        )
    )

    out = model.training_step(batch)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["link_loss"])
    assert torch.isfinite(out["kl"])
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert all(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.posterior_head.parameters())


@pytest.mark.parametrize("family", _FAMILIES)
def test_embeddings_shape_and_unit_norm_for_sphere_families(family: Any) -> None:
    batch = _batch()
    model = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(
            family=family, input_dim=8, hidden_dims=[16, 8], latent_dim=4
        )
    )

    embeddings = model.embeddings(batch)

    assert embeddings.shape == (50, 4)
    if family != "gaussian":
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(50), atol=1e-4)


def test_no_decoder_on_the_model() -> None:
    model = PairwiseMlpVAE(PairwiseMlpVAEConfig(input_dim=8, latent_dim=4))

    assert not hasattr(model, "decoder")


def test_temperature_is_learned_by_default_and_fixed_when_given() -> None:
    learned = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(family="vmf", input_dim=8, latent_dim=4)
    )
    fixed = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(
            family="vmf", input_dim=8, latent_dim=4, fixed_temperature=1.0
        )
    )

    assert learned.log_temperature.requires_grad
    assert learned.temperature().item() == pytest.approx(5.0)
    assert not fixed.log_temperature.requires_grad
    assert fixed.temperature().item() == 1.0


def test_fixed_temperature_ignored_for_gaussian() -> None:
    model = PairwiseMlpVAE(
        PairwiseMlpVAEConfig(
            family="gaussian", input_dim=8, latent_dim=4, fixed_temperature=1.0
        )
    )
    z = torch.randn(5, 4)
    pairs = torch.tensor([[0, 1], [2, 3]])
    inner = (z[pairs[0]] * z[pairs[1]]).sum(-1)

    from tnbbeta_vae.models.pairwise import pairwise_logits

    logits = pairwise_logits(z, pairs, model.config.family, model.temperature())

    assert torch.allclose(logits, inner)
