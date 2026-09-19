"""Tests for tnbbeta_vae.models.conv_gaussian_vae."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.distributions import Independent

from tnbbeta_vae.data import gaussian_blob_batch
from tnbbeta_vae.models import ConvGaussianVAE, ConvGaussianVAEConfig
from tnbbeta_vae.registry import build_model, list_registered_models
from tnbbeta_vae.training import Trainer


def test_registered_under_expected_name() -> None:
    assert "conv_gaussian_vae" in list_registered_models()


def test_build_via_registry_applies_overrides() -> None:
    model = build_model("conv_gaussian_vae", latent_dim=4, hidden_channels=8)

    assert isinstance(model, ConvGaussianVAE)
    assert model.config.latent_dim == 4


def test_forward_shapes() -> None:
    torch.manual_seed(0)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    reconstruction, posterior, z = model(x)

    assert reconstruction.shape == x.shape
    assert isinstance(posterior, Independent)
    assert posterior.batch_shape == torch.Size([6])
    assert z.shape == (6, 4)


def test_training_step_returns_finite_loss_and_metrics() -> None:
    torch.manual_seed(1)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    outputs = model.training_step(x)

    expected_keys = {
        "loss",
        "log_likelihood",
        "kl",
        "posterior_sigma_mean",
        "posterior_sigma_min",
        "posterior_sigma_max",
        "posterior_mu_std_mean",
        "posterior_active_units",
    }
    assert set(outputs) == expected_keys
    for value in outputs.values():
        assert value.dim() == 0
        assert torch.isfinite(value)


def test_closed_form_kl_is_nonnegative_for_every_batch() -> None:
    """Exact KL can't go negative, unlike a single-sample MC estimate."""
    torch.manual_seed(2)
    model = _small_model()

    for _ in range(20):
        x = torch.rand(6, 3, 32, 32)
        assert model.training_step(x)["kl"] >= 0


def test_training_step_gradients_flow_to_every_parameter() -> None:
    torch.manual_seed(3)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    model.training_step(x)["loss"].backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_trainer_runs_end_to_end_on_gaussian_blob_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(4)
    model = _small_model()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        model_name="conv_gaussian_vae",
        config=model.config,
    )
    dataloader = [gaussian_blob_batch(batch_size=8, image_size=32)[0] for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2)

    assert (trainer.run_logger.run_dir / "metrics.jsonl").exists()


def _small_model() -> ConvGaussianVAE:
    return ConvGaussianVAE(ConvGaussianVAEConfig(latent_dim=4, hidden_channels=8))


def test_generate_returns_valid_images() -> None:
    images = _small_model().generate(5)

    assert images.shape == (5, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 1
