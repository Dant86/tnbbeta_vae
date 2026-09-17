"""Tests for tnbbeta_vae.models.conv_vae."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models import ConvTNBBetaSphericalVAE, ConvTNBBetaSphericalVAEConfig
from tnbbeta_vae.registry import build_model, list_registered_models
from tnbbeta_vae.training import Trainer


def _small_model() -> ConvTNBBetaSphericalVAE:
    config = ConvTNBBetaSphericalVAEConfig(latent_dim=4, hidden_channels=8)
    return ConvTNBBetaSphericalVAE(config)


def test_registered_under_expected_name() -> None:
    assert "conv_tnbbeta_spherical_vae" in list_registered_models()


def test_build_via_registry_applies_overrides() -> None:
    model = build_model("conv_tnbbeta_spherical_vae", latent_dim=4, hidden_channels=8)

    assert isinstance(model, ConvTNBBetaSphericalVAE)
    assert model.config.latent_dim == 4


def test_forward_shapes_and_unit_norm_latent() -> None:
    torch.manual_seed(0)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    reconstruction, posterior, z = model(x)

    assert reconstruction.shape == x.shape
    assert isinstance(posterior, TNBBetaSpherical)
    assert posterior.batch_shape == torch.Size([6])
    assert z.shape == (6, 4)
    assert torch.allclose(z.norm(dim=-1), torch.ones(6), atol=1e-4)


def test_training_step_returns_finite_loss_and_metrics() -> None:
    torch.manual_seed(1)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    outputs = model.training_step(x)

    assert set(outputs) == {"loss", "log_likelihood", "kl"}
    for value in outputs.values():
        assert value.dim() == 0
        assert torch.isfinite(value)
    assert outputs["kl"] >= 0  # KL Monte Carlo estimate should be non-negative here


def test_training_step_gradients_flow_to_every_parameter() -> None:
    torch.manual_seed(2)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    outputs = model.training_step(x)
    outputs["loss"].backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_trainer_runs_end_to_end_on_synthetic_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: the real model, the real Trainer, a tiny synthetic dataset."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(3)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        model_name="conv_tnbbeta_spherical_vae",
        config=model.config,
    )
    dataloader = [torch.rand(4, 3, 32, 32) for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2)

    metrics_path = trainer.run_logger.run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert len(metrics_path.read_text().splitlines()) > 0
