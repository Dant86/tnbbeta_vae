"""Tests for tnbbeta_vae.models.conv_vae."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models import ConvTNBBetaSphericalVAE, ConvTNBBetaSphericalVAEConfig
from tnbbeta_vae.registry import build_model, list_registered_models
from tnbbeta_vae.training import Trainer


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


def test_kl_estimate_is_nonnegative_in_expectation() -> None:
    """A single-sample KL estimate can be negative; its expectation must not be.

    KL(q||p) >= 0 always (Gibbs' inequality), but log q(z) - log p(z) for
    one z~q is only an unbiased *estimator* of it -- individual draws can
    go negative. Average many independent draws to check the estimator's
    mean, not any single value (a per-draw ``kl >= 0`` assertion would be
    statistically wrong and would eventually fail on some seed).
    """
    torch.manual_seed(4)
    model = _small_model()
    x = torch.rand(6, 3, 32, 32)

    kls = torch.stack([model.training_step(x)["kl"] for _ in range(200)])

    assert kls.mean() > 0


def test_more_elbo_samples_reduces_loss_variance() -> None:
    """num_elbo_samples > 1 should lower the loss estimator's variance."""
    torch.manual_seed(5)
    single_sample_model = _small_model()
    multi_sample_config = ConvTNBBetaSphericalVAEConfig(
        latent_dim=4, hidden_channels=8, num_elbo_samples=16
    )
    multi_sample_model = ConvTNBBetaSphericalVAE(multi_sample_config)
    multi_sample_model.load_state_dict(single_sample_model.state_dict())
    x = torch.rand(6, 3, 32, 32)

    single_sample_losses = torch.stack(
        [single_sample_model.training_step(x)["loss"] for _ in range(100)]
    )
    multi_sample_losses = torch.stack(
        [multi_sample_model.training_step(x)["loss"] for _ in range(100)]
    )

    assert multi_sample_losses.std() < single_sample_losses.std()


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


def _small_model() -> ConvTNBBetaSphericalVAE:
    config = ConvTNBBetaSphericalVAEConfig(latent_dim=4, hidden_channels=8)
    return ConvTNBBetaSphericalVAE(config)
