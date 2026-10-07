"""Tests for tnbbeta_vae.models.sphere_diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tnbbeta_vae.models import (
    ConvGaussianVAE,
    ConvGaussianVAEConfig,
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
)
from tnbbeta_vae.models.sphere_diffusion import (
    SphereDiffusionPrior,
    SphereDiffusionPriorConfig,
)
from tnbbeta_vae.registry import build_model, list_registered_models
from tnbbeta_vae.training import Trainer


def _train_tiny_vae(tmp_path: Path, run_name: str = "vae_run") -> str:
    """Writes a real (tiny, 1-epoch) VAE checkpoint under tmp_path, returns its name."""
    model = ConvTNBBetaSphericalVAE(
        ConvTNBBetaSphericalVAEConfig(latent_dim=4, hidden_channels=8)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer, "conv_tnbbeta_spherical_vae", model.config,
        runs_dir=tmp_path / "runs",
    )  # fmt: skip
    trainer.fit(
        [torch.rand(4, 3, 32, 32) for _ in range(2)],
        num_epochs=1,
        checkpoint_dir=tmp_path / "checkpoints" / run_name,
    )
    return run_name


def _train_tiny_gaussian_vae(tmp_path: Path, run_name: str = "gauss_run") -> str:
    model = ConvGaussianVAE(ConvGaussianVAEConfig(latent_dim=4, hidden_channels=8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer, "conv_gaussian_vae", model.config, runs_dir=tmp_path / "runs"
    )
    trainer.fit(
        [torch.rand(4, 3, 32, 32) for _ in range(2)],
        num_epochs=1,
        checkpoint_dir=tmp_path / "checkpoints" / run_name,
    )
    return run_name


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    return tmp_path


def _small_diffusion_model(tmp_path: Path, **overrides: object) -> SphereDiffusionPrior:
    run_name = _train_tiny_vae(tmp_path)
    defaults: dict[str, object] = {
        "vae_run_name": run_name,
        "denoiser_hidden_dim": 16,
        "denoiser_depth": 2,
        "num_reverse_steps": 3,
    }
    config = SphereDiffusionPriorConfig(**(defaults | overrides))  # pyright: ignore[reportArgumentType]
    return SphereDiffusionPrior(config)


def test_registered_under_expected_name() -> None:
    assert "tnbbeta_spherical_diffusion_prior" in list_registered_models()


def test_build_via_registry_applies_overrides(tmp_path: Path) -> None:
    run_name = _train_tiny_vae(tmp_path)

    model = build_model(
        "tnbbeta_spherical_diffusion_prior",
        vae_run_name=run_name,
        denoiser_hidden_dim=16,
    )

    assert isinstance(model, SphereDiffusionPrior)
    assert model.config.denoiser_hidden_dim == 16


def test_rejects_a_non_tnbbeta_spherical_vae_checkpoint(tmp_path: Path) -> None:
    """Review Focus: pointing at the wrong model kind must fail clearly."""
    run_name = _train_tiny_gaussian_vae(tmp_path)

    with pytest.raises(TypeError, match="ConvTNBBetaSphericalVAE"):
        SphereDiffusionPrior(SphereDiffusionPriorConfig(vae_run_name=run_name))


def test_training_step_returns_finite_loss(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(5, 3, 32, 32)

    outputs = model.training_step(images)

    assert "loss" in outputs
    assert outputs["loss"].dim() == 0
    assert torch.isfinite(outputs["loss"])


def test_training_step_never_samples_t_below_t_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus: t must stay bounded away from 0, not Uniform(0, t_max)."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path, t_min=0.5, t_max=1.0)
    seen_t: list[torch.Tensor] = []
    original = torch.distributions.Uniform

    class _RecordingUniform(original):  # type: ignore[misc]
        def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            value = super().sample(*args, **kwargs)
            seen_t.append(value)
            return value

    monkeypatch.setattr(torch.distributions, "Uniform", _RecordingUniform)
    model.training_step(torch.rand(20, 3, 32, 32))

    assert seen_t, "training_step never drew a t -- test fixture is out of date"
    assert all((t >= 0.5).all() for t in seen_t)


def test_training_step_gradients_flow_only_to_the_denoiser(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(5, 3, 32, 32)

    outputs = model.training_step(images)
    outputs["loss"].backward()

    for name, param in model.denoiser.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_vae_parameters_receive_no_gradient(tmp_path: Path) -> None:
    """Review Focus: 'frozen' means .grad stays None after a real backward(), not just
    requires_grad=False."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(5, 3, 32, 32)

    outputs = model.training_step(images)
    outputs["loss"].backward()

    for param in model.vae.parameters():
        assert not param.requires_grad
        assert param.grad is None


def test_train_mode_does_not_unfreeze_the_vae(tmp_path: Path) -> None:
    model = _small_diffusion_model(tmp_path)

    model.train()

    assert model.denoiser.training
    assert not model.vae.training


def test_generate_returns_valid_images(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)

    images = model.generate(4)

    assert images.shape == (4, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert images.min() >= 0 and images.max() <= 1


def test_generate_handles_single_reverse_step(tmp_path: Path) -> None:
    """Review Focus: the reverse loop's boundary (K=1) must not produce nan."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path, num_reverse_steps=1)

    images = model.generate(3)

    assert torch.isfinite(images).all()


def _available_accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_available_accelerator() is None, reason="no accelerator available")
def test_training_step_and_generate_work_on_an_accelerator(tmp_path: Path) -> None:
    """Review Focus: t/steps tensors must follow the model's device, not just CPU."""
    torch.manual_seed(0)
    device = _available_accelerator()
    assert device is not None
    model = _small_diffusion_model(tmp_path).to(device)
    images = torch.rand(5, 3, 32, 32, device=device)

    outputs = model.training_step(images)
    assert torch.isfinite(outputs["loss"])

    generated = model.generate(3)
    assert generated.device.type == device
    assert torch.isfinite(generated).all()


def test_trainer_runs_end_to_end(tmp_path: Path) -> None:
    """Smoke test: the real model, the real Trainer, unstructured random batches."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        model_name="tnbbeta_spherical_diffusion_prior",
        config=model.config,
        runs_dir=tmp_path / "runs2",
    )
    dataloader = [torch.rand(4, 3, 32, 32) for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2)

    metrics_path = trainer.run_logger.run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert len(metrics_path.read_text().splitlines()) > 0
