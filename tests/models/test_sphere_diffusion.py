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


def test_schedule_eps_target_derived_from_latent_dim(tmp_path: Path) -> None:
    """Review Focus: eps_target must be (latent_dim - 1) / 2, not a literal default.

    ``_train_tiny_vae`` uses ``latent_dim=4``, giving ``eps_target=1.5`` -- a
    non-trivial value, not the kind of number that would pass by accident.
    """
    model = _small_diffusion_model(tmp_path)

    assert model.latent_dim == 4
    assert model.schedule.eps_target == (model.latent_dim - 1) / 2
    assert model.schedule.eps_target == 1.5


def test_rejects_a_non_tnbbeta_spherical_vae_checkpoint(tmp_path: Path) -> None:
    """Review Focus: pointing at the wrong model kind must fail clearly."""
    run_name = _train_tiny_gaussian_vae(tmp_path)

    with pytest.raises(TypeError, match="ConvTNBBetaSphericalVAE"):
        SphereDiffusionPrior(SphereDiffusionPriorConfig(vae_run_name=run_name))


def test_missing_vae_checkpoint_raises_a_clear_error(tmp_path: Path) -> None:
    """Review Focus: a bad vae_run_name must not surface a bare torch.load error."""
    with pytest.raises(FileNotFoundError, match="vae_run_name") as exc_info:
        SphereDiffusionPrior(SphereDiffusionPriorConfig(vae_run_name="does_not_exist"))

    message = str(exc_info.value)
    assert "does_not_exist" in message
    assert "TNBBETA_CHECKPOINT_DIR" in message


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


def test_training_step_loss_decreases_on_overfit_batch(tmp_path: Path) -> None:
    """Spec's promised overfit convergence check: loss decreases.

    Trains the denoiser on a small, fixed batch of images for a few hundred
    steps and asserts the loss drops meaningfully -- the one test that would
    catch the denoiser becoming a no-op. Loss is averaged over several
    evaluations before and after training (rather than a single call) to
    smooth over the randomness in ``training_step``'s own ``t`` and
    ``z_0`` draws, so the comparison isn't flaky against the fixed seed.
    """
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=1e-2)
    images = torch.rand(8, 3, 32, 32)

    def _mean_loss(num_evals: int = 10) -> float:
        with torch.no_grad():
            return (
                sum(
                    model.training_step(images)["loss"].item() for _ in range(num_evals)
                )
                / num_evals
            )

    initial_loss = _mean_loss()
    for _ in range(300):
        optimizer.zero_grad()
        loss = model.training_step(images)["loss"]
        loss.backward()
        optimizer.step()
    final_loss = _mean_loss()

    assert final_loss < initial_loss * 0.9, (initial_loss, final_loss)


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


def test_forward_delegates_to_the_vae(tmp_path: Path) -> None:
    """Review Focus: eval scripts call model(batch)[0] expecting a reconstruction."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(4, 3, 32, 32)

    reconstruction, posterior, z = model(images)

    assert reconstruction.shape == images.shape
    assert z.shape == (4, model.latent_dim)


def test_posterior_and_prior_delegates_to_the_vae(tmp_path: Path) -> None:
    """Review Focus: most eval scripts call model.posterior_and_prior(batch)."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(4, 3, 32, 32)

    posterior, prior = model.posterior_and_prior(images)

    assert posterior.mean_direction.shape == (4, model.latent_dim)
    assert prior.mean_direction.shape[-1] == model.latent_dim


def test_log_likelihood_delegates_to_the_vae(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(4, 3, 32, 32)
    posterior, _ = model.posterior_and_prior(images)
    z = posterior.rsample()

    log_likelihood = model.log_likelihood(images, z)

    assert log_likelihood.shape == (4,)
    assert torch.isfinite(log_likelihood).all()


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


def test_trainer_runs_with_kl_warmup_epochs(tmp_path: Path) -> None:
    """Review Focus: --kl-warmup-epochs must not crash training_step with an
    unexpected kl_weight kwarg, even though this model has no KL term."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        model_name="tnbbeta_spherical_diffusion_prior",
        config=model.config,
        runs_dir=tmp_path / "runs3",
    )
    dataloader = [torch.rand(4, 3, 32, 32) for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2, kl_warmup_epochs=1)

    metrics_path = trainer.run_logger.run_dir / "metrics.jsonl"
    assert metrics_path.exists()
