"""Tests for tnbbeta_vae.models.conv_vae."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tnbbeta_vae.data import gaussian_blob_batch
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

    expected_keys = {
        "loss",
        "log_likelihood",
        "kl",
        "likelihood_scale",
        "posterior_p_mean",
        "posterior_p_min",
        "posterior_p_max",
        "posterior_q_mean",
        "posterior_q_min",
        "posterior_q_max",
        "posterior_epsilon_mean",
    }
    assert set(outputs) == expected_keys
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


def test_trainer_runs_end_to_end_on_random_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: the real model, the real Trainer, unstructured random batches."""
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


def test_trainer_runs_end_to_end_on_gaussian_blob_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test with the actual collapse-diagnostic synthetic dataset.

    Confirms the full pipeline (model, Trainer, RunLogger) works with
    tnbbeta_vae.data.gaussian_blob_batch end to end, and that the
    posterior-collapse diagnostics make it into the logged metrics --
    this is the setup meant for locally watching p/q/epsilon trends over
    a real (if small) run.
    """
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(6)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        model_name="conv_tnbbeta_spherical_vae",
        config=model.config,
    )
    dataloader = [gaussian_blob_batch(batch_size=8, image_size=32)[0] for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2)

    metrics_path = trainer.run_logger.run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    step_records = [r for r in records if "loss" in r]
    assert len(step_records) == 3 * 2
    assert all(math.isfinite(r["posterior_q_max"]) for r in step_records)
    assert all(math.isfinite(r["posterior_epsilon_mean"]) for r in step_records)


def _small_model() -> ConvTNBBetaSphericalVAE:
    config = ConvTNBBetaSphericalVAEConfig(latent_dim=4, hidden_channels=8)
    return ConvTNBBetaSphericalVAE(config)


def test_generate_returns_valid_images() -> None:
    images = _small_model().generate(5)

    assert images.shape == (5, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 1


def test_p_and_q_are_bounded_away_from_the_boundary() -> None:
    model = ConvTNBBetaSphericalVAE(
        ConvTNBBetaSphericalVAEConfig(latent_dim=4, hidden_channels=8)
    )
    with torch.no_grad():
        model.posterior_head.bias[-3:-1] = torch.tensor([50.0, -50.0])  # p -> 1, q -> 0

        posterior = model._encode(torch.rand(3, 3, 32, 32))

    assert torch.allclose(posterior.p, torch.full((3,), 1 - 1e-6), atol=1e-7)
    assert torch.allclose(posterior.q, torch.full((3,), 1e-6), atol=1e-7)


def _mnist_model(**overrides: object) -> ConvTNBBetaSphericalVAE:
    config = ConvTNBBetaSphericalVAEConfig(
        image_channels=1,
        image_size=28,
        hidden_channels=8,
        latent_dim=6,
        likelihood="bernoulli",
        **overrides,  # pyright: ignore[reportArgumentType]
    )
    return ConvTNBBetaSphericalVAE(config)


def test_fixed_epsilon_shrinks_the_posterior_head_and_holds_epsilon_constant() -> None:
    torch.manual_seed(0)
    model = _mnist_model(fixed_epsilon=0.5)
    default_model = _mnist_model()

    assert (
        model.posterior_head.out_features
        == default_model.posterior_head.out_features - 1
    )
    x = torch.bernoulli(torch.rand(5, 1, 28, 28))
    posterior = model._encode(x)

    assert torch.equal(posterior.epsilon, torch.full((5,), 0.5))
    assert not posterior.epsilon.requires_grad


def test_fixed_epsilon_trains_with_finite_gradients_everywhere() -> None:
    torch.manual_seed(0)
    model = _mnist_model(fixed_epsilon=1.0)
    x = torch.bernoulli(torch.rand(6, 1, 28, 28))

    out = model.training_step(x)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert all(p.grad is not None for p in model.posterior_head.parameters())


def test_default_config_keeps_epsilon_learned() -> None:
    torch.manual_seed(0)
    model = _mnist_model()
    x = torch.bernoulli(torch.rand(5, 1, 28, 28))

    posterior = model._encode(x)

    assert model.config.fixed_epsilon is None
    assert len(posterior.epsilon.unique()) > 1
