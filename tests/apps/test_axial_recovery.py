"""Tests for the axial-mixture data and the S^1 axial-recovery experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from apps.synthetic import axial_recovery
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig


def test_tnbbeta_diagnostics_m_mean_uses_latent_dim_minus_one_over_two() -> None:
    """Pins the bimodality-indicator formula directly, independent of training.

    ``m_mean`` must be ``epsilon_mean - (latent_dim - 1) / 2``, not off by a
    constant -- getting this wrong would silently invert the mechanistic
    success criterion (see the plan's Review Focus).
    """
    torch.manual_seed(0)
    x = torch.randn(16, 100)
    for latent_dim in (2, 4):
        model = MlpVAE(
            MlpVAEConfig(family=cast("Any", "tnbbeta"), latent_dim=latent_dim)
        )
        posterior, _ = model.posterior_and_prior(x)
        centre = axial_recovery._centre(model, x, "tnbbeta")

        diagnostics = axial_recovery._tnbbeta_diagnostics(model, x, centre)

        epsilon = posterior.epsilon  # pyright: ignore[reportAttributeAccessIssue]
        expected = epsilon.mean().item() - (latent_dim - 1) / 2
        assert diagnostics["m_mean"] == pytest.approx(expected)


def test_axial_angle_error_treats_phi_and_phi_plus_pi_as_equal() -> None:
    generator = torch.Generator().manual_seed(0)
    true = torch.rand(500, generator=generator) * 2 * math.pi - math.pi
    rotated = true + 1.3
    reflected = -true + 0.4
    flipped = true + math.pi
    good = [
        torch.stack([a.cos(), a.sin()], dim=-1)
        for a in (true, rotated, reflected, flipped)
    ]
    random = torch.randn(500, 2, generator=generator)

    for latent in good:
        assert axial_recovery.axial_angle_error(latent, true) < 1e-3
    assert axial_recovery.axial_angle_error(random, true) > 0.5


def test_experiment_writes_metrics_and_figure(tmp_path: Path) -> None:
    axial_recovery.main(
        [
            "--out-dir",
            str(tmp_path),
            "--epochs",
            "2",
            "--num-train",
            "256",
            "--num-test",
            "64",
            "--batch-size",
            "64",
            "--kl-warmup-epochs",
            "1",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "axial_recovery.json").read_text())
    assert set(results) == {"gaussian", "vmf", "power_spherical", "tnbbeta"}
    for metrics in results.values():
        for key in (
            "angle_error",
            "angle_error_sample",
            "reconstruction_angle_error",
            "prior_manifold_ratio",
            "test_ll",
            "test_kl",
        ):
            assert math.isfinite(metrics[key])
    assert {"p_mean", "p_std", "centre_axis_resultant", "m_mean"} <= set(
        results["tnbbeta"]
    )
    assert "p_mean" not in results["vmf"]
    assert "p_mean" not in results["power_spherical"]
    html = (tmp_path / "axial_recovery.html").read_text()
    assert "N-VAE" in html and "TNBBeta" in html and "Power Spherical" in html
