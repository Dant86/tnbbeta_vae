"""Tests for the circle-mixture data and the S^1 recovery experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from apps.synthetic import circle_recovery
from tnbbeta_vae.data.circle_mixture import circle_mixture_data


def test_samples_are_reproducible_and_shaped() -> None:
    data = circle_mixture_data(ambient_dim=20, seed=1)

    first = data.sample(50, torch.Generator().manual_seed(3))
    second = data.sample(50, torch.Generator().manual_seed(3))

    x, angle, component = first
    assert x.shape == (50, 20)
    assert angle.shape == component.shape == (50,)
    assert set(component.tolist()) <= {0, 1, 2}
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_embedding_is_a_fixed_function_of_the_angle() -> None:
    data = circle_mixture_data(ambient_dim=20, seed=1)
    angle = torch.linspace(-3, 3, 7)

    assert torch.equal(data.embed(angle), data.embed(angle))
    assert not torch.equal(
        data.embed(angle), circle_mixture_data(20, seed=2).embed(angle)
    )
    assert torch.allclose(
        data.embed(torch.tensor([-math.pi])),
        data.embed(torch.tensor([math.pi])),
        atol=1e-5,
    )


def test_angle_error_ignores_rotation_and_reflection_but_flags_a_bad_latent() -> None:
    generator = torch.Generator().manual_seed(0)
    true = torch.rand(500, generator=generator) * 2 * math.pi - math.pi
    rotated = true + 1.3
    reflected = -true + 0.4
    good = [torch.stack([a.cos(), a.sin()], dim=-1) for a in (true, rotated, reflected)]
    random = torch.randn(500, 2, generator=generator)

    for latent in good:
        assert circle_recovery.angle_error(latent, true) < 1e-3
    assert circle_recovery.angle_error(random, true) > 1.0


def test_experiment_writes_metrics_and_figure(tmp_path: Path) -> None:
    circle_recovery.main(
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

    results = json.loads((tmp_path / "circle_recovery.json").read_text())
    assert set(results) == {"gaussian", "vmf", "tnbbeta"}
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
    assert {"p_mean", "p_std", "centre_axis_resultant"} <= set(results["tnbbeta"])
    assert "p_mean" not in results["vmf"]
    html = (tmp_path / "circle_recovery.html").read_text()
    assert "N-VAE" in html and "TNBBeta" in html
