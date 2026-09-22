"""Tests for apps/eval/confidence_probe.py and its table support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from apps.eval import confidence_probe, svae_table
from apps.train import main as train_main
from tnbbeta_vae.data.mnist import MnistImages


def _fake_mnist(*_args: object, split: str, **_kwargs: object) -> MnistImages:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(24, 1, 28, 28, generator=generator)
    return MnistImages(images, torch.arange(24) % 10, dynamic=split == "train")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train_main, "load_mnist", _fake_mnist)
    monkeypatch.setattr(confidence_probe, "load_mnist", _fake_mnist)


@pytest.mark.parametrize(
    ("model", "confidence_name"),
    [
        ("conv_gaussian_vae", "mean_std"),
        ("conv_vmf_vae", "kappa"),
        ("conv_tnbbeta_spherical_vae", "p"),
    ],
)
def test_writes_accuracy_from_the_confidence_scalar_alone(
    model: str, confidence_name: str, tmp_path: Path
) -> None:
    train_main.main(
        [
            "--model",
            model,
            "--dataset",
            "mnist",
            "--set",
            "latent_dim=3",
            "--set",
            "hidden_channels=8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--run-name",
            "m",
        ]  # fmt: skip
    )

    confidence_probe.main(
        [
            "--run-name",
            "m",
            "--label-counts",
            "4",
            "8",
            "--num-subsets",
            "3",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    result = json.loads(
        (tmp_path / "ckpt" / "m" / "confidence_probe_final.json").read_text()
    )
    assert result["confidence"] == confidence_name
    for count in (4, 8):
        assert 0.0 <= result[f"acc_{count}"] <= 1.0
        assert result[f"acc_{count}_std"] >= 0.0


def test_confidence_helper_shapes_and_picks_the_right_scalar() -> None:
    from tnbbeta_vae.models import (
        ConvGaussianVAE,
        ConvGaussianVAEConfig,
        ConvTNBBetaSphericalVAE,
        ConvTNBBetaSphericalVAEConfig,
        ConvVonMisesFisherVAE,
        ConvVonMisesFisherVAEConfig,
    )

    common: dict[str, Any] = {
        "image_channels": 1, "image_size": 28, "hidden_channels": 8, "latent_dim": 4,
    }  # fmt: skip
    x = torch.rand(5, 1, 28, 28)
    models: list[tuple[Any, str]] = [
        (ConvGaussianVAE(ConvGaussianVAEConfig(**common)), "conv_gaussian_vae"),
        (ConvVonMisesFisherVAE(ConvVonMisesFisherVAEConfig(**common)), "conv_vmf_vae"),
        (
            ConvTNBBetaSphericalVAE(ConvTNBBetaSphericalVAEConfig(**common)),
            "conv_tnbbeta_spherical_vae",
        ),
    ]
    for model, kind in models:
        posterior, _ = model.posterior_and_prior(x)
        confidence = confidence_probe._confidence(posterior, kind)
        assert confidence.shape == (5, 1)


def test_confidence_table_reads_the_confidence_probe_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for model, accuracy in (("gauss", 0.12), ("tnb", 0.45)):
        for seed in range(3):
            run_dir = tmp_path / "ckpt" / f"mnist_{model}_d5_seed{seed}"
            run_dir.mkdir(parents=True)
            record = {f"acc_{n}": accuracy for n in (100, 600, 1000)}
            (run_dir / "confidence_probe_final.json").write_text(json.dumps(record))

    svae_table.main(
        [
            "--kind",
            "confidence",
            "--models",
            "gauss",
            "tnb",
            "--dims",
            "5",
            "--seeds",
            "0",
            "1",
            "2",
        ]  # fmt: skip
    )

    rows = capsys.readouterr().out.strip().splitlines()
    assert "TNBBeta N=1000" in rows[0]
    assert "**0.45" in rows[2]
