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


def _train(model: str, run_name: str, tmp_path: Path) -> Path:
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
            run_name,
        ]  # fmt: skip
    )
    return tmp_path / "ckpt" / run_name


def test_param_epsilon_writes_a_differently_named_file_for_tnbbeta(
    tmp_path: Path,
) -> None:
    run_dir = _train("conv_tnbbeta_spherical_vae", "m", tmp_path)

    confidence_probe.main(
        [
            "--run-name",
            "m",
            "--param",
            "epsilon",
            "--label-counts",
            "4",
            "--num-subsets",
            "3",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    assert not (run_dir / "confidence_probe_final.json").exists()
    result = json.loads((run_dir / "confidence_probe_epsilon_final.json").read_text())
    assert result["confidence"] == "epsilon"
    assert 0.0 <= result["acc_4"] <= 1.0


@pytest.mark.parametrize("model", ["conv_gaussian_vae", "conv_vmf_vae"])
def test_param_epsilon_is_a_no_op_for_non_tnbbeta_runs(
    model: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _train(model, "m", tmp_path)

    confidence_probe.main(
        [
            "--run-name",
            "m",
            "--param",
            "epsilon",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    assert not (run_dir / "confidence_probe_epsilon_final.json").exists()
    assert "skipping" in capsys.readouterr().out


def test_default_param_still_writes_the_original_filename(tmp_path: Path) -> None:
    run_dir = _train("conv_tnbbeta_spherical_vae", "m", tmp_path)

    confidence_probe.main(
        [
            "--run-name",
            "m",
            "--label-counts",
            "4",
            "--num-subsets",
            "3",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    result = json.loads((run_dir / "confidence_probe_final.json").read_text())
    assert result["confidence"] == "p"


def test_confidence_extractors_shape_and_pick_the_right_scalar() -> None:
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
        param = confidence_probe._DEFAULT_PARAM[kind]
        confidence = confidence_probe._EXTRACTORS[(kind, param)](posterior)
        assert confidence.shape == (5, 1)

    tnb_model = next(m for m, k in models if k == "conv_tnbbeta_spherical_vae")
    posterior, _ = tnb_model.posterior_and_prior(x)
    epsilon = confidence_probe._EXTRACTORS[("conv_tnbbeta_spherical_vae", "epsilon")](
        posterior
    )
    assert epsilon.shape == (5, 1)


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


def test_confidence_epsilon_table_reads_the_epsilon_probe_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for seed in range(3):
        run_dir = tmp_path / "ckpt" / f"mnist_tnb_d5_seed{seed}"
        run_dir.mkdir(parents=True)
        record = {f"acc_{n}": 0.5 for n in (100, 600, 1000)}
        (run_dir / "confidence_probe_epsilon_final.json").write_text(json.dumps(record))

    svae_table.main(
        [
            "--kind",
            "confidence_epsilon",
            "--models",
            "tnb",
            "--dims",
            "5",
            "--seeds",
            "0",
            "1",
            "2",
        ]
    )

    rows = capsys.readouterr().out.strip().splitlines()
    assert "TNBBeta N=100" in rows[0]
    assert "0.50 ± 0.00" in rows[2]
