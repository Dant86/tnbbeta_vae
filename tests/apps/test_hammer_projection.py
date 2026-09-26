"""Smoke tests for apps.eval.hammer_projection (tiny fake MNIST, CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apps.eval import hammer_projection
from apps.train import main as train_main
from tnbbeta_vae.data.mnist import MnistImages


def _fake_mnist(*_args: object, split: str, **_kwargs: object) -> MnistImages:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(16, 1, 28, 28, generator=generator)
    return MnistImages(images, torch.arange(16) % 10, dynamic=split == "train")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train_main, "load_mnist", _fake_mnist)
    monkeypatch.setattr(hammer_projection, "load_mnist", _fake_mnist)


def _train(model: str, run_name: str, latent_dim: int = 3) -> None:
    train_main.main(
        [
            "--model",
            model,
            "--dataset",
            "mnist",
            "--run-name",
            run_name,
            "--set",
            f"latent_dim={latent_dim}",
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
        ]  # fmt: skip
    )


def test_plots_both_panels_and_writes_a_png(tmp_path: Path) -> None:
    _train("conv_vmf_vae", "vmf_run")
    _train("conv_tnbbeta_spherical_vae", "tnb_run")

    output = tmp_path / "hammer.png"
    hammer_projection.main(
        [
            "--vmf-run",
            "vmf_run",
            "--tnb-run",
            "tnb_run",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]  # fmt: skip
    )

    assert output.exists() and output.stat().st_size > 0


def test_rejects_a_non_spherical_model(tmp_path: Path) -> None:
    _train("conv_gaussian_vae", "gauss_run")
    _train("conv_tnbbeta_spherical_vae", "tnb_run")

    with pytest.raises(ValueError, match="no spherical latent"):
        hammer_projection.main(
            [
                "--vmf-run",
                "gauss_run",
                "--tnb-run",
                "tnb_run",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--output",
                str(tmp_path / "hammer.png"),
            ]  # fmt: skip
        )


def test_rejects_a_run_not_trained_on_s2(tmp_path: Path) -> None:
    _train("conv_vmf_vae", "vmf_run", latent_dim=6)
    _train("conv_tnbbeta_spherical_vae", "tnb_run", latent_dim=3)

    with pytest.raises(ValueError, match="not 3"):
        hammer_projection.main(
            [
                "--vmf-run",
                "vmf_run",
                "--tnb-run",
                "tnb_run",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--output",
                str(tmp_path / "hammer.png"),
            ]  # fmt: skip
        )
