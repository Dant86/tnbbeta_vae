"""Smoke test for apps.eval.sample_grid (tiny fake models, CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apps.eval import sample_grid
from apps.train import main as train_main
from tnbbeta_vae.data.cifar10 import Cifar10Images


class _FakeCifar10Base:
    def __init__(self, n: int = 16) -> None:
        self._images = torch.rand(n, 3, 32, 32)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._images[index], 0


def _fake_load_cifar10(*_args: object, **_kwargs: object) -> Cifar10Images:
    return Cifar10Images(_FakeCifar10Base())


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train_main, "load_cifar10", _fake_load_cifar10)


def _train(model: str, run_name: str, extra_sets: list[str] | None = None) -> None:
    args = [
        "--model", model, "--dataset", "cifar10", "--run-name", run_name,
        "--set", "latent_dim=4", "--set", "hidden_channels=8",
    ]  # fmt: skip
    for setting in extra_sets or []:
        args += ["--set", setting]
    args += [
        "--epochs", "1", "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ]  # fmt: skip
    train_main.main(args)


def test_writes_a_png_comparing_both_priors(tmp_path: Path) -> None:
    _train("conv_tnbbeta_spherical_vae", "vae_run")
    _train(
        "tnbbeta_spherical_diffusion_prior", "diffusion_run",
        extra_sets=["vae_run_name=vae_run", "num_reverse_steps=2"],
    )  # fmt: skip

    output = tmp_path / "grid.png"
    sample_grid.main(
        [
            "--vae-run",
            "vae_run",
            "--diffusion-run",
            "diffusion_run",
            "--num-samples",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]  # fmt: skip
    )

    assert output.exists() and output.stat().st_size > 0
