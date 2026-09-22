"""Tests for apps/eval/svae_latitude.py (the TNBBeta p/q "cap vs ring" diagnostic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.eval import svae_latitude
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
    monkeypatch.setattr(svae_latitude, "load_mnist", _fake_mnist)


def _train(model: str, run_name: str) -> None:
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


def test_writes_json_and_html_for_a_tnbbeta_run(tmp_path: Path) -> None:
    _train("conv_tnbbeta_spherical_vae", "m")

    svae_latitude.main(["--run-name", "m", "--num-workers", "0", "--device", "cpu"])

    run_dir = tmp_path / "ckpt" / "m"
    result = json.loads((run_dir / "latitude_final.json").read_text())
    assert 0.0 <= result["p_mean"] <= 1.0
    assert 0.0 <= result["q_mean"] <= 1.0
    assert set(result["per_class"]) == {str(k) for k in range(10)}
    for stats in result["per_class"].values():
        assert 0.0 <= stats["p_mean"] <= 1.0
    html = (run_dir / "latitude_final.html").read_text()
    assert "p (latitude) vs q" in html


def test_skips_gracefully_for_a_non_tnbbeta_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _train("conv_gaussian_vae", "m")

    svae_latitude.main(["--run-name", "m", "--num-workers", "0", "--device", "cpu"])

    run_dir = tmp_path / "ckpt" / "m"
    assert not (run_dir / "latitude_final.json").exists()
    assert not (run_dir / "latitude_final.html").exists()
    assert "skipping" in capsys.readouterr().out


def test_scatter_figure_has_one_trace_per_present_class_and_downsamples() -> None:
    labels = np.arange(200) % 10
    p = np.random.default_rng(0).uniform(size=200)
    q = np.random.default_rng(1).uniform(size=200)

    figure = svae_latitude._scatter_figure(p, q, labels, "run", max_points=50, seed=0)

    assert len(figure.data) == 10  # pyright: ignore[reportArgumentType]
    total = sum(len(trace.x) for trace in figure.data)  # pyright: ignore[reportAttributeAccessIssue]
    assert total == 50
