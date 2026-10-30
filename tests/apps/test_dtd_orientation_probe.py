"""Tests for apps/eval/dtd_orientation_probe.py (torchvision mocked; no network)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from apps.eval import dtd_orientation_probe
from apps.train import main as train_main
from tnbbeta_vae.data.dtd import DtdImages

_SIZE = 32


class _FakeDtdBase:
    """A tiny stand-in for torchvision's ``DTD``, with striped/noisy images."""

    classes = ["banded", "bumpy"]
    class_to_idx = {"banded": 0, "bumpy": 1}

    def __init__(self, generator: torch.Generator) -> None:
        ys, xs = torch.meshgrid(
            torch.arange(_SIZE, dtype=torch.float32),
            torch.arange(_SIZE, dtype=torch.float32),
            indexing="ij",
        )
        stripes = (torch.cos(2 * math.pi * 6 / _SIZE * xs) + 1) / 2
        stripes = stripes.unsqueeze(0).expand(3, -1, -1)
        noise = torch.rand(3, _SIZE, _SIZE, generator=generator)
        self._images = [stripes] * 6 + [noise] * 6
        self._labels_list = [0] * 6 + [1] * 6

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._images[index], self._labels_list[index]


def _fake_dtd(*_args: object, split: str, **_kwargs: object) -> DtdImages:
    generator = torch.Generator().manual_seed(0 if split == "train" else 1)
    return DtdImages(_FakeDtdBase(generator), categories=("banded", "bumpy"))


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train_main, "load_dtd", _fake_dtd)
    monkeypatch.setattr(dtd_orientation_probe, "load_dtd", _fake_dtd)


def _train(model: str, run_name: str) -> None:
    train_main.main(
        [
            "--model",
            model,
            "--dataset",
            "dtd",
            "--set",
            "latent_dim=4",
            "--set",
            "hidden_channels=8",
            "--set",
            f"image_size={_SIZE}",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--run-name",
            run_name,
        ]  # fmt: skip
    )


def test_writes_table1_metrics_and_tnbbeta_pqm_for_a_tnbbeta_run(
    tmp_path: Path,
) -> None:
    _train("conv_tnbbeta_spherical_vae", "d")

    dtd_orientation_probe.main(
        [
            "--run-name",
            "d",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--num-samples",
            "4",
            "--batch-size",
            "4",
        ]  # fmt: skip
    )

    run_dir = tmp_path / "ckpt" / "d"
    result = json.loads((run_dir / "dtd_orientation_probe_final.json").read_text())
    for key in ("test_ll", "test_elbo", "test_kl", "mean_coherence"):
        assert key in result
    assert result["model_name"] == "conv_tnbbeta_spherical_vae"
    assert result["num_images"] == 12
    tnbbeta = result["tnbbeta"]
    for key in ("p_mean", "p_std", "q_mean", "q_std", "m_mean", "m_std"):
        assert key in tnbbeta
    assert 0.0 <= tnbbeta["p_mean"] <= 1.0
    assert 0.0 <= tnbbeta["q_mean"] <= 1.0

    html = (run_dir / "dtd_orientation_probe_final.html").read_text()
    assert "structure-tensor coherence" in html


def test_writes_table1_metrics_without_tnbbeta_block_for_a_gaussian_run(
    tmp_path: Path,
) -> None:
    _train("conv_gaussian_vae", "d")

    dtd_orientation_probe.main(
        [
            "--run-name",
            "d",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--num-samples",
            "4",
            "--batch-size",
            "4",
        ]  # fmt: skip
    )

    run_dir = tmp_path / "ckpt" / "d"
    result = json.loads((run_dir / "dtd_orientation_probe_final.json").read_text())
    assert result["model_name"] == "conv_gaussian_vae"
    for key in ("test_ll", "test_elbo", "test_kl", "mean_coherence"):
        assert key in result
    assert "tnbbeta" not in result

    html = (run_dir / "dtd_orientation_probe_final.html").read_text()
    assert "no TNBBeta p" in html
