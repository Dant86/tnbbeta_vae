"""Tests for apps.eval.svae_concentration and its apps.eval.svae_table dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.eval import svae_concentration, svae_table
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
    monkeypatch.setattr(svae_concentration, "load_mnist", _fake_mnist)


def _train(model: str, run_name: str) -> None:
    train_main.main(
        [
            "--model",
            model,
            "--dataset",
            "mnist",
            "--run-name",
            run_name,
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
        ]  # fmt: skip
    )


@pytest.mark.parametrize("model", ["conv_vmf_vae", "conv_tnbbeta_spherical_vae"])
def test_cli_writes_ring_check_and_concentration_fields(
    model: str, tmp_path: Path
) -> None:
    _train(model, "m")

    svae_concentration.main(
        ["--run-name", "m", "--num-workers", "0", "--device", "cpu"]
    )

    path = tmp_path / "ckpt" / "m" / "concentration_final.json"
    result = json.loads(path.read_text())
    for key in ("far_side_pct", "antipodal_pct", "r_bar", "equivalent_kappa"):
        assert key in result
    assert len(result["per_class"]) == 10
    assert 0.0 <= result["r_bar"] <= 1.0
    assert 0.0 <= result["far_side_pct"] <= 100.0


def test_is_a_no_op_for_non_spherical_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _train("conv_gaussian_vae", "m")

    svae_concentration.main(
        ["--run-name", "m", "--num-workers", "0", "--device", "cpu"]
    )

    assert not (tmp_path / "ckpt" / "m" / "concentration_final.json").exists()
    assert "no spherical latent" in capsys.readouterr().out


def test_class_stats_on_a_tight_cluster_is_near_a_perfect_cap() -> None:
    rng = np.random.default_rng(0)
    centre = np.array([1.0, 0.0, 0.0])
    noise = rng.normal(scale=0.01, size=(500, 3))
    vectors = centre + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    stats = svae_concentration._class_stats(vectors, dim=3)

    assert stats["r_bar"] > 0.99
    assert stats["far_side_pct"] == 0.0
    assert stats["antipodal_pct"] == 0.0
    assert stats["equivalent_kappa"] > 100.0


def test_class_stats_on_a_uniform_sphere_has_no_concentration() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20000, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    stats = svae_concentration._class_stats(vectors, dim=3)

    assert stats["r_bar"] < 0.05
    assert stats["far_side_pct"] == pytest.approx(50.0, abs=5.0)


def test_table_reads_the_concentration_files_and_never_bolds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for model, r_bar in (("vmfs", 0.95), ("tnbs", 0.93)):
        for seed in range(3):
            run_dir = tmp_path / "ckpt" / f"mnist_{model}_d2_seed{seed}"
            run_dir.mkdir(parents=True)
            record = {
                "far_side_pct": 0.1,
                "antipodal_pct": 0.0,
                "r_bar": r_bar,
                "equivalent_kappa": 40.0,
            }
            (run_dir / "concentration_final.json").write_text(json.dumps(record))

    svae_table.main(
        [
            "--kind",
            "concentration",
            "--models",
            "vmfs",
            "tnbs",
            "--dims",
            "2",
            "--seeds",
            "0",
            "1",
            "2",
        ]  # fmt: skip
    )

    rows = capsys.readouterr().out.strip().splitlines()
    assert "kappa_eq" in rows[0]
    assert "**" not in rows[2]
    assert "0.95" in rows[2] and "0.93" in rows[2]
