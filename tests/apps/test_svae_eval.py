"""Tests for the S-VAE metrics CLI, the table aggregator and the sweep script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from apps.eval import svae_metrics, svae_table
from apps.train import main as train_main
from tnbbeta_vae.data.mnist import MnistImages

_REPO = Path(__file__).resolve().parents[2]


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
    monkeypatch.setattr(svae_metrics, "load_mnist", _fake_mnist)


@pytest.mark.parametrize(
    "model", ["conv_gaussian_vae", "conv_vmf_vae", "conv_tnbbeta_spherical_vae"]
)
def test_metrics_cli_writes_the_four_table_columns(model: str, tmp_path: Path) -> None:
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

    svae_metrics.main(
        [
            "--run-name",
            "m",
            "--num-samples",
            "6",
            "--sample-chunk",
            "4",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    result = json.loads(
        (tmp_path / "ckpt" / "m" / "svae_metrics_final_test.json").read_text()
    )
    assert result["model_name"] == model
    assert result["latent_dim"] == 3
    assert result["kl"] >= -1.0
    assert result["elbo"] == pytest.approx(result["re"] - result["kl"], abs=1e-3)
    assert all(result[key] < 0 for key in ("ll", "re"))


def _write_runs(
    tmp_path: Path, model: str, dim: int, lls: list[float], kl: float = 5.0
) -> None:
    for seed, ll in enumerate(lls):
        run_dir = tmp_path / "ckpt" / f"mnist_{model}_d{dim}_seed{seed}"
        run_dir.mkdir(parents=True)
        record = {"ll": ll, "elbo": ll - 1, "re": ll - 5, "kl": kl}
        (run_dir / "svae_metrics_final_test.json").write_text(json.dumps(record))


def test_table_bolds_only_significant_winners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_runs(tmp_path, "gauss", 2, [-100.0, -100.2, -99.8, -100.1])
    _write_runs(tmp_path, "vmf", 2, [-90.0, -90.1, -89.9, -90.2])
    _write_runs(tmp_path, "gauss", 10, [-80.0, -80.5, -79.5, -80.2])
    _write_runs(tmp_path, "vmf", 10, [-80.1, -80.4, -79.6, -80.3])

    svae_table.main(
        ["--models", "gauss", "vmf", "--dims", "2", "10", "--seeds", "0", "1", "2", "3"]
    )

    rows = capsys.readouterr().out.strip().splitlines()
    assert "N-VAE LL" in rows[0] and "S-VAE (vMF) KL" in rows[0]
    dim2, dim10 = rows[2], rows[3]
    assert "**-90.05 ± 0.13**" in dim2
    assert "**-100.03" not in dim2 and dim2.count("**") == 6
    assert "**" not in dim10
    assert "5.00 ± 0.00" in dim2


def test_table_skips_missing_runs_and_shows_dashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_runs(tmp_path, "gauss", 2, [-100.0, -101.0])

    svae_table.main(
        ["--models", "gauss", "tnb", "--dims", "2", "--seeds", "0", "1", "2"]
    )

    captured = capsys.readouterr()
    assert "| 2 | -100.50 ± 0.71" in captured.out
    assert "| - | - | - | - |" in captured.out
    assert "4 run(s) had no metrics file" in captured.err


@pytest.mark.parametrize(
    ("task", "run_name", "model", "dim", "seed"),
    [
        (0, "mnist_gauss_d2_seed0", "conv_gaussian_vae", "2", "0"),
        (4, "mnist_gauss_d2_seed4", "conv_gaussian_vae", "2", "4"),
        (5, "mnist_gauss_d5_seed0", "conv_gaussian_vae", "5", "0"),
        (30, "mnist_vmf_d2_seed0", "conv_vmf_vae", "2", "0"),
        (89, "mnist_tnb_d128_seed4", "conv_tnbbeta_spherical_vae", "128", "4"),
    ],
)
def test_sweep_script_maps_array_index_to_model_dim_seed(
    tmp_path: Path, task: int, run_name: str, model: str, dim: str, seed: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv_calls.txt"
    stub = bin_dir / "uv"
    stub.write_text(f'#!/bin/bash\necho "$@" >> "{log}"\n')
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(_REPO),
        "SLURM_ARRAY_TASK_ID": str(task),
    }

    result = subprocess.run(
        ["bash", str(_REPO / "scripts" / "slurm" / "mnist_sweep.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    train_call, eval_call = log.read_text().splitlines()
    assert f"--model {model} --run-name {run_name}" in train_call
    assert f"--set latent_dim={dim}" in train_call
    assert f"--seed {seed}" in train_call
    assert "--patience 50 --kl-warmup-epochs 100" in train_call
    assert f"apps.eval.svae_metrics --run-name {run_name}" in eval_call
