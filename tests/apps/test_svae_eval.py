"""Tests for the S-VAE metrics CLI, the table aggregator and the sweep script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from apps.eval import svae_knn, svae_metrics, svae_table
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
    monkeypatch.setattr(svae_knn, "load_mnist", _fake_mnist)


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
    ("task", "run_name", "model", "dim", "seed", "latent"),
    [
        (0, "mnist_gauss_d2_seed0", "conv_gaussian_vae", "2", "0", "2"),
        (4, "mnist_gauss_d2_seed4", "conv_gaussian_vae", "2", "4", "2"),
        (5, "mnist_gauss_d5_seed0", "conv_gaussian_vae", "5", "0", "5"),
        (30, "mnist_vmf_d2_seed0", "conv_vmf_vae", "2", "0", "2"),
        (89, "mnist_tnb_d128_seed4", "conv_tnbbeta_spherical_vae", "128", "4", "128"),
        (90, "mnist_vmfk_d2_seed0", "conv_vmf_vae", "2", "0", "2"),
        (114, "mnist_vmfk_d40_seed4", "conv_vmf_vae", "40", "4", "40"),
        (120, "mnist_vmfs_d2_seed0", "conv_vmf_vae", "2", "0", "3"),
        (144, "mnist_vmfs_d40_seed4", "conv_vmf_vae", "40", "4", "41"),
        (150, "mnist_tnbs_d2_seed0", "conv_tnbbeta_spherical_vae", "2", "0", "3"),
        (180, "mnist_vmfks_d2_seed0", "conv_vmf_vae", "2", "0", "3"),
        (204, "mnist_vmfks_d40_seed4", "conv_vmf_vae", "40", "4", "41"),
    ],
)
def test_sweep_script_maps_array_index_to_model_dim_seed(
    tmp_path: Path,
    task: int,
    run_name: str,
    model: str,
    dim: str,
    seed: str,
    latent: str,
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
    train_call, eval_call, knn_call = log.read_text().splitlines()
    assert f"--model {model} --run-name {run_name}" in train_call
    # The run name carries the paper's d; sphere models named "...s" get ambient d + 1.
    assert f"--set latent_dim={latent} " in train_call
    assert f"--seed {seed}" in train_call
    assert "--patience 50 --kl-warmup-epochs 100" in train_call
    # Only the corrected vMF variants start at kappa = their ambient dimension.
    corrected = run_name.startswith(("mnist_vmfk_", "mnist_vmfks_"))
    assert (f"--set initial_kappa={latent}" in train_call) == corrected
    assert f"apps.eval.svae_metrics --run-name {run_name}" in eval_call
    assert f"apps.eval.svae_knn --run-name {run_name}" in knn_call


def test_knn_accuracy_recovers_clusters_in_both_geometries() -> None:
    torch.manual_seed(0)
    centres = torch.tensor([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]])
    labels = torch.arange(3).repeat_interleave(20)
    points = centres[labels] + 0.3 * torch.randn(60, 2)
    unit = torch.nn.functional.normalize(points, dim=-1)

    euclidean = svae_knn.knn_accuracy(
        points[::2], labels[::2], points[1::2], labels[1::2], 5, "euclidean"
    )
    geodesic = svae_knn.knn_accuracy(
        unit[::2], labels[::2], unit[1::2], labels[1::2], 5, "geodesic"
    )

    assert euclidean == pytest.approx(1.0)
    assert geodesic == pytest.approx(1.0)
    swapped = svae_knn.knn_accuracy(
        points[::2], labels[::2].roll(10), points[1::2], labels[1::2], 5, "euclidean"
    )
    assert swapped < 0.5


def test_knn_accuracy_caps_k_at_the_number_of_labelled_points() -> None:
    features = torch.tensor([[0.0], [10.0]])
    labels = torch.tensor([0, 1])

    accuracy = svae_knn.knn_accuracy(
        features, labels, torch.tensor([[1.0]]), torch.tensor([0]), 5, "euclidean"
    )

    assert accuracy in (0.0, 1.0)


@pytest.mark.parametrize(
    "model", ["conv_gaussian_vae", "conv_vmf_vae", "conv_tnbbeta_spherical_vae"]
)
def test_knn_cli_writes_accuracies_per_label_count(model: str, tmp_path: Path) -> None:
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

    svae_knn.main(
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

    result = json.loads((tmp_path / "ckpt" / "m" / "svae_knn_final.json").read_text())
    assert result["geometry"] == (
        "euclidean" if model == "conv_gaussian_vae" else "geodesic"
    )
    for count in (4, 8):
        assert 0.0 <= result[f"acc_{count}"] <= 1.0
        assert result[f"acc_{count}_std"] >= 0.0


def test_knn_table_reads_the_knn_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for model, accuracy in (("gauss", 0.70), ("tnb", 0.80)):
        for seed in range(3):
            run_dir = tmp_path / "ckpt" / f"mnist_{model}_d5_seed{seed}"
            run_dir.mkdir(parents=True)
            record = {f"acc_{n}": accuracy + 0.01 * seed for n in (100, 600, 1000)}
            (run_dir / "svae_knn_final.json").write_text(json.dumps(record))

    svae_table.main(
        [
            "--kind",
            "knn",
            "--models",
            "gauss",
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
    assert "TNBBeta N=1000" in rows[0]
    assert "**0.81 ± 0.01**" in rows[2]
    assert "0.71 ± 0.01" in rows[2] and "**0.71" not in rows[2]


def test_significant_winner_needs_a_significant_margin_over_every_competitor() -> None:
    clear = {
        "a": [1.0, 1.1, 0.9, 1.0],
        "b": [0.0, 0.1, -0.1, 0.0],
        "c": [0.2, 0.3, 0.1, 0.2],
    }
    close = {"a": [1.0, 1.1, 0.9, 1.0], "b": [0.98, 1.08, 0.9, 1.0]}

    assert svae_table.significant_winner(clear, 0.01) == "a"
    assert svae_table.significant_winner(close, 0.01) is None
    assert svae_table.significant_winner({"a": [1.0, 2.0], "b": [1.0]}, 0.01) is None
    assert (
        svae_table.format_mean_std([1.0, 3.0], True, scale=100) == "**200.00 ± 141.42**"
    )
