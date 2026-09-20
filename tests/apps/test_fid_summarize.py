"""Tests for the FID and summary eval scripts (tiny fake data, CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.eval import fid, summarize
from apps.train import main as train_main
from tnbbeta_vae.data.cifar10 import Cifar10Images


def _fake_dataset(*_args: object, train: bool, **_kwargs: object) -> Cifar10Images:
    generator = torch.Generator().manual_seed(0 if train else 1)
    return Cifar10Images(
        [(torch.rand(3, 32, 32, generator=generator), i % 3) for i in range(8)]
    )


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train_main, "load_cifar10", _fake_dataset)
    monkeypatch.setattr(fid, "load_cifar10", _fake_dataset)


def test_frechet_distance_is_zero_for_identical_and_grows_with_shift() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 6))

    assert fid.frechet_distance(features, features) == pytest.approx(0.0, abs=1e-6)
    shifted = fid.frechet_distance(features, features + 2.0)
    assert shifted == pytest.approx(6 * 4.0, rel=1e-3)


def test_fid_main_writes_prior_fid_and_real_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fid,
        "_build_extractor",
        lambda _device: lambda images: images.flatten(1)[:, :12],
    )
    train_main.main(
        [
            "--model",
            "conv_gaussian_vae",
            "--set",
            "latent_dim=4",
            "--set",
            "hidden_channels=8",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--run-name",
            "smoke",
        ]  # fmt: skip
    )

    fid.main(
        [
            "--run-name",
            "smoke",
            "--num-samples",
            "8",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "ckpt" / "smoke" / "fid_final.json").read_text())
    assert results["num_samples"] == 8
    assert results["fid_prior"] > 0
    assert results["fid_real_floor"] > 0


def test_linear_probe_separates_linearly_separable_classes() -> None:
    from apps.eval import export_latents

    torch.manual_seed(0)
    labels = torch.arange(200) % 10
    features = torch.nn.functional.one_hot(labels, 10).float() * 3
    features += 0.1 * torch.randn_like(features)

    assert export_latents._linear_accuracy(features, labels, 100) > 0.95


def test_summarize_prints_row_with_gaps_and_dashes_for_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "ckpt" / "run_a"
    run_dir.mkdir(parents=True)
    test = {
        "epochs_completed": 200,
        "likelihood_scale": 0.06,
        "elbo": -100.0,
        "kl": 300.0,
        "psnr_db": 23.0,
        "prior_nn_ratio": 1.1,
    }
    train = {"elbo": -90.0, "psnr_db": 23.5}
    (run_dir / "eval_final_test.json").write_text(json.dumps(test))
    (run_dir / "eval_final_train.json").write_text(json.dumps(train))

    summarize.main(["run_a"])

    row = capsys.readouterr().out.strip().splitlines()[-1]
    assert row.startswith("| run_a | 200 |")
    assert "| 0.50 | -10.0 |" in row
    assert "| - |" in row

    with pytest.raises(SystemExit):
        summarize.main([])
