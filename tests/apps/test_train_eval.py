"""End-to-end smoke tests for the train and eval CLIs (tiny fake data, CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.eval import export_latents
from apps.eval import main as eval_main
from apps.train import main as train_main
from tnbbeta_vae.data.cifar10 import Cifar10Images

_MODELS = [
    ("conv_gaussian_vae", []),
    ("conv_tnbbeta_spherical_vae", []),
]


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
    monkeypatch.setattr(eval_main, "load_cifar10", _fake_dataset)
    monkeypatch.setattr(export_latents, "load_cifar10", _fake_dataset)


def _train_args(model: str, extra: list[str], epochs: int) -> list[str]:
    return [
        "--model", model, "--set", "latent_dim=4", "--set", "hidden_channels=8",
        "--epochs", str(epochs), "--batch-size", "4", "--num-workers", "0",
        "--device", "cpu", "--run-name", "smoke", *extra,
    ]  # fmt: skip


@pytest.mark.parametrize(("model", "extra"), _MODELS)
def test_train_then_eval(model: str, extra: list[str], tmp_path: Path) -> None:
    train_main.main(_train_args(model, extra, epochs=1))

    run_checkpoints = tmp_path / "ckpt" / "smoke"
    assert (run_checkpoints / "latest.pt").exists()
    assert (run_checkpoints / "final.pt").exists()
    assert (tmp_path / "runs" / "smoke" / "train_args.json").exists()

    eval_main.main(
        [
            "--run-name",
            "smoke",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--num-samples",
            "2",
            "--num-prior-samples",
            "8",
            "--reference-size",
            "8",
            "--batch-size",
            "4",
        ]  # fmt: skip
    )

    results = json.loads((run_checkpoints / "eval_final_test.json").read_text())
    for key in ("elbo", "kl", "mse", "psnr_db", "prior_nn_ratio"):
        assert key in results
    assert results["model_name"] == model
    assert (run_checkpoints / "prior_samples_final.png").exists()
    assert (run_checkpoints / "reconstructions_final.png").exists()


def test_resume_skips_completed_and_continues_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_main.main(_train_args("conv_gaussian_vae", [], epochs=1))

    train_main.main([*_train_args("conv_gaussian_vae", [], epochs=1), "--resume"])
    assert "already complete" in capsys.readouterr().out

    (tmp_path / "ckpt" / "smoke" / "final.pt").unlink()
    train_main.main([*_train_args("conv_gaussian_vae", [], epochs=2), "--resume"])
    assert "Resumed from epoch 1" in capsys.readouterr().out
    checkpoint = torch.load(tmp_path / "ckpt" / "smoke" / "final.pt")
    assert checkpoint["epochs_completed"] == 2


def test_select_device_refuses_a_silent_cpu_fallback_under_slurm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")

    with pytest.raises(SystemExit) as excinfo:
        train_main._select_device(None)
    assert excinfo.value.code == train_main.NO_GPU_EXIT_CODE
    assert train_main._select_device("cpu").type == "cpu"


def test_select_device_uses_cpu_when_no_gpu_was_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in train_main._SLURM_GPU_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    assert train_main._select_device(None).type == "cpu"


@pytest.mark.parametrize(("model", "extra"), _MODELS)
def test_export_latents_writes_parameters_labels_and_probe(
    model: str, extra: list[str], tmp_path: Path
) -> None:
    train_main.main(_train_args(model, extra, epochs=1))

    export_latents.main(
        [
            "--run-name",
            "smoke",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--batch-size",
            "4",
        ]
    )

    run_checkpoints = tmp_path / "ckpt" / "smoke"
    latents = np.load(run_checkpoints / "latents_final_test.npz")
    assert latents["labels"].tolist() == [i % 3 for i in range(8)]
    for key in ("direction", "z", "kl"):
        assert len(latents[key]) == 8
    is_tnbbeta = model == "conv_tnbbeta_spherical_vae"
    assert ("concentration" in latents.files) != is_tnbbeta
    assert latents["direction"].shape[1] == 4
    assert ("p" in latents.files) == is_tnbbeta
    probe = json.loads((run_checkpoints / "latent_probe_final_test.json").read_text())
    assert set(probe) >= {"direction", "z_sample", "direction_cosine", "z_cosine"}
    if is_tnbbeta:
        mode = latents["mode_direction"]
        flips = np.where(latents["p"][:, None] > 0.5, 1.0, -1.0)
        assert np.allclose(mode, flips * latents["direction"])
