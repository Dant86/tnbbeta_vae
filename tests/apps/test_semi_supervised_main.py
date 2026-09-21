"""End-to-end test of apps/semi_supervised/main.py on a tiny fake MNIST (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from apps.semi_supervised import main as semi_main
from tnbbeta_vae.data.mnist import MnistImages


def _fake_mnist(*_args: object, split: str, **_kwargs: object) -> MnistImages:
    generator = torch.Generator().manual_seed({"train": 0, "val": 1, "test": 2}[split])
    images = torch.rand(40, 1, 28, 28, generator=generator)
    return MnistImages(images, torch.arange(40) % 10, dynamic=split == "train")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(semi_main, "load_mnist", _fake_mnist)


def _args(*extra: str) -> list[str]:
    return [
        "--run-name", "semi", "--z1-dim", "3", "--z2-dim", "3", "--num-labels", "10",
        "--batch-size", "8", "--epochs", "2", "--patience", "5", "--device", "cpu",
        *extra,
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("z1", "z2"), [("gaussian", "gaussian"), ("vmf", "vmf"), ("tnbbeta", "gaussian")]
)
def test_trains_writes_accuracy_and_keeps_the_best_epoch(
    z1: str, z2: str, tmp_path: Path
) -> None:
    semi_main.main(
        _args("--z1-family", z1, "--z2-family", z2, "--kappa-init", "dimension")
    )

    checkpoints = tmp_path / "ckpt" / "semi"
    result = json.loads((checkpoints / "semi_supervised_final.json").read_text())
    assert result["z1_family"] == z1 and result["z2_family"] == z2
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["num_labels"] == 10
    assert result["kappa_init"] == "dimension"
    assert (checkpoints / "best.pt").exists() and (checkpoints / "final.pt").exists()
    final = torch.load(checkpoints / "final.pt")
    assert final["config"]["num_examples"] == 40
    metrics = (tmp_path / "runs" / "semi" / "metrics.jsonl").read_text()
    assert '"val_loss"' in metrics


def test_resume_skips_a_finished_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    semi_main.main(_args())
    capsys.readouterr()

    semi_main.main(_args("--resume"))

    assert "already complete" in capsys.readouterr().out
