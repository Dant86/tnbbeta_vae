"""Tests for tnbbeta_vae.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from tnbbeta_vae import paths


def test_defaults_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("TNBBETA_DATA_DIR", "TNBBETA_CHECKPOINT_DIR", "TNBBETA_RUNS_DIR"):
        monkeypatch.delenv(name, raising=False)

    assert paths.data_dir() == Path("data")
    assert paths.checkpoint_dir() == Path("checkpoints")
    assert paths.runs_dir() == Path("runs")


def test_reads_dotenv_but_real_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TNBBETA_DATA_DIR=/from/dotenv\nTNBBETA_CHECKPOINT_DIR=/ckpt/dotenv\n"
    )
    monkeypatch.delenv("TNBBETA_DATA_DIR", raising=False)
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", "/ckpt/real")

    assert paths.data_dir() == Path("/from/dotenv")
    assert paths.checkpoint_dir() == Path("/ckpt/real")
