"""Tests for require_readable_storage and its use by the training entry points."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from apps.semi_supervised import main as semi_main
from apps.train import main as train_main
from tnbbeta_vae.training.device import NO_GPU_EXIT_CODE, require_readable_storage


def test_readable_and_missing_directories_pass(tmp_path: Path) -> None:
    (tmp_path / "there").mkdir()
    (tmp_path / "there" / "file.txt").write_text("x")

    require_readable_storage(tmp_path / "there", tmp_path / "not_created_yet")


def test_an_io_error_exits_with_the_retry_code_and_names_the_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(_path: object) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "scandir", broken)

    with pytest.raises(SystemExit) as excinfo:
        require_readable_storage(tmp_path)

    assert excinfo.value.code == NO_GPU_EXIT_CODE
    err = capsys.readouterr().err
    assert str(tmp_path) in err and "Input/output error" in err


@pytest.mark.parametrize(
    ("entry_point", "argv"),
    [
        (
            train_main.main,
            ["--model", "conv_gaussian_vae", "--run-name", "r", "--resume"],
        ),
        (semi_main.main, ["--run-name", "r", "--resume"]),
    ],
)
def test_training_entry_points_check_storage_before_touching_any_file(
    entry_point: object,
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))

    def broken(_path: object) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "scandir", broken)

    with pytest.raises(SystemExit) as excinfo:
        entry_point(argv)  # pyright: ignore[reportCallIssue]

    assert excinfo.value.code == NO_GPU_EXIT_CODE
