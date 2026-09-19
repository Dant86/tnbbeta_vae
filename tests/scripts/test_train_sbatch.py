"""Tests for scripts/slurm/train.sbatch's retry logic, using stub uv/scontrol/sbatch."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "slurm" / "train.sbatch"


def _run(
    tmp_path: Path,
    *,
    train_status: int,
    excluded: str = "(null)",
    attempt: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch_calls.txt"
    stubs = {
        "uv": f"exit {train_status}",
        "scontrol": f'echo "JobId=1 ExcNodeList={excluded} NumNodes=1"',
        "sbatch": f'echo "$@" >> "{log}"',
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(_REPO),
        "SLURM_JOB_ID": "1",
        "SLURMD_NODENAME": "k007",
    }
    if attempt is not None:
        env["TNB_ATTEMPT"] = attempt
    result = subprocess.run(
        ["bash", str(_SCRIPT), "conv_x", "run1", "--epochs", "3"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_success_does_not_resubmit(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, train_status=0)

    assert result.returncode == 0
    assert calls == []


def test_other_failures_propagate_without_resubmitting(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, train_status=3)

    assert result.returncode == 3
    assert calls == []


def test_no_gpu_resubmits_excluding_the_node_and_keeping_earlier_excludes(
    tmp_path: Path,
) -> None:
    result, calls = _run(tmp_path, train_status=75, excluded="k002")

    assert result.returncode == 0
    assert len(calls) == 1
    assert "--exclude=k002,k007" in calls[0]
    assert "TNB_ATTEMPT=2" in calls[0]
    assert calls[0].endswith("conv_x run1 --epochs 3")
    assert "resubmitting (attempt 2 of 5)" in result.stderr


def test_no_gpu_with_no_earlier_excludes_excludes_only_this_node(
    tmp_path: Path,
) -> None:
    _, calls = _run(tmp_path, train_status=75, excluded="(null)")

    assert "--exclude=k007 " in calls[0]


@pytest.mark.parametrize("attempt", ["5", "9"])
def test_gives_up_after_the_last_attempt(tmp_path: Path, attempt: str) -> None:
    result, calls = _run(tmp_path, train_status=75, attempt=attempt)

    assert result.returncode == 1
    assert calls == []
    assert "giving up" in result.stderr
