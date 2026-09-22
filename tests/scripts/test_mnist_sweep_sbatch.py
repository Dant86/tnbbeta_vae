"""Tests for mnist_sweep.sbatch's no-GPU retry, using stub uv/scontrol/sbatch."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "slurm" / "mnist_sweep.sbatch"


def _run(
    tmp_path: Path,
    *,
    train_status: int,
    excluded: str = "(null)",
    attempt: str | None = None,
    task: int = 12,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv_calls.txt"
    sbatch_log = tmp_path / "sbatch_calls.txt"
    stubs = {
        "uv": (
            f'echo "$@" >> "{uv_log}"\n'
            f'if [[ "$*" == *apps.train.main* ]]; then exit {train_status}; fi'
        ),
        "scontrol": f'echo "JobId=1 ExcNodeList={excluded} NumNodes=1"',
        "sbatch": f'echo "$@" >> "{sbatch_log}"',
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
        "SLURM_ARRAY_TASK_ID": str(task),
        "SLURMD_NODENAME": "k007",
    }
    if attempt is not None:
        env["TNB_ATTEMPT"] = attempt
    result = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, check=False
    )
    uv_calls = uv_log.read_text().splitlines() if uv_log.exists() else []
    sbatch_calls = sbatch_log.read_text().splitlines() if sbatch_log.exists() else []
    return result, uv_calls, sbatch_calls


def test_success_runs_training_metrics_and_knn_without_resubmitting(
    tmp_path: Path,
) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, train_status=0)

    assert result.returncode == 0
    assert len(uv_calls) == 5
    assert (
        "apps.eval.svae_metrics" in uv_calls[1] and "apps.eval.svae_knn" in uv_calls[2]
    )
    assert "apps.eval.confidence_probe" in uv_calls[3]
    assert "apps.eval.svae_latitude" in uv_calls[4]
    assert sbatch_calls == []


def test_other_failures_propagate_and_skip_evaluation(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, train_status=3)

    assert result.returncode == 3
    assert len(uv_calls) == 1
    assert sbatch_calls == []


def test_no_gpu_resubmits_only_this_task_excluding_the_node(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(
        tmp_path, train_status=75, excluded="k002", task=12
    )

    assert result.returncode == 0
    assert len(uv_calls) == 1  # no evaluation after a node failure
    assert len(sbatch_calls) == 1
    call = sbatch_calls[0]
    assert "--exclude=k002,k007" in call
    assert "--array=12" in call
    assert "TNB_ATTEMPT=2" in call
    assert call.endswith("scripts/slurm/mnist_sweep.sbatch")
    assert "resubmitting task 12 (attempt 2 of 5)" in result.stderr


def test_no_gpu_with_no_earlier_excludes_excludes_only_this_node(
    tmp_path: Path,
) -> None:
    _, _, sbatch_calls = _run(tmp_path, train_status=75, excluded="(null)")

    assert "--exclude=k007 " in sbatch_calls[0]


@pytest.mark.parametrize("attempt", ["5", "9"])
def test_gives_up_after_the_last_attempt(tmp_path: Path, attempt: str) -> None:
    result, _, sbatch_calls = _run(tmp_path, train_status=75, attempt=attempt)

    assert result.returncode == 1
    assert sbatch_calls == []
    assert "giving up" in result.stderr
