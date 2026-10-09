"""Tests for scripts/slurm/link_prediction_run.sbatch (single ad hoc run)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "slurm" / "link_prediction_run.sbatch"


def _run(
    tmp_path: Path, uv_status: int = 0, attempt: str | None = None
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log, sbatch_log = tmp_path / "uv.txt", tmp_path / "sbatch.txt"
    stubs = {
        "uv": f'echo "$@" >> "{uv_log}"\nexit {uv_status}',
        "scontrol": 'echo "JobId=1 ExcNodeList=k002 NumNodes=1"',
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
        "SLURMD_NODENAME": "k003",
    }
    env.pop("SLURM_ARRAY_TASK_ID", None)
    if attempt is not None:
        env["TNB_ATTEMPT"] = attempt
    result = subprocess.run(
        [
            "bash",
            str(_SCRIPT),
            "dblp",
            "tnbbeta",
            "--lrs",
            "0.01",
            "--dropouts",
            "0",
            "--latent-dims",
            "16",
            "--epochs",
            "5",
            "--seeds",
            "0",
        ],  # fmt: skip
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    uv_calls = uv_log.read_text().splitlines() if uv_log.exists() else []
    sbatch_calls = sbatch_log.read_text().splitlines() if sbatch_log.exists() else []
    return result, uv_calls, sbatch_calls


def test_runs_link_prediction_with_the_given_dataset_and_family(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert sbatch_calls == []
    (call,) = uv_calls
    assert "apps.link_prediction.main --dataset dblp --family tnbbeta" in call
    assert "--lrs 0.01 --dropouts 0 --latent-dims 16 --epochs 5 --seeds 0" in call


def test_does_not_hardcode_device(tmp_path: Path) -> None:
    """Regression test for job 2120361 (2026-10-09): a previous version passed
    --device cuda unconditionally, which select_device always honors verbatim with
    no availability check -- on a broken GPU node that skipped select_device's own
    clean exit(75) entirely, crashing later with an uncaught RuntimeError (exit code
    1) that resubmit_if_no_gpu doesn't recognize, so it never resubmitted. --device
    must stay unset here so apps.link_prediction.main's own select_device(None) can
    detect that case and exit 75 on purpose.
    """
    _, uv_calls, _ = _run(tmp_path)

    (call,) = uv_calls
    assert "--device" not in call


def test_no_gpu_resubmits_the_whole_job_with_its_arguments(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, uv_status=75)

    assert result.returncode == 0
    assert len(uv_calls) == 1
    (call,) = sbatch_calls
    assert "--exclude=k002,k003" in call and "--array" not in call
    assert "TNB_ATTEMPT=2" in call
    assert call.endswith(
        "scripts/slurm/link_prediction_run.sbatch dblp tnbbeta "
        "--lrs 0.01 --dropouts 0 --latent-dims 16 --epochs 5 --seeds 0"
    )
    assert "resubmitting (attempt 2 of 5)" in result.stderr


def test_other_failures_propagate(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, uv_status=3)

    assert result.returncode == 3 and len(uv_calls) == 1 and sbatch_calls == []
