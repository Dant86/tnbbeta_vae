"""Tests for semi_supervised_sweep.sbatch using stub uv/scontrol/sbatch."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "slurm" / "semi_supervised_sweep.sbatch"


def _run(
    tmp_path: Path, task: int, uv_status: int = 0
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log, sbatch_log = tmp_path / "uv.txt", tmp_path / "sbatch.txt"
    stubs = {
        "uv": f'echo "$@" >> "{uv_log}"; exit {uv_status}',
        "scontrol": 'echo "JobId=1 ExcNodeList=(null) NumNodes=1"',
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
        "SLURMD_NODENAME": "k003",
    }
    result = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, check=False
    )
    uv_calls = uv_log.read_text().splitlines() if uv_log.exists() else []
    sbatch_calls = sbatch_log.read_text().splitlines() if sbatch_log.exists() else []
    return result, uv_calls, sbatch_calls


@pytest.mark.parametrize(
    ("task", "run_name", "z1_family", "z2_family", "z1_ambient", "z2_ambient", "seed"),
    [
        # Gaussian latents use d; sphere latents (S^d in R^(d+1)) get ambient d + 1.
        (0, "semi_nn_z5_5_seed0", "gaussian", "gaussian", "5", "5", "0"),
        (4, "semi_nn_z5_5_seed4", "gaussian", "gaussian", "5", "5", "4"),
        (5, "semi_nn_z5_10_seed0", "gaussian", "gaussian", "5", "10", "0"),
        (44, "semi_nn_z50_50_seed4", "gaussian", "gaussian", "50", "50", "4"),
        (45, "semi_ss_z5_5_seed0", "vmf", "vmf", "6", "6", "0"),
        (90, "semi_sn_z5_5_seed0", "vmf", "gaussian", "6", "5", "0"),
        (135, "semi_tt_z5_5_seed0", "tnbbeta", "tnbbeta", "6", "6", "0"),
        (224, "semi_tn_z50_50_seed4", "tnbbeta", "gaussian", "51", "50", "4"),
    ],
)
def test_array_index_maps_to_variant_dims_and_seed(
    tmp_path: Path,
    task: int,
    run_name: str,
    z1_family: str,
    z2_family: str,
    z1_ambient: str,
    z2_ambient: str,
    seed: str,
) -> None:
    result, uv_calls, _ = _run(tmp_path, task)

    assert result.returncode == 0, result.stderr
    (call,) = uv_calls
    assert f"--run-name {run_name} --resume" in call
    assert f"--z1-family {z1_family} --z2-family {z2_family}" in call
    assert f"--z1-dim {z1_ambient} --z2-dim {z2_ambient}" in call
    assert "--kappa-init dimension" in call
    assert f"--seed {seed}" in call


def test_no_gpu_resubmits_only_this_task_and_other_failures_propagate(
    tmp_path: Path,
) -> None:
    result, _, sbatch_calls = _run(tmp_path, 7, uv_status=75)

    assert result.returncode == 0
    (call,) = sbatch_calls
    assert "--exclude=k003" in call and "--array=7" in call and "TNB_ATTEMPT=2" in call

    (tmp_path / "again").mkdir()
    other, _, other_sbatch = _run(tmp_path / "again", 7, uv_status=3)
    assert other.returncode == 3 and other_sbatch == []
