"""Tests for scripts/slurm/fixed_mean_direction_sweep.sbatch, using stub uv/sbatch."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "slurm" / "fixed_mean_direction_sweep.sbatch"


def _run(
    tmp_path: Path, task: int, train_status: int = 0
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log, sbatch_log = tmp_path / "uv.txt", tmp_path / "sbatch.txt"
    stubs = {
        "uv": (
            f'echo "$@" >> "{uv_log}"\n'
            f'if [[ "$*" == *apps.train.main* ]]; then exit {train_status}; fi'
        ),
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
    ("task", "run_name", "seed"),
    [
        (0, "mnistfixmu_tnb_d5_seed0", "0"),
        (1, "mnistfixmu_tnb_d5_seed1", "1"),
        (2, "mnistfixmu_tnb_d5_seed2", "2"),
    ],
)
def test_array_index_maps_to_seed(
    tmp_path: Path, task: int, run_name: str, seed: str
) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, task)

    assert result.returncode == 0, result.stderr
    train, metrics, knn, confidence, confidence_epsilon, latitude = uv_calls
    assert f"--model conv_tnbbeta_spherical_vae --run-name {run_name} --resume" in train
    assert "--set latent_dim=6 --set hidden_channels=32" in train
    assert "--set fixed_mean_direction=true" in train
    assert "--batch-size 64 --epochs 1000 --patience 50 --kl-warmup-epochs 100" in train
    assert f"--seed {seed}" in train
    assert f"apps.eval.svae_metrics --run-name {run_name}" in metrics
    assert f"apps.eval.svae_knn --run-name {run_name}" in knn
    assert f"apps.eval.confidence_probe --run-name {run_name}" in confidence
    assert "--param" not in confidence
    assert f"apps.eval.confidence_probe --run-name {run_name}" in confidence_epsilon
    assert "--param epsilon" in confidence_epsilon
    assert f"apps.eval.svae_latitude --run-name {run_name}" in latitude
    assert sbatch_calls == []


def test_no_gpu_resubmits_only_this_task(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, task=1, train_status=75)

    assert result.returncode == 0
    assert len(uv_calls) == 1  # no evaluation after a node failure
    (call,) = sbatch_calls
    assert "--exclude=k003" in call and "--array=1" in call and "TNB_ATTEMPT=2" in call
    assert call.endswith("scripts/slurm/fixed_mean_direction_sweep.sbatch")


def test_other_failures_propagate_and_skip_evaluation(tmp_path: Path) -> None:
    result, uv_calls, sbatch_calls = _run(tmp_path, task=0, train_status=3)

    assert result.returncode == 3 and len(uv_calls) == 1 and sbatch_calls == []
