"""Tests for scripts/slurm/link_prediction.sbatch, using a stub uv."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("task", "dataset", "family"),
    [
        (0, "cora", "gaussian"),
        (2, "cora", "tnbbeta"),
        (4, "citeseer", "vmf"),
        (8, "pubmed", "tnbbeta"),
    ],
)
def test_sweep_script_maps_array_index_to_dataset_and_family(
    tmp_path: Path, task: int, dataset: str, family: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv_calls.txt"
    stub = bin_dir / "uv"
    stub.write_text(f'#!/bin/bash\necho "$@" >> "{log}"\n')
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(_REPO),
        "SLURM_ARRAY_TASK_ID": str(task),
    }

    result = subprocess.run(
        ["bash", str(_REPO / "scripts" / "slurm" / "link_prediction.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        log.read_text()
        .strip()
        .endswith(f"apps.link_prediction.main --dataset {dataset} --family {family}")
    )


def test_no_gpu_resubmits_only_this_task_excluding_the_node(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch_calls.txt"
    stubs = {
        "uv": "exit 75",
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
        "SLURM_ARRAY_TASK_ID": "4",
        "SLURMD_NODENAME": "k003",
    }

    result = subprocess.run(
        ["bash", str(_REPO / "scripts" / "slurm" / "link_prediction.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call = sbatch_log.read_text().strip()
    assert "--exclude=k002,k003" in call and "--array=4" in call
    assert "TNB_ATTEMPT=2" in call
