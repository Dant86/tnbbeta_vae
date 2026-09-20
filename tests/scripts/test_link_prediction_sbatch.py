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
