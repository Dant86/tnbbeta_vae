"""Node health checks that make bad Slurm nodes fail fast (exit code 75) for a retry."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "NO_GPU_EXIT_CODE",
    "SLURM_GPU_VARIABLES",
    "require_readable_storage",
    "select_device",
]

SLURM_GPU_VARIABLES = ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS")
# EX_TEMPFAIL: the sbatch scripts resubmit the job (excluding the node) on this code,
# both for a node whose GPU does not start and for one that cannot read shared storage.
NO_GPU_EXIT_CODE = 75


def select_device(requested: str | None) -> torch.device:
    """Picks the device to run on, refusing a silent CPU fallback under Slurm.

    If a GPU was allocated to the job but CUDA can't initialize (a node problem),
    PyTorch quietly falls back to the CPU and the run crawls. Failing here makes that
    visible immediately. An explicit ``requested`` device is always honored.

    Args:
        requested: The ``--device`` argument, or ``None`` to choose.

    Returns:
        The device to run on.

    Raises:
        SystemExit: With ``NO_GPU_EXIT_CODE`` if a Slurm GPU was allocated but CUDA
            is unavailable.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if any(os.environ.get(name) for name in SLURM_GPU_VARIABLES):
        print(
            "Slurm allocated a GPU but CUDA is unavailable on this node; refusing "
            "to fall back to the CPU. Resubmit, excluding this node (sbatch "
            "--exclude=<node>), or pass --device cpu to run on the CPU anyway.",
            file=sys.stderr,
        )
        raise SystemExit(NO_GPU_EXIT_CODE)
    return torch.device("cpu")


def require_readable_storage(*directories: Path) -> None:
    """Exits with ``NO_GPU_EXIT_CODE`` if this node cannot read a shared directory.

    A node with a broken view of the shared filesystem (an ``Input/output error`` on a
    plain ``stat``) otherwise crashes every task it is handed within seconds. Failing
    with the retry code lets the sbatch scripts resubmit the task on another node. A
    directory that does not exist yet is fine.

    Args:
        directories: Directories the run reads or writes (data, checkpoints, logs).

    Raises:
        SystemExit: With ``NO_GPU_EXIT_CODE`` if listing a directory raises an error
            other than "not found".
    """
    for directory in directories:
        try:
            with os.scandir(directory) as entries:
                next(entries, None)
        except FileNotFoundError:
            continue
        except OSError as error:
            print(
                f"This node cannot read {directory} ({error}); refusing to start so "
                "the task can be resubmitted on another node.",
                file=sys.stderr,
            )
            raise SystemExit(NO_GPU_EXIT_CODE) from error
