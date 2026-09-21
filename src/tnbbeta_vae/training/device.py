"""Device selection that refuses a silent CPU fallback under Slurm."""

from __future__ import annotations

import os
import sys

import torch

__all__ = ["NO_GPU_EXIT_CODE", "SLURM_GPU_VARIABLES", "select_device"]

SLURM_GPU_VARIABLES = ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS")
# EX_TEMPFAIL: the sbatch scripts resubmit the job (excluding the node) on this code.
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
