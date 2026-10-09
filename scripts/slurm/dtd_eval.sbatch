#!/bin/bash
# Evaluates a finished DTD run (apps.eval.dtd_orientation_probe). Submit from the
# repo root:
#   sbatch scripts/slurm/dtd_eval.sbatch <run-name> [extra eval args...]
# Results are written next to the run's checkpoint. Mirrors eval.sbatch's resource
# profile (a GPU, 16G) -- that script hardcodes apps.eval.main, which doesn't apply
# here, so this is its DTD-specific sibling rather than an ad hoc `sbatch --wrap`.
#SBATCH --job-name=dtd-eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

RUN_NAME="$1"
shift 1

uv run --no-sync python -m apps.eval.dtd_orientation_probe --run-name "${RUN_NAME}" "$@"
