#!/bin/bash
# Evaluates a finished run. Submit from the repo root:
#   sbatch scripts/slurm/eval.sbatch <run-name> [extra eval args...]
# Results are written next to the run's checkpoint (see apps/eval/main.py).
#SBATCH --job-name=tnbbeta-eval
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

uv run --no-sync python -m apps.eval.main --run-name "${RUN_NAME}" "$@"
