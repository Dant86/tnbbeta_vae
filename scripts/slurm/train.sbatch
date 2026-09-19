#!/bin/bash
# Trains one model. Submit from the repo root:
#   sbatch scripts/slurm/train.sbatch <model> <run-name> [extra train args...]
# e.g.
#   sbatch scripts/slurm/train.sbatch conv_vmf_vae vmf_s0.1_d32_seed0 \
#       --set latent_dim=32 --set likelihood_scale=0.1 --epochs 50
#
# Safe to resubmit or requeue: --resume continues from the run's latest
# checkpoint and exits immediately if the run already finished.
#SBATCH --job-name=tnbbeta-train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODEL="$1"
RUN_NAME="$2"
shift 2

uv run --no-sync python -m apps.train.main \
    --model "${MODEL}" --run-name "${RUN_NAME}" --resume \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
