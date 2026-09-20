#!/bin/bash
# Job array over (model x latent dim x seed). The likelihood scale is learned
# (LIKELIHOOD_SCALE_START only sets where it starts).
# Edit the grid below; keep --array's upper bound at (product of sizes) - 1.
#   2 models x 2 latent dims x 3 seeds = 12 tasks -> 0-11
# Submit from the repo root:  sbatch scripts/slurm/sweep.sbatch
# %6 limits concurrency to 6 tasks at a time.
#SBATCH --job-name=tnbbeta-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-11%6
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODELS=(conv_gaussian_vae conv_tnbbeta_spherical_vae)
LATENT_DIMS=(32 128)
LIKELIHOOD_SCALE_START=0.1
SEEDS=(0 1 2)
HIDDEN_CHANNELS=64
EPOCHS=100

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
i=$((i / ${#SEEDS[@]}))
DIM="${LATENT_DIMS[$((i % ${#LATENT_DIMS[@]}))]}"
i=$((i / ${#LATENT_DIMS[@]}))
MODEL="${MODELS[$i]}"

RUN_NAME="${MODEL#conv_}_d${DIM}_h${HIDDEN_CHANNELS}_seed${SEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME}"

uv run --no-sync python -m apps.train.main \
    --model "${MODEL}" --run-name "${RUN_NAME}" --resume \
    --set latent_dim="${DIM}" --set likelihood_scale="${LIKELIHOOD_SCALE_START}" \
    --set hidden_channels="${HIDDEN_CHANNELS}" \
    --epochs "${EPOCHS}" --seed "${SEED}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
