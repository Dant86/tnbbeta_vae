#!/bin/bash
# Job array over (model x likelihood sigma x latent dim x seed).
# Edit the grid below; keep --array's upper bound at (product of sizes) - 1.
#   3 models x 2 sigmas x 2 latent dims x 3 seeds = 36 tasks -> 0-35
# Submit from the repo root:  sbatch scripts/slurm/sweep.sbatch
# %6 limits concurrency to 6 tasks at a time.
#SBATCH --job-name=tnbbeta-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-35%6
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODELS=(conv_gaussian_vae conv_vmf_vae conv_tnbbeta_spherical_vae)
SIGMAS=(0.1 0.2)
LATENT_DIMS=(32 128)
SEEDS=(0 1 2)
EPOCHS=50

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
i=$((i / ${#SEEDS[@]}))
DIM="${LATENT_DIMS[$((i % ${#LATENT_DIMS[@]}))]}"
i=$((i / ${#LATENT_DIMS[@]}))
SIGMA="${SIGMAS[$((i % ${#SIGMAS[@]}))]}"
i=$((i / ${#SIGMAS[@]}))
MODEL="${MODELS[$i]}"

EXTRA=()
if [[ "${MODEL}" == conv_tnbbeta_spherical_vae ]]; then
    EXTRA+=(--uniform-prior)
fi

RUN_NAME="${MODEL#conv_}_s${SIGMA}_d${DIM}_seed${SEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME}"

uv run --no-sync python -m apps.train.main \
    --model "${MODEL}" --run-name "${RUN_NAME}" --resume \
    --set latent_dim="${DIM}" --set likelihood_scale="${SIGMA}" \
    --epochs "${EPOCHS}" --seed "${SEED}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" ${EXTRA[@]+"${EXTRA[@]}"}
