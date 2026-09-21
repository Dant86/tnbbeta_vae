#!/bin/bash
# S-VAE Table 1 sweep: job array over (model x latent dim x seed) on MNIST, with the
# paper's protocol (batch 64, KL warm-up 100 epochs, early stopping with patience 50 on
# the 10k validation images). Each task trains, then writes svae_metrics_final_test.json (Table 1) and
# svae_knn_final.json (Table 2).
# Edit the grid below; keep --array's upper bound at (product of sizes) - 1.
#   3 models x 6 latent dims x 5 seeds = 90 tasks -> 0-89
# Submit from the repo root:  sbatch scripts/slurm/mnist_sweep.sbatch
# MNIST must already be downloaded (apps.data.download_mnist). Safe to resubmit: the
# training step resumes, and finished runs are skipped. A task that lands on a node with
# no usable GPU resubmits itself excluding that node (scripts/slurm/no_gpu_retry.sh, up to
# 5 attempts); the attempt just made shows as COMPLETED in sacct, with a "resubmitting"
# line in its .err file. Rerun other failed tasks with
#   sbatch --exclude=k002 --array=<ids> scripts/slurm/mnist_sweep.sbatch
#SBATCH --job-name=mnist-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-89%12
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

MODELS=(conv_gaussian_vae conv_vmf_vae conv_tnbbeta_spherical_vae)
SHORT_NAMES=(gauss vmf tnb)
LATENT_DIMS=(2 5 10 20 40 128)
SEEDS=(0 1 2 3 4)
HIDDEN_CHANNELS=32

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
i=$((i / ${#SEEDS[@]}))
DIM="${LATENT_DIMS[$((i % ${#LATENT_DIMS[@]}))]}"
i=$((i / ${#LATENT_DIMS[@]}))
MODEL="${MODELS[$i]}"
RUN_NAME="mnist_${SHORT_NAMES[$i]}_d${DIM}_seed${SEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME}"

set +e
uv run --no-sync python -m apps.train.main \
    --model "${MODEL}" --run-name "${RUN_NAME}" --resume --dataset mnist \
    --set latent_dim="${DIM}" --set hidden_channels="${HIDDEN_CHANNELS}" \
    --batch-size 64 --epochs 1000 --patience 50 --kl-warmup-epochs 100 \
    --seed "${SEED}" --num-workers "${SLURM_CPUS_PER_TASK:-4}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/mnist_sweep.sbatch"; then
    exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then
    exit "${STATUS}"
fi

uv run --no-sync python -m apps.eval.svae_metrics --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.svae_knn --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
