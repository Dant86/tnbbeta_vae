#!/bin/bash
# Trains ONE model on MNIST with the S-VAE paper's protocol (batch 64, 100-epoch KL warm-up,
# early stopping with patience 50 on the validation set), then writes its Table 1 metrics,
# k-NN accuracy, confidence-only probe, and (TNBBeta only) the p/q latitude diagnostic. Same
# as one task of mnist_sweep.sbatch, for ad hoc runs. From the repo root:
#   sbatch scripts/slurm/mnist_run.sbatch <model> <run-name> [extra apps.train.main args...]
# e.g. a vMF run that starts at kappa = 40 instead of the reference kappa of about 1.7:
#   sbatch --exclude=k002,k003 scripts/slurm/mnist_run.sbatch conv_vmf_vae mnist_vmfk_d40_seed0 \
#       --set latent_dim=40 --set initial_kappa=40 --seed 0
# Later --set flags override the defaults below (hidden_channels=32). Safe to resubmit: training
# resumes, and a task on a node with no usable GPU resubmits itself excluding that node
# (scripts/slurm/no_gpu_retry.sh, up to 5 attempts).
#SBATCH --job-name=mnist-run
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

MODEL="$1"
RUN_NAME="$2"
shift 2

set +e
uv run --no-sync python -m apps.train.main \
    --model "${MODEL}" --run-name "${RUN_NAME}" --resume --dataset mnist \
    --set hidden_channels=32 \
    --batch-size 64 --epochs 1000 --patience 50 --kl-warmup-epochs 100 \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/mnist_run.sbatch" "${MODEL}" "${RUN_NAME}" "$@"; then
    exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then
    exit "${STATUS}"
fi

uv run --no-sync python -m apps.eval.svae_metrics --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
uv run --no-sync python -m apps.eval.svae_knn --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
uv run --no-sync python -m apps.eval.confidence_probe --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
uv run --no-sync python -m apps.eval.svae_latitude --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
