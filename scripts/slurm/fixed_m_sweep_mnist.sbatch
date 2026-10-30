#!/bin/bash
# Fixed-m sweep on MNIST: free training always lands unimodal (large m); does forcing
# m below / near the modality threshold hurt (or help) reconstruction?
#
# Ambient latent_dim=6 (the paper's d=5), threshold (6-1)/2 = 2.5, so m > -2.5. Grid:
#   -2, -1, 0, 1, 3, 8, 20  (epsilon 0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 22.5)
# 7 conditions x 3 seeds = 21 tasks -> 0-20. Free-epsilon reference: the existing
# mnist_tnbs_d5 runs (same protocol as fixed_epsilon_sweep.sbatch).
#
# Run names: mnistfixm_m<M>_d5_seed<seed> (minus sign written "n", '.' as "p").
#   uv run python -m apps.eval.svae_table --prefix mnistfixm --models <names> ...
# Safe to resubmit: training resumes, finished runs are skipped.
#SBATCH --job-name=fixed-m-mnist
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-20
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

MS=(-2 -1 0 1 3 8 20)
SEEDS=(0 1 2)
DIM=5
LATENT_DIM=6
HIDDEN_CHANNELS=32

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
M="${MS[$((i / ${#SEEDS[@]}))]}"
EPSILON="$(awk -v m="${M}" -v d="${LATENT_DIM}" 'BEGIN{print m + (d-1)/2}')"
TAG="$(echo "${M}" | sed 's/-/n/; s/\./p/')"
RUN_NAME="mnistfixm_m${TAG}_d${DIM}_seed${SEED}"
echo "Task ${i}: ${RUN_NAME} (m=${M}, fixed_epsilon=${EPSILON})"

set +e
uv run --no-sync python -m apps.train.main \
    --model conv_tnbbeta_spherical_vae --run-name "${RUN_NAME}" --resume --dataset mnist \
    --set latent_dim="${LATENT_DIM}" --set hidden_channels="${HIDDEN_CHANNELS}" \
    --set fixed_epsilon="${EPSILON}" \
    --batch-size 64 --epochs 1000 --patience 50 --kl-warmup-epochs 100 \
    --seed "${SEED}" --num-workers "${SLURM_CPUS_PER_TASK:-4}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_m_sweep_mnist.sbatch"; then
    exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then
    exit "${STATUS}"
fi

uv run --no-sync python -m apps.eval.svae_metrics --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
