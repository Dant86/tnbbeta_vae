#!/bin/bash
# S-VAE Table 3: semi-supervised MNIST (100 labels) with the stacked M1+M2 model. Job array
# over (variant x z1 dim x z2 dim x seed); each task trains with early stopping and writes
# semi_supervised_final.json (test accuracy).
#   variants: nn = Gaussian+Gaussian, ss = vMF+vMF, sn = vMF+Gaussian,
#             tt = TNBBeta+TNBBeta, tn = TNBBeta+Gaussian (families of z1 and z2)
#   5 variants x 3 z1 dims x 3 z2 dims x 5 seeds = 225 tasks -> 0-224
# CONVENTION: as in the S-VAE paper, the dims below are MANIFOLD dimensions: a sphere latent
# with dimension d is S^d in R^(d+1), so its ambient (model) dimension is d + 1; Gaussian
# latents use d. Run names keep d. Every vMF head starts at kappa = its ambient dimension
# (--kappa-init dimension); the reference start collapses at high dimension.
# Submit from the repo root:  sbatch scripts/slurm/semi_supervised_sweep.sbatch
# (MNIST must already be downloaded.) Safe to resubmit: runs resume and finished ones are
# skipped. A task on a node with no usable GPU resubmits itself excluding that node
# (scripts/slurm/no_gpu_retry.sh). Print the table with
#   uv run python -m apps.semi_supervised.table
#SBATCH --job-name=semi-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-224%12
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

VARIANTS=(nn ss sn tt tn)
Z1_FAMILIES=(gaussian vmf vmf tnbbeta tnbbeta)
Z2_FAMILIES=(gaussian vmf gaussian tnbbeta gaussian)
Z1_OFFSET=(0 1 1 1 1)
Z2_OFFSET=(0 1 0 1 0)
DIMS=(5 10 50)
SEEDS=(0 1 2 3 4)
ALPHA=0.5

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
i=$((i / ${#SEEDS[@]}))
Z2_DIM="${DIMS[$((i % ${#DIMS[@]}))]}"
i=$((i / ${#DIMS[@]}))
Z1_DIM="${DIMS[$((i % ${#DIMS[@]}))]}"
i=$((i / ${#DIMS[@]}))
V="${i}"
Z1_AMBIENT=$((Z1_DIM + Z1_OFFSET[V]))
Z2_AMBIENT=$((Z2_DIM + Z2_OFFSET[V]))
RUN_NAME="semi_${VARIANTS[$V]}_z${Z1_DIM}_${Z2_DIM}_seed${SEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME}"

set +e
uv run --no-sync python -m apps.semi_supervised.main --run-name "${RUN_NAME}" --resume \
    --z1-family "${Z1_FAMILIES[$V]}" --z2-family "${Z2_FAMILIES[$V]}" \
    --z1-dim "${Z1_AMBIENT}" --z2-dim "${Z2_AMBIENT}" --kappa-init dimension \
    --alpha "${ALPHA}" --seed "${SEED}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/semi_supervised_sweep.sbatch"; then
    exit 0
fi
exit "${STATUS}"
