#!/bin/bash
# S-VAE Table 4: link prediction on Cora, Citeseer and Pubmed for the Gaussian, vMF and
# TNBBeta latents. One array task per (dataset, family) runs the whole hyperparameter grid
# (3 learning rates x 3 dropouts x 3 latent dims, 5 seeds each) and writes
# $TNBBETA_CHECKPOINT_DIR/link_prediction/<dataset>_<family>.json.
#   3 datasets x 3 families = 9 tasks -> 0-8
# The Planetoid files must already be downloaded (apps.data.download_planetoid). A task
# that lands on a node with no usable GPU resubmits itself excluding that node
# (scripts/slurm/no_gpu_retry.sh, up to 5 attempts).
# Submit from the repo root:  sbatch scripts/slurm/link_prediction.sbatch
# Then print the table:  uv run python -m apps.link_prediction.table
#SBATCH --job-name=link-prediction
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

DATASETS=(cora citeseer pubmed)
FAMILIES=(gaussian vmf tnbbeta)

i="${SLURM_ARRAY_TASK_ID}"
FAMILY="${FAMILIES[$((i % ${#FAMILIES[@]}))]}"
DATASET="${DATASETS[$((i / ${#FAMILIES[@]}))]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${DATASET} ${FAMILY}"

set +e
uv run --no-sync python -m apps.link_prediction.main --dataset "${DATASET}" --family "${FAMILY}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/link_prediction.sbatch"; then
    exit 0
fi
exit "${STATUS}"
