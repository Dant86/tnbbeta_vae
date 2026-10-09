#!/bin/bash
# S-VAE Table 4-style link prediction on Cora, Citeseer and Pubmed for all four latent
# families. One array task per (dataset, family) runs the whole hyperparameter grid
# (3 learning rates x 3 dropouts x the family's own --latent-dims list below, 5 seeds
# each) and writes $TNBBETA_CHECKPOINT_DIR/link_prediction/<dataset>_<family>.json
# (apps.link_prediction.main's "configurations" list in that file keeps every
# (lr, dropout, latent_dim) cell's result, not just the validation-selected one -- the
# per-dimension breakdown this is actually a dimension sweep for).
#   3 datasets x 4 families = 12 tasks -> 0-11
#
# CONVENTION (same as scripts/slurm/mnist_sweep.sbatch and CLAUDE.md's "Sphere models'
# latent_dim is the AMBIENT dimension" note): MANIFOLD_DIMS below are the Gaussian's own
# "d" (its latent space is R^d); the Gaussian family uses latent_dim = d directly, and
# every sphere family (vmf, tnbbeta, power_spherical) uses latent_dim = d + 1 -- one
# more ambient dimension than the Gaussian at the same manifold dimension d, so the
# sphere has the same degrees of freedom as the Gaussian being compared against,
# instead of one fewer.
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
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

DATASETS=(cora citeseer pubmed)
FAMILIES=(gaussian vmf tnbbeta power_spherical)
AMBIENT_OFFSET=(0 1 1 1)
MANIFOLD_DIMS=(16 32 64)

i="${SLURM_ARRAY_TASK_ID}"
FAMILY_INDEX=$((i % ${#FAMILIES[@]}))
FAMILY="${FAMILIES[$FAMILY_INDEX]}"
DATASET="${DATASETS[$((i / ${#FAMILIES[@]}))]}"
OFFSET="${AMBIENT_OFFSET[$FAMILY_INDEX]}"
LATENT_DIMS=()
for DIM in "${MANIFOLD_DIMS[@]}"; do
    LATENT_DIMS+=($((DIM + OFFSET)))
done
echo "Task ${SLURM_ARRAY_TASK_ID}: ${DATASET} ${FAMILY} (latent dims ${LATENT_DIMS[*]})"

set +e
uv run --no-sync python -m apps.link_prediction.main \
    --dataset "${DATASET}" --family "${FAMILY}" --latent-dims "${LATENT_DIMS[@]}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/link_prediction.sbatch"; then
    exit 0
fi
exit "${STATUS}"
