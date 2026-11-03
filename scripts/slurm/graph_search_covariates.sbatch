#!/bin/bash
# Graph-side covariates of the metric-first graph-search study (CPU only). Usage:
#   STAGE=discovery DRAWS=32 sbatch scripts/slurm/graph_search_covariates.sbatch
#SBATCH --job-name=gs-covariates
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --no-sync python -m apps.eval.graph_search_covariates \
    --stage "${STAGE:?set STAGE}" --draws "${DRAWS:?set DRAWS}" --workers "${SLURM_CPUS_PER_TASK}"
