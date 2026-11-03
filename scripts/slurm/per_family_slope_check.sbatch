#!/bin/bash
# Descriptive per-family slope check (CPU only, cached fresh results). Usage: sbatch scripts/slurm/per_family_slope_check.sbatch
#SBATCH --job-name=pf-slope
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --no-sync python -m apps.eval.per_family_slope_check
