#!/bin/bash
# Grid test of the vMF sampler (apps/synthetic/vmf_sampler_bias.py), CPU only.
#   sbatch --array=0-3 scripts/slurm/vmf_sampler_bias.sbatch
# Task 0..3 = (float32, float64) x (library, exact envelope).
#SBATCH --job-name=vmf-bias
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
DTYPES=(float32 float32 float64 float64)
EXACT=(0 1 0 1)
uv run --no-sync python -m apps.synthetic.vmf_sampler_bias run \
    --dtype "${DTYPES[SLURM_ARRAY_TASK_ID]}" --exact-envelope "${EXACT[SLURM_ARRAY_TASK_ID]}"
