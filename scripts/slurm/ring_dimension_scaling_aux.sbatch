#!/bin/bash
# CPU-only helpers of the ring dimension-scaling study: AUX=pytest (full test suite),
# AUX=info (information.json) or AUX=summarize.
#   AUX=pytest sbatch scripts/slurm/ring_dimension_scaling_aux.sbatch
#SBATCH --job-name=ring-dimscale-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT="${OUT:-ring_recovery/dimension_scaling}"
case "${AUX:?set AUX=pytest|info|summarize}" in
    pytest) uv run --no-sync pytest -q -p no:cacheprovider ;;
    info) uv run --no-sync python -m apps.synthetic.ring_dimension_scaling info --out-dir "${OUT}" ;;
    summarize) uv run --no-sync python -m apps.synthetic.ring_dimension_scaling summarize --out-dir "${OUT}" ;;
esac
