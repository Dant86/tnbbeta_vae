#!/bin/bash
# CPU-only helpers: AUX=mytests | quality | pytest | summarize.
#SBATCH --job-name=kappa-ctl-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
FILES="apps/synthetic/linear_decoder_kappa_control.py apps/synthetic/linear_decoder_kappa_control_plot.py tests/apps/test_linear_decoder_kappa_control.py"
case "${AUX:?set AUX=pytest|mytests|quality|summarize}" in
    pytest) uv run --no-sync pytest -q -p no:cacheprovider ;;
    mytests) uv run --no-sync pytest -q -p no:cacheprovider --no-cov tests/apps/test_linear_decoder_kappa_control.py ;;
    quality) uv run --no-sync ruff format --check ${FILES}; uv run --no-sync ruff check ${FILES}; uv run --no-sync pyright ${FILES} ;;
    summarize) uv run --no-sync python -m apps.synthetic.linear_decoder_kappa_control summarize --out-dir ring_recovery/kappa_control --parent-dir ring_recovery/linear_decoder ;;
esac
