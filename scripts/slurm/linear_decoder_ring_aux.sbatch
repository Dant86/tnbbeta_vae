#!/bin/bash
# CPU-only helpers of the linear-decoder ring study. AUX=pytest (full test suite),
# AUX=mytests (this study's tests only), AUX=quality (ruff + pyright), AUX=summarize or AUX=plot.
#   AUX=pytest sbatch scripts/slurm/linear_decoder_ring_aux.sbatch
#SBATCH --job-name=lin-ring-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT="${OUT:-ring_recovery/linear_decoder}"
case "${AUX:?set AUX=pytest|mytests|quality|summarize|plot}" in
    pytest) uv run --no-sync pytest -q -p no:cacheprovider ;;
    mytests) uv run --no-sync pytest -q -p no:cacheprovider --no-cov tests/data/test_linear_ring.py tests/apps/test_linear_decoder_ring.py tests/apps/test_linear_decoder_ring_plot.py ;;
    quality) uv run --no-sync ruff format --check apps/synthetic/linear_decoder_ring.py apps/synthetic/linear_decoder_ring_plot.py src/tnbbeta_vae/data/linear_ring.py tests/data/test_linear_ring.py tests/apps/test_linear_decoder_ring.py tests/apps/test_linear_decoder_ring_plot.py; uv run --no-sync ruff check apps/synthetic/linear_decoder_ring.py apps/synthetic/linear_decoder_ring_plot.py src/tnbbeta_vae/data/linear_ring.py tests/data/test_linear_ring.py tests/apps/test_linear_decoder_ring.py tests/apps/test_linear_decoder_ring_plot.py; uv run --no-sync pyright apps/synthetic/linear_decoder_ring.py apps/synthetic/linear_decoder_ring_plot.py src/tnbbeta_vae/data/linear_ring.py tests/data/test_linear_ring.py tests/apps/test_linear_decoder_ring.py tests/apps/test_linear_decoder_ring_plot.py ;;
    summarize) uv run --no-sync python -m apps.synthetic.linear_decoder_ring summarize --out-dir "${OUT}" ;;
    plot) uv run --no-sync python -m apps.synthetic.linear_decoder_ring_plot --out-dir "${OUT}" --fig-dir writeup/results/figures ;;
esac
