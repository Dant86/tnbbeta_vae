#!/bin/bash
# CPU-only helpers of the axial Bingham study: AUX=threshold|normaliser|representability|
# anisotropy (pre-training checks, apps/synthetic/axial_bingham_checks.py), AUX=pytest (full
# suite) or AUX=summarize or AUX=plot (apps/synthetic/axial_bingham*.py), AUX=debug (slurm_logs/debug/dbg.py).
#   AUX=pytest sbatch scripts/slurm/axial_bingham_aux.sbatch
#SBATCH --job-name=axial-bingham-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT="${OUT:-ring_recovery/axial_bingham}"
case "${AUX:?set AUX}" in
    pytest) uv run --no-sync pytest -q -p no:cacheprovider ;;
    summarize) uv run --no-sync python -m apps.synthetic.axial_bingham summarize --out-dir "${OUT}" ;;
    threshold|normaliser|representability|representability_general|anisotropy)
        uv run --no-sync python -m apps.synthetic.axial_bingham_checks "${AUX}" --out-dir "${OUT}/checks" ;;
    debug) PYTHONPATH=. uv run --no-sync python slurm_logs/debug/dbg.py ${DEBUG_ARGS:?set DEBUG_ARGS} ;;
    plot) PYTHONPATH=. uv run --no-sync python -m apps.synthetic.axial_bingham_plot --out-dir "${OUT}" --fig-dir writeup/results/figures ;;
esac
