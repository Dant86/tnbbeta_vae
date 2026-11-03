#!/bin/bash
# CPU jobs of the scale_free density figure study: AUX=pytest | analyze.
#   AUX=analyze sbatch scripts/slurm/scale_free_density_aux.sbatch   (after the density stage runs finish)
#SBATCH --job-name=sf-density-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
case "${AUX:?set AUX=pytest|analyze|check}" in
    pytest) uv run --no-sync pytest -q -p no:cacheprovider tests/apps/test_scale_free_density_figure.py tests/apps/test_graph_search.py ;;
    full) uv run --no-sync pytest -q -p no:cacheprovider ;;
    analyze) uv run --no-sync python -m apps.eval.scale_free_density_figure analyze --workers "${SLURM_CPUS_PER_TASK}" ;;
    check) uv run --no-sync ruff format --check . ; uv run --no-sync ruff check . ; uv run --no-sync pyright ;;
esac
