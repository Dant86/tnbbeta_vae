#!/bin/bash
# CPU jobs of the dense real-graph study: AUX=pytest | full | analyze | check | smoke.
#   AUX=pytest sbatch scripts/slurm/dense_graph_aux.sbatch
#   AUX=analyze sbatch scripts/slurm/dense_graph_aux.sbatch   (after the runs finish)
#SBATCH --job-name=dense-graph-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export REAL_GRAPH_STUDY=dense
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
case "${AUX:?set AUX=pytest|full|analyze|check}" in
    pytest) env -u REAL_GRAPH_STUDY uv run --no-sync pytest -q -p no:cacheprovider tests/apps/test_real_graph.py tests/apps/test_real_graph_analysis.py tests/data/test_dense_graphs.py ;;
    full) env -u REAL_GRAPH_STUDY uv run --no-sync pytest -q -p no:cacheprovider ;;
    analyze) uv run --no-sync python -m apps.real_graph.analyze ${ANALYZE_ARGS:-} ;;
    check) uv run --no-sync ruff format --check apps/real_graph src/tnbbeta_vae/data/dense_graphs.py src/tnbbeta_vae/models/real_graph_eval.py tests/apps/test_real_graph.py tests/apps/test_real_graph_analysis.py ; uv run --no-sync ruff check . ; uv run --no-sync pyright ;;
esac
