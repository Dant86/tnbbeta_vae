#!/bin/bash
# CPU checkpoint diagnostics (apps/synthetic/ring_condition_b_diagnostics.py), 2 seeds per task.
#   STAGE3=ring_recovery/l23_d3/stage3 OUT=ring_recovery/b_diag/l23 sbatch --array=0-4 scripts/slurm/ring_b_diagnostics.sbatch
#SBATCH --job-name=ring-b-diag
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${SLURM_SUBMIT_DIR}"
STAGE3="${STAGE3:-ring_recovery/l23_d3/stage3}"
OUT="${OUT:-ring_recovery/b_diag/l23}"
mkdir -p "${OUT}"
s0=$(( SLURM_ARRAY_TASK_ID * 2 ))
uv run --no-sync python -m apps.synthetic.ring_condition_b_diagnostics \
    --stage3-dir "${STAGE3}" --out "${OUT}/diag_${SLURM_ARRAY_TASK_ID}.json" \
    --seeds "${s0}" "$(( s0 + 1 ))" --workers 16
