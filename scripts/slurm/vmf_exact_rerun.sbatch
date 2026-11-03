#!/bin/bash
# CPU rerun of the vMF arms of the metric-first graph search with the library vs exact
# vMF sampler (apps/synthetic/vmf_exact_rerun.py). Usage:
#   uv run python -m apps.synthetic.vmf_exact_rerun manifest
#   TOTAL_TASKS=16 sbatch --array=0-15 scripts/slurm/vmf_exact_rerun.sbatch
# Worker p of task t takes shard (t * PROCS + p) of (TOTAL_TASKS * PROCS); finished runs
# are skipped, so resubmitting is safe.
#SBATCH --job-name=vmf-exact-rerun
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PROCS="${PROCS:-6}"
TOTAL_TASKS="${TOTAL_TASKS:?set TOTAL_TASKS}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

LOG_DIR="slurm_logs/vmf-exact-rerun_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    uv run --no-sync python -m apps.synthetic.vmf_exact_rerun run \
        --shard "${shard}" --num-shards "${NUM_SHARDS}" --device cpu \
        > "${LOG_DIR}/shard_${shard}.log" 2>&1 &
    PIDS+=($!)
done
STATUS=0
for pid in "${PIDS[@]}"; do
    set +e
    wait "${pid}"
    worker_status=$?
    set -e
    if (( worker_status != 0 )); then
        echo "a worker failed with status ${worker_status}" >&2
        STATUS=${worker_status}
    fi
done
exit "${STATUS}"
