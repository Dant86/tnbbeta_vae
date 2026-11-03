#!/bin/bash
# CPU-only variant of graph_search.sbatch, for when every GPU is taken (the GCN runs of this
# study are small: about 30 s per run on 4 threads). Usage:
#   TOTAL_TASKS=48 STAGE=fresh sbatch --array=0-47 scripts/slurm/graph_search_cpu.sbatch
# Each task runs PROCS worker processes on one node; worker p of task t takes shard
# (t * PROCS + p) of (TOTAL_TASKS * PROCS) shards of the manifest. Finished runs are
# skipped, so resubmitting is safe. The device is the CPU by construction (no --gres).
#SBATCH --job-name=graph-search-cpu
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
STAGE="${STAGE:?set STAGE}"
PROCS="${PROCS:-6}"
TOTAL_TASKS="${TOTAL_TASKS:?set TOTAL_TASKS}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

LOG_DIR="slurm_logs/graph-search-cpu_${STAGE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    uv run --no-sync python -m apps.synthetic.graph_search run \
        --stage "${STAGE}" --shard "${shard}" --num-shards "${NUM_SHARDS}" --device cpu \
        > "${LOG_DIR}/shard_${shard}.log" 2>&1 &
    PIDS+=($!)
done
STATUS=0
for pid in "${PIDS[@]}"; do
    set +e
    wait "${pid}"
    worker_status=$?
    set -e
    if (( worker_status != 0 )); then STATUS=${worker_status}; fi
done
exit "${STATUS}"
