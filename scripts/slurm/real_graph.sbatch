#!/bin/bash
# One stage of the real-graph revision (apps/real_graph/run.py). Usage:
#   uv run python -m apps.real_graph.run manifest --stage tune          # first (login node)
#   STAGE=tune TOTAL_TASKS=12 sbatch --array=0-11 scripts/slurm/real_graph.sbatch
# Each array task runs PROCS worker processes sharing its GPU; worker p of task t takes
# shard (t * PROCS + p) of (TOTAL_TASKS * PROCS) shards of the manifest. Finished runs are
# skipped, so resubmitting (or the no-GPU resubmit below) is safe. No --device flag: a node
# with a bad GPU exits 75 and the script resubmits itself.
#SBATCH --job-name=real-graph
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

STAGE="${STAGE:?set STAGE (tune, main, thin, community or pin)}"
PROCS="${PROCS:-6}"
# Exported so a no-GPU resubmission of one task keeps its shards.
export TOTAL_TASKS="${TOTAL_TASKS:?set TOTAL_TASKS (the array size)}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))

LOG_DIR="slurm_logs/real-graph_${STAGE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    uv run --no-sync python -m apps.real_graph.run run \
        --stage "${STAGE}" --shard "${shard}" --num-shards "${NUM_SHARDS}" \
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
        if (( worker_status == NO_GPU_EXIT_CODE || STATUS == 0 )); then STATUS=${worker_status}; fi
    fi
done

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/real_graph.sbatch"; then
    exit 0
fi
exit "${STATUS}"
