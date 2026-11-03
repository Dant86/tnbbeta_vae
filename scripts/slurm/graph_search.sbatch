#!/bin/bash
# One stage of the wide graph-type search (apps/synthetic/graph_search.py). Usage:
#   uv run python -m apps.synthetic.graph_search manifest --stage pilot --draws 4   # first
#   STAGE=pilot sbatch --array=0-23 scripts/slurm/graph_search.sbatch
# Each array task runs PROCS worker processes sharing its GPU; worker p of task t takes
# shard (t * PROCS + p) of (array size * PROCS) shards of the manifest. Finished
# runs are skipped, so resubmitting (or the no-GPU resubmit below) is safe. The `general`
# QOS runs 24 tasks at once; extra array tasks queue. No --device flag: a node with a bad
# GPU exits 75 and the script resubmits itself.
#SBATCH --job-name=graph-search
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

STAGE="${STAGE:?set STAGE (pilot, discovery or confirmation)}"
PROCS="${PROCS:-6}"
# Exported so a no-GPU resubmission of one task (task count 1 there) keeps its shards.
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))

LOG_DIR="slurm_logs/graph-search_${STAGE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    uv run --no-sync python -m apps.synthetic.graph_search run \
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/graph_search.sbatch"; then
    exit 0
fi
exit "${STATUS}"
