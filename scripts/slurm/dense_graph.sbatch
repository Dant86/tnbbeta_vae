#!/bin/bash
# One stage of the dense real-graph study (apps/real_graph/run.py with REAL_GRAPH_STUDY=dense).
#   REAL_GRAPH_STUDY=dense uv run python -m apps.real_graph.run manifest --stage tune   # login node
#   STAGE=tune TOTAL_TASKS=12 sbatch --array=0-11 scripts/slurm/dense_graph.sbatch
#   STAGE=main TOTAL_TASKS=6  sbatch --array=0-5  scripts/slurm/dense_graph.sbatch
# STAGE=tune: worker p of task t takes shard (t * PROCS + p) of (TOTAL_TASKS * PROCS)
# shards of the manifest. STAGE=main: task t is one cell (3 graphs x features on/off) and
# its PROCS workers share that cell's 4 families x 10 seeds. Finished runs are skipped, so
# resubmitting (or the no-GPU resubmit below) is safe. No --device flag: a node with a bad
# GPU exits 75 and the script resubmits itself.
#SBATCH --job-name=dense-graph
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=k003,m001
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export REAL_GRAPH_STUDY=dense

STAGE="${STAGE:?set STAGE (tune or main)}"
PROCS="${PROCS:-6}"
# Exported so a no-GPU resubmission of one task keeps its shards.
export TOTAL_TASKS="${TOTAL_TASKS:?set TOTAL_TASKS (the array size)}"
CELLS=(
    "amazon_computers|on" "amazon_computers|off"
    "amazon_photo|on" "amazon_photo|off"
    "squirrel|on" "squirrel|off"
)

LOG_DIR="slurm_logs/dense-graph_${STAGE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    if [[ "${STAGE}" == "main" ]]; then
        SHARD_ARGS=(--cell "${CELLS[SLURM_ARRAY_TASK_ID]}" --shard "${p}" --num-shards "${PROCS}")
    else
        SHARD_ARGS=(--shard $(( SLURM_ARRAY_TASK_ID * PROCS + p )) --num-shards $(( TOTAL_TASKS * PROCS )))
    fi
    uv run --no-sync python -m apps.real_graph.run run --stage "${STAGE}" "${SHARD_ARGS[@]}" \
        > "${LOG_DIR}/shard_${p}.log" 2>&1 &
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/dense_graph.sbatch"; then
    exit 0
fi
exit "${STATUS}"
