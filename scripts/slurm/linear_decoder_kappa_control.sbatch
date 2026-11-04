#!/bin/bash
# Tuned-kappa control (apps/synthetic/linear_decoder_kappa_control.py), 2280 runs: 19 new arms x
# d {3,5,9,17} x sigma {0.1,0.2,0.4} x 10 seeds. Cells are listed by `... cells` (non-H arms first).
#   sbatch --array=0-N scripts/slurm/linear_decoder_kappa_control.sbatch
# Each array task runs PROCS worker processes sharing its GPU; worker p of task t takes cells
# (t * PROCS + p) + k * TOTAL_TASKS * PROCS. Finished cells are skipped, so resubmitting is safe.
# REVERSE=1 walks the cell list from the end (to add capacity to a running array).
# OFFSET=n starts the forward walk at cell n (a third array starting mid-list).
# SKIP_ARMS=<egrep> drops matching cells from the list (Deviation D2: pinned kappa >= 10).
# No --device flag: a node with a bad GPU exits 75 and the script resubmits itself.
#SBATCH --job-name=kappa-ctl
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

PROCS="${PROCS:-8}"
OUT="${OUT:-ring_recovery/kappa_control}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mapfile -t CELLS < <(uv run --no-sync python -m apps.synthetic.linear_decoder_kappa_control cells | { grep -Ev "${SKIP_ARMS:-^$}" || true; })

run_cell() {
    local dim="$1" sigma="$2" arm="$3" seed="$4"
    [[ -f "${OUT}/L/d${dim}_s${sigma}/${arm}/s${seed}.json" ]] && return 0
    # shellcheck disable=SC2086
    uv run --no-sync python -m apps.synthetic.linear_decoder_kappa_control train \
        --dim "${dim}" --sigma "${sigma}" --arm "${arm}" --seed "${seed}" \
        --out-dir "${OUT}" ${TRAIN_ARGS}
}

LOG_DIR="slurm_logs/kappa-ctl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    (
        for ((k = shard; k < ${#CELLS[@]}; k += NUM_SHARDS)); do
            # REVERSE=1 walks the list from the end (a second array helping the first).
            if [[ "${REVERSE:-0}" == "1" ]]; then i=$(( ${#CELLS[@]} - 1 - k )); else i=$(( (k + ${OFFSET:-0}) % ${#CELLS[@]} )); fi
            # shellcheck disable=SC2086
            run_cell ${CELLS[i]}
        done
    ) > "${LOG_DIR}/shard_${shard}.log" 2>&1 &
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/linear_decoder_kappa_control.sbatch"; then
    exit 0
fi
exit "${STATUS}"
