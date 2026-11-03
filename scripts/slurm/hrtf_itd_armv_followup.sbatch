#!/bin/bash
# ITD-only HRTF arm V follow-up (apps/synthetic/hrtf_itd_armv_followup.py), 80 runs:
#   decoder {learned_axis, fixed_axis} x family {vmf, power_spherical, tnbbeta,
#   tnbbeta_init0} x split seed 0-9
#   sbatch --array=0-9 scripts/slurm/hrtf_itd_armv_followup.sbatch
# Each array task runs PROCS (default 8) worker processes sharing its GPU; worker p of
# task t takes cell (t * PROCS + p), stepping by the total shard count, so 10 tasks x 8
# workers = 80 shards run all 80 cells concurrently (the `general` QOS allows 24 running
# tasks; size the array to the quota left, extra tasks just queue). Finished cells are
# skipped, so resubmitting stragglers is safe. No --device flag: a node with a bad GPU
# exits 75 and the script resubmits itself (scripts/slurm/no_gpu_retry.sh). TRAIN_ARGS
# passes extra flags (smoke tests only).
#SBATCH --job-name=hrtf-armv-fu
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

PROCS="${PROCS:-8}"
OUT="${OUT:-hrtf_runs/armv_followup}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=()
for decoder in learned_axis fixed_axis; do
    for family in vmf power_spherical tnbbeta tnbbeta_init0; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            CELLS+=("${decoder} ${family} ${seed}")
        done
    done
done

run_cell() {
    local decoder="$1" family="$2" seed="$3"
    [[ -f "${OUT}/${decoder}/${family}/s${seed}.json" ]] && return 0
    # shellcheck disable=SC2086
    uv run --no-sync python -m apps.synthetic.hrtf_itd_armv_followup train \
        --decoder "${decoder}" --family "${family}" --seed "${seed}" \
        --out-dir "${OUT}" ${TRAIN_ARGS}
}

LOG_DIR="slurm_logs/hrtf-armv-fu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    (
        for ((i = shard; i < ${#CELLS[@]}; i += NUM_SHARDS)); do
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/hrtf_itd_armv_followup.sbatch"; then
    exit 0
fi
exit "${STATUS}"
