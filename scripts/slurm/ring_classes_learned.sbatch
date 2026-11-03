#!/bin/bash
# Four-class ring designs with a learned decoder (apps/synthetic/ring_classes_learned.py),
# 240 runs: {D1, D2g, D2l} x {mlp, single_index} x {vmf, power_spherical, tnbbeta,
# tnbbeta_init0} x 10 seeds.
#   sbatch --array=0-N scripts/slurm/ring_classes_learned.sbatch
# Each array task runs PROCS (default 4) worker processes sharing its GPU; worker p of task
# t takes cell (t * PROCS + p), stepping by the total shard count (TOTAL_TASKS * PROCS), so
# all 240 cells are covered as long as TOTAL_TASKS * PROCS shards exist. TOTAL_TASKS defaults
# to the array size (the `general` QOS allows 24 running / 200 submitted tasks per user, and
# other jobs may share that quota). Finished cells are skipped, so resubmitting stragglers
# is safe. No --device flag: a node with a bad GPU exits 75 and the script resubmits itself
# (scripts/slurm/no_gpu_retry.sh). TRAIN_ARGS passes extra flags (smoke tests only).
#SBATCH --job-name=ring-learned
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

PROCS="${PROCS:-4}"
OUT="${OUT:-ring_recovery/classes_learned}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
    for setting in D1 D2g D2l; do
        for decoder in mlp single_index; do
            for arm in vmf power_spherical tnbbeta tnbbeta_init0; do
                CELLS+=("${setting} ${decoder} ${arm} ${seed}")
            done
        done
    done
done

run_cell() {
    local setting="$1" decoder="$2" arm="$3" seed="$4"
    [[ -f "${OUT}/${setting}/${decoder}/${arm}/s${seed}.json" ]] && return 0
    # shellcheck disable=SC2086
    uv run --no-sync python -m apps.synthetic.ring_classes_learned train \
        --setting "${setting}" --decoder "${decoder}" --arm "${arm}" --seed "${seed}" \
        --out-dir "${OUT}" ${TRAIN_ARGS}
}

LOG_DIR="slurm_logs/ring-learned_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/ring_classes_learned.sbatch"; then
    exit 0
fi
exit "${STATUS}"
