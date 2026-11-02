#!/bin/bash
# Four-class ring designs (apps/synthetic/ring_classes.py), 90 runs:
#   design 1 x {vmf, power_spherical, tnbbeta, tnbbeta_init0, tnbbeta_fixed_mu} x 10 seeds
#   design 2 x {vmf, power_spherical, tnbbeta, tnbbeta_init0} x 10 seeds
#   sbatch --array=0-22 scripts/slurm/ring_classes.sbatch
# Each array task runs PROCS (default 4) worker processes sharing its GPU; worker p of
# task t takes cell (t * PROCS + p), stepping by the total shard count, so 23 tasks x 4
# workers = 92 shards run all 90 cells concurrently (the `general` QOS allows 24 running
# tasks). Finished cells are skipped, so resubmitting stragglers is safe. No --device
# flag: a node with a bad GPU exits 75 and the script resubmits itself
# (scripts/slurm/no_gpu_retry.sh). TRAIN_ARGS passes extra flags (smoke tests only).
#SBATCH --job-name=ring-classes
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
OUT="${OUT:-ring_recovery/classes}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=()
for design in 1 2; do
    if [[ "${design}" == "1" ]]; then
        ARMS=(vmf power_spherical tnbbeta tnbbeta_init0 tnbbeta_fixed_mu)
    else
        ARMS=(vmf power_spherical tnbbeta tnbbeta_init0)
    fi
    for arm in "${ARMS[@]}"; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            CELLS+=("${design} ${arm} ${seed}")
        done
    done
done

run_cell() {
    local design="$1" arm="$2" seed="$3"
    [[ -f "${OUT}/design${design}/${arm}/s${seed}.json" ]] && return 0
    # shellcheck disable=SC2086
    uv run --no-sync python -m apps.synthetic.ring_classes train \
        --design "${design}" --arm "${arm}" --seed "${seed}" \
        --out-dir "${OUT}" ${TRAIN_ARGS}
}

LOG_DIR="slurm_logs/ring-classes_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/ring_classes.sbatch"; then
    exit 0
fi
exit "${STATUS}"
