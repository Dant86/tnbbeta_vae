#!/bin/bash
# Sphere-PCA linear-decoder ring study (apps/synthetic/linear_decoder_ring.py), 960 runs:
# decoder L x d {3,5,9,17} x sigma {0.1,0.2,0.4} x 4 arms x 10 seeds (480), plus decoders
# F, M, Lb at sigma 0.2 only (160 each). Cells are queued L, F, M, Lb (seed outermost within
# a decoder), the order of `linear_decoder_ring cells`.
#   sbatch --array=0-N scripts/slurm/linear_decoder_ring.sbatch
# Each array task runs PROCS (default 6) worker processes sharing its GPU; worker p of task
# t takes cell (t * PROCS + p), stepping by the total shard count (TOTAL_TASKS * PROCS), so
# all cells are covered as long as TOTAL_TASKS * PROCS shards exist. TOTAL_TASKS defaults
# to the array size (the `general` QOS allows 24 running / 200 submitted tasks per user, and
# other jobs may share that quota). Finished cells are skipped, so resubmitting stragglers
# is safe. No --device flag: a node with a bad GPU exits 75 and the script resubmits itself
# (scripts/slurm/no_gpu_retry.sh). TRAIN_ARGS passes extra flags (smoke tests only).
#SBATCH --job-name=lin-ring
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

PROCS="${PROCS:-6}"
OUT="${OUT:-ring_recovery/linear_decoder}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=()
for decoder in L F M Lb; do
    if [[ "${decoder}" == "L" ]]; then SIGMAS=(0.1 0.2 0.4); else SIGMAS=(0.2); fi
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        for dim in 3 5 9 17; do
            for sigma in "${SIGMAS[@]}"; do
                for arm in vmf power_spherical tnbbeta tnbbeta_init0; do
                    CELLS+=("${decoder} ${dim} ${sigma} ${arm} ${seed}")
                done
            done
        done
    done
done

run_cell() {
    local decoder="$1" dim="$2" sigma="$3" arm="$4" seed="$5"
    [[ -f "${OUT}/${decoder}/d${dim}_s${sigma}/${arm}/s${seed}.json" ]] && return 0
    # shellcheck disable=SC2086
    uv run --no-sync python -m apps.synthetic.linear_decoder_ring train \
        --decoder "${decoder}" --dim "${dim}" --sigma "${sigma}" --arm "${arm}" \
        --seed "${seed}" --out-dir "${OUT}" ${TRAIN_ARGS}
}

LOG_DIR="slurm_logs/lin-ring_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/linear_decoder_ring.sbatch"; then
    exit 0
fi
exit "${STATUS}"
