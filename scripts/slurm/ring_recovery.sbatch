#!/bin/bash
# Ring-posterior Levels 2 and 3 at one ambient dimension (apps/synthetic/ring_recovery.py).
#   DIM=3 sbatch --array=0-16 scripts/slurm/ring_recovery.sbatch
# Cells: 2 stage-2 runs (2000- and 10000-step budgets), then every stage-3 run
# (2 conditions x 5 families x 10 seeds = 100). Each array task runs PROCS worker processes
# sharing its GPU; worker p of task t takes cell (t * PROCS + p) of the list, stepping by
# the total shard count, so ceil(102 / (array size * PROCS)) cells each. Finished cells are
# skipped, so resubmitting is safe. No --device flag: a node with a bad GPU exits 75 and the
# script resubmits itself (scripts/slurm/no_gpu_retry.sh).
#SBATCH --job-name=ring-recovery
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

DIM="${DIM:-3}"
PROCS="${PROCS:-6}"
OUT="${OUT:-ring_recovery/l23_d${DIM}}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=("stage2 2000" "stage2 10000")
for condition in true learned; do
    for family in vmf power_spherical tnbbeta tnbbeta_fixed_m_neg tnbbeta_fixed_m_ring; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            CELLS+=("stage3 ${condition} ${family} ${seed}")
        done
    done
done

run_cell() {
    local kind="$1"; shift
    if [[ "${kind}" == "stage2" ]]; then
        local steps="$1"
        [[ -f "${OUT}/stage2/d${DIM}_steps${steps}.json" ]] && return 0
        uv run --no-sync python -m apps.synthetic.ring_recovery stage2 \
            --dim "${DIM}" --steps "${steps}" --out-dir "${OUT}/stage2"
    else
        local condition="$1" family="$2" seed="$3"
        [[ -f "${OUT}/stage3/d${DIM}_${condition}_${family}_s${seed}.json" ]] && return 0
        uv run --no-sync python -m apps.synthetic.ring_recovery stage3 \
            --dim "${DIM}" --condition "${condition}" --family "${family}" \
            --seed "${seed}" --out-dir "${OUT}/stage3"
    fi
}

LOG_DIR="slurm_logs/ring-recovery_d${DIM}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/ring_recovery.sbatch"; then
    exit 0
fi
exit "${STATUS}"
