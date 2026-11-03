#!/bin/bash
# Exploratory 'tnbbeta_p_init' stage-3 variant at one ambient dimension: 2 conditions x 10
# seeds = 20 cells, one per worker (outside the pre-registered Holm family).
#   DIM=3 PROCS=1 sbatch --array=0-19 scripts/slurm/ring_recovery_p_init.sbatch
# Worker p of array task t takes cell (t * PROCS + p); finished cells are skipped. No
# --device flag: a bad-GPU node exits 75 and the script resubmits (no_gpu_retry.sh).
#SBATCH --job-name=ring-p-init
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

DIM="${DIM:-3}"
PROCS="${PROCS:-1}"
OUT="${OUT:-ring_recovery/l23_d${DIM}}"
VARIANT=tnbbeta_p_init
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=()
for condition in true learned; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        CELLS+=("${condition} ${seed}")
    done
done

STATUS=0
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    index=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    (( index < ${#CELLS[@]} )) || continue
    read -r condition seed <<< "${CELLS[index]}"
    [[ -f "${OUT}/stage3/d${DIM}_${condition}_${VARIANT}_s${seed}.json" ]] && continue
    uv run --no-sync python -m apps.synthetic.ring_recovery stage3 \
        --dim "${DIM}" --condition "${condition}" --family "${VARIANT}" \
        --seed "${seed}" --out-dir "${OUT}/stage3" &
    PIDS+=($!)
done
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/ring_recovery_p_init.sbatch"; then
    exit 0
fi
exit "${STATUS}"
