#!/bin/bash
# Axial Bingham study (apps/synthetic/axial_bingham.py), 2160 runs: d {2,3,4} x sigma
# {0.1,0.25,0.5,1} x decoder {fixed,linear,mlp} x arm {vmf,power_spherical,tnbbeta,
# tnbbeta_init0,vmf_sym,ps_sym} x 10 seeds.
#   sbatch --array=0-N scripts/slurm/axial_bingham.sbatch
# CPU-only: the networks are tiny (a 3-epoch CPU benchmark ran at ~2 s/epoch single-threaded,
# faster per run than four runs sharing one GPU in the earlier ring studies), so no GPU is
# requested and --device cpu is explicit. Each array task runs PROCS (default 8) single-thread
# worker processes; worker p of task t takes cell (t * PROCS + p), stepping by the total shard
# count (TOTAL_TASKS * PROCS). TOTAL_TASKS defaults to the array size (the `general` QOS allows
# 24 running / 200 submitted tasks per user, shared with other jobs). Cells are ordered with the
# confirmatory dimensions (d = 2, 3) first. Finished cells are skipped, so resubmitting
# stragglers is safe. TRAIN_ARGS passes extra flags (smoke tests only).
#SBATCH --job-name=axial-bingham
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=11:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PROCS="${PROCS:-8}"
OUT="${OUT:-ring_recovery/axial_bingham}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CELLS=()
for dim in 2 3 4; do
    for sigma in 0.1 0.25 0.5 1; do
        for decoder in fixed linear mlp; do
            for arm in vmf power_spherical tnbbeta tnbbeta_init0 vmf_sym ps_sym; do
                for seed in 0 1 2 3 4 5 6 7 8 9; do
                    CELLS+=("${dim} ${sigma} ${decoder} ${arm} ${seed}")
                done
            done
        done
    done
done

run_cell() {
    local dim="$1" sigma="$2" decoder="$3" arm="$4" seed="$5"
    [[ -f "${OUT}/d${dim}/s${sigma}/${decoder}/${arm}/s${seed}.json" ]] && return 0
    # shellcheck disable=SC2086
    uv run --no-sync python -m apps.synthetic.axial_bingham train \
        --dim "${dim}" --sigma "${sigma}" --decoder "${decoder}" --arm "${arm}" \
        --seed "${seed}" --device cpu --out-dir "${OUT}" ${TRAIN_ARGS}
}

LOG_DIR="slurm_logs/axial-bingham_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOG_DIR}"
PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    (
        for ((i = shard; i < ${#CELLS[@]}; i += NUM_SHARDS)); do
            # shellcheck disable=SC2086
            run_cell ${CELLS[i]} || echo "CELL FAILED: ${CELLS[i]}"
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
        if (( STATUS == 0 )); then STATUS=${worker_status}; fi
    fi
done
exit "${STATUS}"
