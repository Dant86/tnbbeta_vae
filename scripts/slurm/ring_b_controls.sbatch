#!/bin/bash
# Controls for "why does the learned-decoder condition B not show TNBBeta ahead"
# (writeup/results/ring_condition_b_unpacked_2026-11-01.md). All cells are stage-3 runs at d = 3
# into ring_recovery/b_controls/<tag>/stage3; none touches the finished l23_d3 outputs.
#   PROCS=8 sbatch --array=0-19 scripts/slurm/ring_b_controls.sbatch
# Tags: n32 / n128 = more ELBO samples (learned decoder, all four free families);
#       mckl_n4 / mckl_n32 = vMF and PS trained with a Monte Carlo KL like TNBBeta;
#       a_n32 = true-decoder tnbbeta_p_init with 32 samples; ep300 = 300 epochs, seeds 0-4,
#       learned decoder (4 families) and true decoder (vmf, tnbbeta_p_init).
# Worker p of task t takes cells t*PROCS+p, stepping by the total shard count. Finished cells are
# skipped. No --device flag: a bad-GPU node exits 75 and the script resubmits itself.
#SBATCH --job-name=ring-b-controls
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

DIM=3
PROCS="${PROCS:-8}"
OUT="${OUT:-ring_recovery/b_controls}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# cell = "tag condition family seed extra-args..."
CELLS=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
    for family in vmf power_spherical tnbbeta tnbbeta_p_init; do
        CELLS+=("n32 learned ${family} ${seed} --num-elbo-samples 32")
        CELLS+=("n128 learned ${family} ${seed} --num-elbo-samples 128")
    done
    for family in vmf power_spherical; do
        CELLS+=("mckl_n4 learned ${family} ${seed} --mc-kl")
        CELLS+=("mckl_n32 learned ${family} ${seed} --mc-kl --num-elbo-samples 32")
    done
    CELLS+=("a_n32 true tnbbeta_p_init ${seed} --num-elbo-samples 32")
done
for seed in 0 1 2 3 4; do
    for family in vmf power_spherical tnbbeta tnbbeta_p_init; do
        CELLS+=("ep300 learned ${family} ${seed} --epochs 300")
    done
    for family in vmf tnbbeta_p_init; do
        CELLS+=("ep300 true ${family} ${seed} --epochs 300")
    done
done

run_cell() {
    local tag="$1" condition="$2" family="$3" seed="$4"
    shift 4
    [[ -f "${OUT}/${tag}/stage3/d${DIM}_${condition}_${family}_s${seed}.json" ]] && return 0
    uv run --no-sync python -m apps.synthetic.ring_recovery stage3 \
        --dim "${DIM}" --condition "${condition}" --family "${family}" \
        --seed "${seed}" --out-dir "${OUT}/${tag}/stage3" "$@"
}

LOG_DIR="slurm_logs/ring-b-controls_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/ring_b_controls.sbatch"; then
    exit 0
fi
exit "${STATUS}"
