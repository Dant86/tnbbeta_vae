#!/bin/bash
# Re-runs of the ring-posterior fits with the exact TNBBeta log q in the ELBO
# (apps/synthetic/tnb_exact_log_q_check.py; see writeup/results/tnb_exact_log_q_in_elbo_2026-11-01.md).
#   sbatch --array=0-2 scripts/slurm/tnb_exact_log_q.sbatch
# Task 0: stage 2, d = 3, 2000 steps, TNBBeta only (amortised-like and random-restart inits).
# Task 1: stage 3, condition B ("learned" decoder), TNBBeta seed 4, default epsilon floor 1e-4.
# Task 2: the same with --epsilon-floor 0.3.
# No --device flag: a node with a bad GPU exits 75 and the script resubmits itself
# (scripts/slurm/no_gpu_retry.sh).
#SBATCH --job-name=tnb-exact-logq
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

OUT="${OUT:-ring_recovery/exact_log_q}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

STATUS=0
set +e
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        uv run --no-sync python -m apps.synthetic.tnb_exact_log_q_check stage2 \
            --out "${OUT}/stage2_d3_steps2000.json"
        STATUS=$?
        ;;
    1 | 2)
        if [[ "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then FLOOR=1e-4; else FLOOR=0.3; fi
        DIR="${OUT}/stage3_floor${FLOOR}"
        uv run --no-sync python -m apps.synthetic.ring_recovery stage3 \
            --dim 3 --condition learned --family tnbbeta --seed 4 \
            --epsilon-floor "${FLOOR}" --out-dir "${DIR}"
        STATUS=$?
        if (( STATUS == 0 )); then
            uv run --no-sync python -m apps.synthetic.tnb_exact_log_q_check stage3 \
                --ckpt "${DIR}/ckpt/d3_learned_tnbbeta_s4.pt" --epsilon-floor "${FLOOR}" \
                --out "${DIR}/check_d3_learned_tnbbeta_s4.json"
            STATUS=$?
        fi
        ;;
esac
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/tnb_exact_log_q.sbatch"; then
    exit 0
fi
exit "${STATUS}"
