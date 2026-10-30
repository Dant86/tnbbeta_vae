#!/bin/bash
# Fixed-m sweep on the synthetic SBM: does forcing TNBBeta's modality offset
# m = epsilon - (latent_dim - 1) / 2 change link prediction, and where?
#
# Free training on the SBM lands at m in roughly [-6, -3] (see
# writeup/results/sbm_pout_dose_response_2026-10-09.md); the old fixed-epsilon ablation
# only tried epsilon in {<1, 1, >1}, which never resolved that band. At latent_dim=16 the
# threshold is 7.5, so m must stay above -7.5 (epsilon > 0). Grid, in m:
#   -7, -6, -4.5, -3, -1.5, -0.5, 0, 0.5, 2, 5   (epsilon 0.5 ... 12.5)
# plus a free-epsilon reference ("free"), 5 seeds each:
#   11 conditions x 5 seeds = 55 tasks -> 0-54, all fanned out at once.
#
# TNBBeta only: vMF / power-spherical / Gaussian don't depend on epsilon; reuse their
# dose-response numbers (same graph, same --graph-seed) as reference.
#
# Outputs: runs/fixed_m_sweep_sbm/<cond>_seed<seed>/sbm_recovery.json. Aggregate with
#   uv run python -m apps.eval.fixed_m_sweep_table --root runs/fixed_m_sweep_sbm
# Safe to resubmit. A task on a node with no usable GPU resubmits itself excluding that
# node (scripts/slurm/no_gpu_retry.sh).
#SBATCH --job-name=fixed-m-sbm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-54
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CONDITIONS=(free -7 -6 -4.5 -3 -1.5 -0.5 0 0.5 2 5)
SEEDS=(0 1 2 3 4)
P_OUT=0.05
LATENT_DIM=16

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
COND="${CONDITIONS[$((i / ${#SEEDS[@]}))]}"
RUN_NAME="fixedm_${COND}"
OUT_DIR="runs/fixed_m_sweep_sbm/${COND}_seed${SEED}"
echo "Task ${i}: cond=${COND} seed=${SEED}"

M_ARGS=()
if [[ "${COND}" != "free" ]]; then
    M_ARGS=(--fixed-m "${COND}")
fi

set +e
uv run --no-sync python -m apps.synthetic.sbm_recovery \
    --out-dir "${OUT_DIR}" --p-out "${P_OUT}" --latent-dim "${LATENT_DIM}" \
    --epochs 500 --device cuda --families tnbbeta \
    --run-name "${RUN_NAME}" --seed "${SEED}" "${M_ARGS[@]}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_m_sweep_sbm.sbatch"; then
    exit 0
fi
exit "${STATUS}"
