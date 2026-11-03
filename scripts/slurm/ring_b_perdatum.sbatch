#!/bin/bash
# Per-datum posterior optimisation against learned decoders (apps/synthetic/ring_b_perdatum.py).
# 15 cells = decoders of {vmf, power_spherical, tnbbeta_p_init} x seeds 0-4; one cell per task, CPU only.
#   sbatch --array=0-14 scripts/slurm/ring_b_perdatum.sbatch
#   refinement from the encoder's output (lower lr):
#   TAG=refine EXTRA="--refine-encoder --lr 0.01 --steps 1000" sbatch --array=0-14 scripts/slurm/ring_b_perdatum.sbatch
#SBATCH --job-name=ring-b-perdatum
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${SLURM_SUBMIT_DIR}"
VARIANTS=(vmf power_spherical tnbbeta_p_init)
variant="${VARIANTS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}"
seed=$(( SLURM_ARRAY_TASK_ID % 5 ))
TAG="${TAG:-perdatum}"
EXTRA="${EXTRA:-}"
uv run --no-sync python -m apps.synthetic.ring_b_perdatum \
    --ckpt "ring_recovery/l23_d3/stage3/ckpt/d3_learned_${variant}_s${seed}.pt" \
    --variant "${variant}" --out "ring_recovery/b_diag/${TAG}/${variant}_s${seed}.json" ${EXTRA}
