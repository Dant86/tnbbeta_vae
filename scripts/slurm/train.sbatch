#!/bin/bash
# Trains one model. Submit from the repo root:
#   sbatch scripts/slurm/train.sbatch <model> <run-name> [extra train args...]
# e.g.
#   sbatch scripts/slurm/train.sbatch conv_tnbbeta_spherical_vae tnb_d32_seed0 \
#       --set latent_dim=32 --set likelihood_scale=0.1 --epochs 50
#
# Safe to resubmit or requeue: --resume continues from the run's latest
# checkpoint and exits immediately if the run already finished.
#
# If the node has no usable GPU (CUDA fails to start although one was
# allocated), apps.train.main exits with code 75. This script then resubmits
# itself with that node added to the exclude list, up to MAX_ATTEMPTS times.
# The attempt just made shows as COMPLETED in sacct; look for the
# "resubmitting" line in its .err file.
#SBATCH --job-name=tnbbeta-train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

NO_GPU_EXIT_CODE=75
MAX_ATTEMPTS=5
ATTEMPT="${TNB_ATTEMPT:-1}"

MODEL="$1"
RUN_NAME="$2"
shift 2

set +e
uv run --no-sync python -m apps.train.main \
    --model "${MODEL}" --run-name "${RUN_NAME}" --resume \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
STATUS=$?
set -e

if [[ "${STATUS}" -ne "${NO_GPU_EXIT_CODE}" ]]; then
    exit "${STATUS}"
fi

if (( ATTEMPT >= MAX_ATTEMPTS )); then
    echo "No usable GPU after ${ATTEMPT} attempts; giving up." >&2
    exit 1
fi

# Keep whatever this job already excluded (e.g. from --exclude at submit time).
EXCLUDED="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p')"
if [[ "${EXCLUDED}" == "(null)" ]]; then
    EXCLUDED=""
fi
NEW_EXCLUDE="${EXCLUDED:+${EXCLUDED},}${SLURMD_NODENAME}"

echo "No usable GPU on ${SLURMD_NODENAME}; resubmitting (attempt $((ATTEMPT + 1))" \
    "of ${MAX_ATTEMPTS}) excluding ${NEW_EXCLUDE}" >&2
sbatch --exclude="${NEW_EXCLUDE}" --export="ALL,TNB_ATTEMPT=$((ATTEMPT + 1))" \
    "${SLURM_SUBMIT_DIR}/scripts/slurm/train.sbatch" "${MODEL}" "${RUN_NAME}" "$@"
