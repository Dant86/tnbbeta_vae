#!/bin/bash
# Runs ONE apps.link_prediction.main invocation on a GPU node, for ad hoc runs (a timing
# smoke test, a single (dataset, family, latent-dim) cell of the com-DBLP comparison) that
# don't fit link_prediction.sbatch's fixed 9-task Planetoid grid. Same as one task of that
# script, but with arbitrary dataset/family/args, the same way mnist_run.sbatch complements
# mnist_sweep.sbatch.
# The dataset must already be downloaded (apps.data.download_planetoid or
# apps.data.download_snap_community). A task that lands on a node with no usable GPU
# resubmits itself excluding that node (scripts/slurm/no_gpu_retry.sh, up to 5 attempts)
# -- but only because --device is left unset below, letting apps.link_prediction.main's
# own select_device(None) detect the broken-node case and exit 75 on purpose. A previous
# version of this script hardcoded --device cuda, which select_device always honors
# verbatim with no availability check at all; on a broken node that skipped the clean
# exit(75) entirely and crashed later, uncaught, inside GraphBatch.to(device) with exit
# code 1 instead -- a code resubmit_if_no_gpu doesn't recognize, so it never resubmitted
# (job 2120361, 2026-10-09). Do not add --device back here.
# Submit from the repo root:
#   sbatch scripts/slurm/link_prediction_run.sbatch <dataset> <family> [extra args...]
# e.g. the com-DBLP timing smoke test (run this one FIRST -- cheap, fast signal before
# committing to the much bigger comparison runs below):
#   sbatch scripts/slurm/link_prediction_run.sbatch dblp tnbbeta \
#       --lrs 0.01 --dropouts 0 --latent-dims 16 --epochs 5 --seeds 0
# or one cell of the real comparison, with a checkpoint saved for the bridge diagnostic:
#   sbatch scripts/slurm/link_prediction_run.sbatch dblp tnbbeta \
#       --lrs 0.01 --dropouts 0 0.2 --latent-dims 16 --epochs 100 --seeds 0 1 2 3 4 \
#       --run-name dblp_tnbbeta_d16
# --mem=64G: a run with multiple configurations x seeds trains several full models
# sequentially in one process (each a full-batch 317,080-node graph); 16G was not enough
# in practice (slurmstepd oom_kill, no Python traceback -- a host-RAM cgroup kill, not a
# CUDA OOM) even though no single part of the pipeline is individually that large --
# consistent with ordinary per-run memory high-water-marks not being returned to the OS
# between sequential runs in one process, not a specific leak.
#SBATCH --job-name=link-prediction-run
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

DATASET="$1"
FAMILY="$2"
shift 2

set +e
uv run --no-sync python -m apps.link_prediction.main \
    --dataset "${DATASET}" --family "${FAMILY}" "$@"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/link_prediction_run.sbatch" "${DATASET}" "${FAMILY}" "$@"; then
    exit 0
fi
exit "${STATUS}"
