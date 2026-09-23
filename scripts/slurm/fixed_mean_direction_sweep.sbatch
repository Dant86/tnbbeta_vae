#!/bin/bash
# Ablation: with mean direction fixed at the pole e_1 for every example, does TNBBeta's
# extra expressivity (p, q, epsilon -- three scalars beyond a location parameter) carry
# usable per-example information, now that it is the *only* channel available?
#
# Motivation (see writeup/results/tnbbeta_vmf_geometric_convergence_2026-09-23.md): free
# training saturates p near 1 with q near 0 -- a tight cap at the mean direction -- and the
# fixed-epsilon ablation showed that even when epsilon's freedom is removed, q rebuilds the
# exact same cap rather than a genuinely different (ring/bimodal) shape. Across every
# metric so far, direction alone appears to carry essentially all discriminative signal;
# p/q/epsilon never have to. Fixing direction removes that escape hatch entirely, so any
# per-example signal in z has to come from the shape (p, q, epsilon) alone.
#
# TNBBeta only (comparing a mu-fixed vMF here would only have kappa, one scalar, doing
# fewer things than a Gaussian VAE with fixed mean would have via its per-dimension sigma
# -- an unmatched comparison not worth running). d=5 (ambient 6), matching the "tnbs"
# convention and its already-collected free-mu baseline (tnbs_d5_seed{0..4}) for
# comparison. 3 seeds, small on purpose: a first look, not a full sweep.
#
#   3 seeds -> 0-2
#
# Run names use the generic "<prefix>_<model>_d<dim>_seed<seed>" pattern so the existing
# apps.eval.svae_table tooling reads them directly (as a single-"model" table):
#   uv run python -m apps.eval.svae_table --prefix mnistfixmu --models tnb --dims 5 --seeds 0 1 2
#   uv run python -m apps.eval.svae_table --prefix mnistfixmu --models tnb --dims 5 --seeds 0 1 2 --kind knn
#   uv run python -m apps.eval.svae_table --prefix mnistfixmu --models tnb --dims 5 --seeds 0 1 2 --kind confidence
# Table 2's direction k-NN is expected to sit at chance (1/10): direction is now identical
# for every example by construction, so it carries no information at all. The confidence
# probe (p, or --param epsilon) is the metric that actually answers the question here.
# svae_latitude has no seed-aggregated table; read its per-run JSON directly.
#
# MNIST must already be downloaded. Safe to resubmit: training resumes, and finished runs
# are skipped. A task on a node with no usable GPU resubmits itself excluding that node
# (scripts/slurm/no_gpu_retry.sh).
#SBATCH --job-name=fixed-mu-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

SEEDS=(0 1 2)
DIM=5
LATENT_DIM=6
HIDDEN_CHANNELS=32

SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="mnistfixmu_tnb_d${DIM}_seed${SEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (fixed_mean_direction=true)"

set +e
uv run --no-sync python -m apps.train.main \
    --model conv_tnbbeta_spherical_vae --run-name "${RUN_NAME}" --resume --dataset mnist \
    --set latent_dim="${LATENT_DIM}" --set hidden_channels="${HIDDEN_CHANNELS}" \
    --set fixed_mean_direction=true \
    --batch-size 64 --epochs 1000 --patience 50 --kl-warmup-epochs 100 \
    --seed "${SEED}" --num-workers "${SLURM_CPUS_PER_TASK:-4}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_mean_direction_sweep.sbatch"; then
    exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then
    exit "${STATUS}"
fi

uv run --no-sync python -m apps.eval.svae_metrics --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.svae_knn --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.confidence_probe --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.confidence_probe --run-name "${RUN_NAME}" \
    --param epsilon --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.svae_latitude --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
