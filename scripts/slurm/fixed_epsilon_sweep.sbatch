#!/bin/bash
# Ablation: does holding TNBBeta's epsilon fixed force p and/or q to pick up its role?
#
# MNIST training saturates p near 1 with q near its floor (the q=0 special case: a cap
# at the mean direction, the same shape a von Mises-Fisher posterior always has), with
# epsilon alone doing the work of "how concentrated" (see the confidence-probe results:
# epsilon carries real class signal at high dimension, p and q carry almost none).
# fixed_epsilon removes epsilon's freedom entirely (the posterior head predicts two
# fewer outputs); this sweep trains at the paper's d=5 (ambient dimension 6, matching the
# "tnbs" convention) across the three qualitatively distinct regimes of the univariate
# TNBbeta's boundary density: epsilon < 1 (U-shaped/bimodal at the poles), = 1 (uniform
# baseline), > 1 (unimodal, tapering at the poles) -- verified locally to sample all
# three with no NaN/Inf.
#
# d=5 (not d=40) on purpose: the confidence probe showed epsilon carries almost no class
# information there already, so there is little for the model to lose by fixing it,
# making any resulting shift in p/q the cleanest possible signal.
#
#   3 epsilon values x 3 seeds = 9 tasks -> 0-8
#
# Run names use the generic "<prefix>_<model>_d<dim>_seed<seed>" pattern so the existing
# apps.eval.svae_table tooling reads them directly, treating each epsilon value as a
# "model": mnistfix_eps0p5_d5_seed0, mnistfix_eps1p0_d5_seed0, mnistfix_eps1p5_d5_seed0, ...
#   uv run python -m apps.eval.svae_table --prefix mnistfix \
#       --models eps0p5 eps1p0 eps1p5 --dims 5 --seeds 0 1 2
#   uv run python -m apps.eval.svae_table --prefix mnistfix --kind knn ...
#   uv run python -m apps.eval.svae_table --prefix mnistfix --kind confidence ...
# (svae_latitude has no seed-aggregated table; read its per-run JSON directly.)
#
# MNIST must already be downloaded. Safe to resubmit: training resumes, and finished runs
# are skipped. A task on a node with no usable GPU resubmits itself excluding that node
# (scripts/slurm/no_gpu_retry.sh).
#SBATCH --job-name=fixed-eps-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

EPSILONS=(0.5 1.0 1.5)
EPS_NAMES=(eps0p5 eps1p0 eps1p5)
SEEDS=(0 1 2)
DIM=5
LATENT_DIM=6
HIDDEN_CHANNELS=32

i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
i=$((i / ${#SEEDS[@]}))
EPSILON="${EPSILONS[$i]}"
RUN_NAME="mnistfix_${EPS_NAMES[$i]}_d${DIM}_seed${SEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (fixed_epsilon=${EPSILON})"

set +e
uv run --no-sync python -m apps.train.main \
    --model conv_tnbbeta_spherical_vae --run-name "${RUN_NAME}" --resume --dataset mnist \
    --set latent_dim="${LATENT_DIM}" --set hidden_channels="${HIDDEN_CHANNELS}" \
    --set fixed_epsilon="${EPSILON}" \
    --batch-size 64 --epochs 1000 --patience 50 --kl-warmup-epochs 100 \
    --seed "${SEED}" --num-workers "${SLURM_CPUS_PER_TASK:-4}"
STATUS=$?
set -e

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_epsilon_sweep.sbatch"; then
    exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then
    exit "${STATUS}"
fi

uv run --no-sync python -m apps.eval.svae_metrics --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.svae_knn --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

# p is still free here (only epsilon is fixed), so the default (p-based) probe stays
# meaningful; an epsilon-based probe would not (epsilon is now a constant).
uv run --no-sync python -m apps.eval.confidence_probe --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"

uv run --no-sync python -m apps.eval.svae_latitude --run-name "${RUN_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
