#!/bin/bash
# Fixed-m sweep of TNBBeta on the real graphs, FEATURELESS, plus a per-bucket
# decomposition (by training degree, and by number of neighbor communities on mag_cs).
#
# The SBM sweep (fixed_m_sweep_sbm.sbatch) showed AUC is flat for m in [-1.5, 2] and
# that a free-epsilon TNBBeta lands at m ~ -4.8 there, but the SBM does not show TNBBeta
# beating vMF, so it cannot say whether m < 0 matters where the real-data advantage lives.
#
# Each dataset runs at TNBBeta's own selected (lr, dropout, latent_dim) from
# $TNBBETA_CHECKPOINT_DIR/link_prediction/<dataset>_nofeat_tnbbeta.json, so the free-epsilon
# condition reproduces that run. The threshold is (latent_dim - 1) / 2 -- 32, 8, 8, 16 --
# so the grid is in FRACTIONS f of it: m = f * (latent_dim - 1) / 2, epsilon = threshold *
# (1 + f) > 0:
#   f in {-0.9, -0.75, -0.6, -0.4, -0.2, 0, 0.2, 0.5, 1}  plus "free"
# (the SBM's learned m ~ -4.8 at threshold 7.5 is f ~ -0.64).
#
#   4 datasets x 10 conditions x 5 seeds = 200 cells. The general QOS allows 200 submitted
#   and only 24 RUNNING jobs per user, so cells are packed CELLS_PER_TASK = 8 to a task and
#   run as parallel processes sharing the task's one GPU: 25 tasks -> 0-24, up to ~190
#   cells in flight at once.
#
# No --device flag on purpose: with none given, a Slurm GPU that CUDA cannot start exits 75,
# which resubmits the task excluding that node. An explicit --device cuda skips that check
# (a bad node then fails every task in seconds and eats the whole array).
# Run names: fixedmg_<dataset>_<cond>_seed<seed>. After training, each task writes
# bucketed_link_prediction.json beside its checkpoint (apps.eval.bucketed_link_prediction;
# its "overall" entry equals the run's own test AUC/AP). Safe to resubmit. A task on a node
# with no usable GPU resubmits itself excluding that node (scripts/slurm/no_gpu_retry.sh).
#SBATCH --job-name=fixed-m-graphs
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-24
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CKPT_DIR="$(uv run --no-sync python -c 'from tnbbeta_vae.paths import checkpoint_dir; print(checkpoint_dir())')"

# Per-dataset TNBBeta selection (featureless): lr, dropout, latent_dim, epochs.
DATASETS=(cora citeseer pubmed mag_cs)
LRS=(0.01 0.01 0.01 0.01)
DROPOUTS=(0.4 0.0 0.4 0.2)
DIMS=(65 17 17 33)
EPOCHS=(200 200 400 200)
CONDITIONS=(free -0.9 -0.75 -0.6 -0.4 -0.2 0 0.2 0.5 1)
SEEDS=(0 1 2 3 4)
CELLS_PER_TASK=8
NUM_CELLS=$(( ${#DATASETS[@]} * ${#CONDITIONS[@]} * ${#SEEDS[@]} ))

# Trains one cell and writes its bucketed decomposition; its output goes to its own log.
run_cell() {
    local i="$1"
    local seed="${SEEDS[$((i % ${#SEEDS[@]}))]}"
    i=$((i / ${#SEEDS[@]}))
    local cond="${CONDITIONS[$((i % ${#CONDITIONS[@]}))]}"
    local d="$((i / ${#CONDITIONS[@]}))"
    local dataset="${DATASETS[$d]}" latent_dim="${DIMS[$d]}"
    local run_name="fixedmg_${dataset}_${cond}"
    echo "cell: ${dataset} cond=${cond} seed=${seed} latent_dim=${latent_dim}"
    # Finished on an earlier submission: nothing to redo.
    if [[ -f "${CKPT_DIR}/${run_name}_seed${seed}/bucketed_link_prediction.json" ]]; then
        echo "already done"
        return 0
    fi

    local m_args=()
    if [[ "${cond}" != "free" ]]; then
        local m
        m="$(awk -v f="${cond}" -v d="${latent_dim}" 'BEGIN{printf "%.6g", f * (d - 1) / 2}')"
        echo "fixed m=${m}"
        m_args=(--fixed-m "${m}")
    fi

    uv run --no-sync python -m apps.link_prediction.main \
        --dataset "${dataset}" --family tnbbeta --ignore-features \
        --lrs "${LRS[$d]}" --dropouts "${DROPOUTS[$d]}" --latent-dims "${latent_dim}" \
        --epochs "${EPOCHS[$d]}" --seeds "${seed}" --run-name "${run_name}" \
        --results-name "${run_name}_seed${seed}" "${m_args[@]}"
    uv run --no-sync python -m apps.eval.bucketed_link_prediction \
        --dataset "${dataset}" --run-name "${run_name}" --seed "${seed}" --ignore-features
}

LOG_DIR="slurm_logs/fixed-m-graphs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cells"
mkdir -p "${LOG_DIR}"
PIDS=()
CELLS=()
for ((k = 0; k < CELLS_PER_TASK; k++)); do
    cell=$(( SLURM_ARRAY_TASK_ID * CELLS_PER_TASK + k ))
    if (( cell >= NUM_CELLS )); then break; fi
    run_cell "${cell}" > "${LOG_DIR}/cell_${cell}.log" 2>&1 &
    PIDS+=($!)
    CELLS+=("${cell}")
done

# Wait for every cell, remembering the worst status (a no-GPU node fails them all at once).
STATUS=0
for idx in "${!PIDS[@]}"; do
    set +e
    wait "${PIDS[$idx]}"
    cell_status=$?
    set -e
    if (( cell_status != 0 )); then
        echo "cell ${CELLS[$idx]} failed with status ${cell_status}" >&2
        if (( cell_status == NO_GPU_EXIT_CODE || STATUS == 0 )); then STATUS=${cell_status}; fi
    fi
done

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_m_sweep_graphs.sbatch"; then
    exit 0
fi
exit "${STATUS}"
