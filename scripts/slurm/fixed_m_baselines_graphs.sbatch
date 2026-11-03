#!/bin/bash
# Matched-dimension baselines for fixed_m_sweep_graphs.sbatch: vMF, power-spherical and
# Gaussian on the same featureless graphs, edge splits (split_edges is a function of the
# seed) and per-bucket evaluation as the TNBBeta cells.
#
# Why: each family's selected latent_dim differs wildly in the featureless Figure 5 runs
# (vMF picks d=8, the grid's floor, on all four datasets; TNBBeta picks 16-64), so "TNBBeta
# beats vMF" is confounded with dimension. Here every family gets TNBBeta's own selected
# (lr, dropout, epochs) and dimension per dataset, so only the family differs:
#   cora 65, citeseer 17, pubmed 17, mag_cs 33 (ambient, for the sphere families); the
#   Gaussian gets latent_dim - 1, the same manifold dimension (the "d" convention).
#
#   4 datasets x 3 families x 5 seeds = 60 cells, packed 8 per task -> 8 tasks -> 0-7
#   (see fixed_m_sweep_graphs.sbatch for why: 24 running / 200 submitted job QOS cap).
#
# Run names: fixedmgbase_<dataset>_<family>_seed<seed>. Each cell writes its summary to
# its own --results-name, never the shared <dataset>_nofeat_<family>.json (those hold the
# full grid searches behind Figure 5). Safe to resubmit: finished cells are skipped. No
# --device flag, so a bad GPU node exits 75 and the task resubmits excluding it.
#SBATCH --job-name=fixed-m-baselines
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CKPT_DIR="$(uv run --no-sync python -c 'from tnbbeta_vae.paths import checkpoint_dir; print(checkpoint_dir())')"

# Per-dataset TNBBeta selection (featureless): lr, dropout, ambient latent_dim, epochs.
DATASETS=(cora citeseer pubmed mag_cs)
LRS=(0.01 0.01 0.01 0.01)
DROPOUTS=(0.4 0.0 0.4 0.2)
DIMS=(65 17 17 33)
EPOCHS=(200 200 400 200)
FAMILIES=(vmf power_spherical gaussian)
SEEDS=(0 1 2 3 4)
CELLS_PER_TASK=8
NUM_CELLS=$(( ${#DATASETS[@]} * ${#FAMILIES[@]} * ${#SEEDS[@]} ))

run_cell() {
    local i="$1"
    local seed="${SEEDS[$((i % ${#SEEDS[@]}))]}"
    i=$((i / ${#SEEDS[@]}))
    local family="${FAMILIES[$((i % ${#FAMILIES[@]}))]}"
    local d="$((i / ${#FAMILIES[@]}))"
    local dataset="${DATASETS[$d]}" latent_dim="${DIMS[$d]}"
    if [[ "${family}" == "gaussian" ]]; then latent_dim=$((latent_dim - 1)); fi
    local run_name="fixedmgbase_${dataset}_${family}"
    echo "cell: ${dataset} ${family} seed=${seed} latent_dim=${latent_dim}"
    if [[ -f "${CKPT_DIR}/${run_name}_seed${seed}/bucketed_link_prediction.json" ]]; then
        echo "already done"
        return 0
    fi

    uv run --no-sync python -m apps.link_prediction.main \
        --dataset "${dataset}" --family "${family}" --ignore-features \
        --lrs "${LRS[$d]}" --dropouts "${DROPOUTS[$d]}" --latent-dims "${latent_dim}" \
        --epochs "${EPOCHS[$d]}" --seeds "${seed}" --run-name "${run_name}" \
        --results-name "${run_name}_seed${seed}"
    uv run --no-sync python -m apps.eval.bucketed_link_prediction \
        --dataset "${dataset}" --run-name "${run_name}" --seed "${seed}" --ignore-features
}

LOG_DIR="slurm_logs/fixed-m-baselines_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cells"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_m_baselines_graphs.sbatch"; then
    exit 0
fi
exit "${STATUS}"
