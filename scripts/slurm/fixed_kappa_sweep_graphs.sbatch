#!/bin/bash
# Fixed-kappa sweep for vMF and power-spherical on the featureless real graphs.
#
# Why: vMF / power-spherical posteriors sit at the prior on Cora and MAG-CS at TNBBeta's
# matched dimension (fixed_m_sweeps_baselines_and_parameter_probes_2026-10-30.md). TNBBeta at
# a fixed shape (fixed m) still matches its free-epsilon AUC, so this pins kappa to a ladder
# of multiples of the ambient dimension d and asks whether a fixed, well-chosen kappa lets
# vMF / power-spherical catch up. If it does, the free-kappa collapse is about learning kappa
# (init / gradient), not about the family's shape. kappa = multiple * d for multiple in
# {0.5, 1, 2, 5}.
#
# Same (lr, dropout, latent_dim, epochs) as TNBBeta's selected featureless run per dataset:
#   cora 65, citeseer 17, pubmed 17, mag_cs 33 (ambient).
#   4 datasets x 2 families x 4 multiples x 5 seeds = 160 cells, 8 per task -> array 0-19.
#
# Run names: fixedk_<dataset>_<family>_x<multiple>_seed<seed>; per-cell --results-name, so
# the shared <dataset>_nofeat_<family>.json grid-search files are never touched. Each cell
# also writes <results-name>_trajectory.json. Safe to resubmit: finished cells are skipped.
#SBATCH --job-name=fixed-kappa
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CKPT_DIR="$(uv run --no-sync python -c 'from tnbbeta_vae.paths import checkpoint_dir; print(checkpoint_dir())')"

DATASETS=(cora citeseer pubmed mag_cs)
LRS=(0.01 0.01 0.01 0.01)
DROPOUTS=(0.4 0.0 0.4 0.2)
DIMS=(65 17 17 33)
EPOCHS=(200 200 400 200)
FAMILIES=(vmf power_spherical)
MULTIPLES=(0.5 1 2 5)
SEEDS=(0 1 2 3 4)
CELLS_PER_TASK=8
NUM_CELLS=$(( ${#DATASETS[@]} * ${#FAMILIES[@]} * ${#MULTIPLES[@]} * ${#SEEDS[@]} ))

run_cell() {
    local i="$1"
    local seed="${SEEDS[$((i % ${#SEEDS[@]}))]}"
    i=$((i / ${#SEEDS[@]}))
    local multiple="${MULTIPLES[$((i % ${#MULTIPLES[@]}))]}"
    i=$((i / ${#MULTIPLES[@]}))
    local family="${FAMILIES[$((i % ${#FAMILIES[@]}))]}"
    local d="$((i / ${#FAMILIES[@]}))"
    local dataset="${DATASETS[$d]}" latent_dim="${DIMS[$d]}"
    local kappa
    kappa=$(awk -v m="${multiple}" -v d="${latent_dim}" 'BEGIN { printf "%g", m * d }')
    local run_name="fixedk_${dataset}_${family}_x${multiple}"
    echo "cell: ${dataset} ${family} kappa=${kappa} (x${multiple} d=${latent_dim}) seed=${seed}"
    if [[ -f "${CKPT_DIR}/link_prediction/${run_name}_seed${seed}_trajectory.json" ]]; then
        echo "already done"
        return 0
    fi

    uv run --no-sync python -m apps.link_prediction.main \
        --dataset "${dataset}" --family "${family}" --ignore-features \
        --lrs "${LRS[$d]}" --dropouts "${DROPOUTS[$d]}" --latent-dims "${latent_dim}" \
        --epochs "${EPOCHS[$d]}" --seeds "${seed}" --fixed-kappa "${kappa}" \
        --run-name "${run_name}" --results-name "${run_name}_seed${seed}" \
        --log-trajectory
}

LOG_DIR="slurm_logs/fixed-kappa_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cells"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/fixed_kappa_sweep_graphs.sbatch"; then
    exit 0
fi
exit "${STATUS}"
