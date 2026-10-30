#!/bin/bash
# Training-dynamics runs for the free-kappa question: why do vMF / power-spherical stay at the
# prior on Cora and MAG-CS? Every cell logs a per-epoch trajectory (kappa mean/median/min/max,
# r_bar, loss terms, gradient norm of the kappa head row vs. the direction rows, validation
# AUC, and the untrained initialization as epoch -1) via --log-trajectory.
#
# Conditions (6) x datasets (4) x seeds (5) = 120 cells, 8 per task -> array 0-14:
#   matched_{vmf,power_spherical,tnbbeta,gaussian}: TNBBeta's selected featureless
#       (lr, dropout, epochs) and dimension per dataset (cora 65, citeseer 17, pubmed 17,
#       mag_cs 33 ambient; Gaussian gets d - 1), as in fixed_m_baselines_graphs.sbatch;
#   own_{vmf,power_spherical}: each family's OWN selected featureless Figure 5 configuration
#       (read from link_prediction/<dataset>_nofeat_<family>.json), to check that the
#       matched-dimension collapse is not just an artifact of running vMF away from the
#       dimension it selects (vMF selects ambient 9 everywhere).
#
# Results are deterministic given the seed, so the matched cells reproduce the earlier
# fixedmgbase_* / fixedmg_free runs; the trajectory logging uses a forked RNG.
# Run names: kdyn_<condition>_<dataset>_seed<seed>. Safe to resubmit.
#SBATCH --job-name=kappa-dynamics
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-14
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CKPT_DIR="$(uv run --no-sync python -c 'from tnbbeta_vae.paths import checkpoint_dir; print(checkpoint_dir())')"

DATASETS=(cora citeseer pubmed mag_cs)
CONDITIONS=(matched_vmf matched_power_spherical matched_tnbbeta matched_gaussian own_vmf own_power_spherical)
SEEDS=(0 1 2 3 4)
CELLS_PER_TASK=8
NUM_CELLS=$(( ${#DATASETS[@]} * ${#CONDITIONS[@]} * ${#SEEDS[@]} ))

# Per-condition, per-dataset "lr dropout latent_dim epochs" (datasets in DATASETS order).
configuration() {
    case "$1:$2" in
        matched_*:cora) echo "0.01 0.4 65 200" ;;
        matched_*:citeseer) echo "0.01 0.0 17 200" ;;
        matched_*:pubmed) echo "0.01 0.4 17 400" ;;
        matched_*:mag_cs) echo "0.01 0.2 33 200" ;;
        own_vmf:cora) echo "0.01 0.2 9 200" ;;
        own_vmf:citeseer) echo "0.01 0.0 9 200" ;;
        own_vmf:pubmed) echo "0.01 0.0 9 400" ;;
        own_vmf:mag_cs) echo "0.01 0.0 9 200" ;;
        own_power_spherical:cora) echo "0.001 0.4 65 200" ;;
        own_power_spherical:citeseer) echo "0.005 0.2 65 200" ;;
        own_power_spherical:pubmed) echo "0.01 0.0 9 400" ;;
        own_power_spherical:mag_cs) echo "0.01 0.4 65 200" ;;
    esac
}

run_cell() {
    local i="$1"
    local seed="${SEEDS[$((i % ${#SEEDS[@]}))]}"
    i=$((i / ${#SEEDS[@]}))
    local condition="${CONDITIONS[$((i % ${#CONDITIONS[@]}))]}"
    local dataset="${DATASETS[$((i / ${#CONDITIONS[@]}))]}"
    local family="${condition#matched_}"
    family="${family#own_}"
    read -r lr dropout latent_dim epochs <<< "$(configuration "${condition}" "${dataset}")"
    if [[ "${condition}" == "matched_gaussian" ]]; then latent_dim=$((latent_dim - 1)); fi
    local run_name="kdyn_${condition}_${dataset}"
    echo "cell: ${condition} ${dataset} seed=${seed} lr=${lr} dropout=${dropout} d=${latent_dim} epochs=${epochs}"
    if [[ -f "${CKPT_DIR}/link_prediction/${run_name}_seed${seed}_trajectory.json" ]]; then
        echo "already done"
        return 0
    fi

    uv run --no-sync python -m apps.link_prediction.main \
        --dataset "${dataset}" --family "${family}" --ignore-features \
        --lrs "${lr}" --dropouts "${dropout}" --latent-dims "${latent_dim}" \
        --epochs "${epochs}" --seeds "${seed}" \
        --run-name "${run_name}" --results-name "${run_name}_seed${seed}" \
        --log-trajectory
}

LOG_DIR="slurm_logs/kappa-dynamics_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cells"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/kappa_dynamics_graphs.sbatch"; then
    exit 0
fi
exit "${STATUS}"
