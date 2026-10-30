#!/bin/bash
# Screening of latent families across synthetic graph structures with known edge
# probabilities (apps/synthetic/graph_families.py). Does a structure exist where TNBBeta's
# expressivity matters and vMF / power-spherical / Gaussian fail?
#
# Graph presets (4): assortative, assortative_dc (degree-corrected), bipartite (extreme
# disassortative), core_periphery. Conditions (7), every one with matched ambient dimension 16
# (Gaussian 15) and the SAME budget (500 epochs) -- the controls the fixed-kappa work showed
# are required, since free kappa is slow to learn:
#   tnbbeta_free, tnbbeta_f0.5 (epsilon fixed at m = 0.5 * (d-1)/2, unimodal),
#   vmf_free, vmf_k1 (kappa = d), power_spherical_free, power_spherical_k1, gaussian.
# Each cell is one (preset, condition) over seeds 0-4: 28 cells, 4 per task -> array 0-6.
# Results: $TNBBETA_CHECKPOINT_DIR/synthetic_graphs/gf_<preset>_<condition>.json. Safe to
# resubmit: finished cells are skipped. No --device flag (bad GPU node -> exit 75, resubmit).
#SBATCH --job-name=graph-families
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CKPT_DIR="$(uv run --no-sync python -c 'from tnbbeta_vae.paths import checkpoint_dir; print(checkpoint_dir())')"

PRESETS=(assortative assortative_dc bipartite core_periphery)
CONDITIONS=(tnbbeta_free tnbbeta_f0.5 vmf_free vmf_k1 power_spherical_free power_spherical_k1 gaussian)
DIM=16
CELLS_PER_TASK=4
NUM_CELLS=$(( ${#PRESETS[@]} * ${#CONDITIONS[@]} ))

run_cell() {
    local i="$1"
    local condition="${CONDITIONS[$((i % ${#CONDITIONS[@]}))]}"
    local preset="${PRESETS[$((i / ${#CONDITIONS[@]}))]}"
    local name="gf_${preset}_${condition}"
    local flags=() latent_dim="${DIM}" family
    case "${condition}" in
        tnbbeta_free) family=tnbbeta ;;
        tnbbeta_f0.5) family=tnbbeta; flags=(--fixed-m 3.75) ;;
        vmf_free) family=vmf ;;
        vmf_k1) family=vmf; flags=(--fixed-kappa "${DIM}") ;;
        power_spherical_free) family=power_spherical ;;
        power_spherical_k1) family=power_spherical; flags=(--fixed-kappa "${DIM}") ;;
        gaussian) family=gaussian; latent_dim=$((DIM - 1)) ;;
    esac
    echo "cell: ${preset} ${condition} (family=${family} d=${latent_dim})"
    if [[ -f "${CKPT_DIR}/synthetic_graphs/${name}.json" ]]; then
        echo "already done"
        return 0
    fi
    uv run --no-sync python -m apps.synthetic.graph_families \
        --graph "${preset}" --family "${family}" --latent-dim "${latent_dim}" \
        --epochs 500 --results-name "${name}" ${flags[@]+"${flags[@]}"}
}

LOG_DIR="slurm_logs/graph-families_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cells"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/graph_families_screen.sbatch"; then
    exit 0
fi
exit "${STATUS}"
