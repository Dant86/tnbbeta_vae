#!/bin/bash
# Does free TNBBeta's bimodality (m < 0) help on a graph its decoder can represent exactly?
# Graph: latent_kernel (P = c * exp(5 cos) on S^3; apps/synthetic/graph_families.py), fitted
# with --decoder-bias so the 1:1-negative-sampling target sigmoid(5 cos - ln M) is exactly
# representable. Conditions (13), ambient dimension 16 (Gaussian 15), 500 epochs, seeds 0-4:
#   tnbbeta_free, tnbbeta_m<M> for M in -6 -4 -2 -0.5 1 3.75 7.5 (epsilon = M + 7.5),
#   vmf_free, vmf_k1 (kappa = d), power_spherical_free, power_spherical_k1, gaussian.
# 4 cells per task -> array 0-3. Results: $TNBBETA_CHECKPOINT_DIR/synthetic_graphs/<LK_PREFIX, default lk>_<cond>.json.
# Safe to resubmit (finished cells skipped). No --device flag (bad GPU node -> exit 75).
#SBATCH --job-name=latent-kernel
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

CKPT_DIR="$(uv run --no-sync python -c 'from tnbbeta_vae.paths import checkpoint_dir; print(checkpoint_dir())')"

CONDITIONS=(tnbbeta_free tnbbeta_m-6 tnbbeta_m-4 tnbbeta_m-2 tnbbeta_m-0.5 tnbbeta_m1 tnbbeta_m3.75 tnbbeta_m7.5 vmf_free vmf_k1 power_spherical_free power_spherical_k1 gaussian)
DIM=16
PREFIX="${LK_PREFIX:-lk}"
CELLS_PER_TASK=4
NUM_CELLS=${#CONDITIONS[@]}

run_cell() {
    local condition="${CONDITIONS[$1]}"
    local name="${PREFIX}_${condition}"
    local flags=() latent_dim="${DIM}" family
    case "${condition}" in
        tnbbeta_free) family=tnbbeta ;;
        tnbbeta_m*) family=tnbbeta; flags=(--fixed-m "${condition#tnbbeta_m}") ;;
        vmf_free) family=vmf ;;
        vmf_k1) family=vmf; flags=(--fixed-kappa "${DIM}") ;;
        power_spherical_free) family=power_spherical ;;
        power_spherical_k1) family=power_spherical; flags=(--fixed-kappa "${DIM}") ;;
        gaussian) family=gaussian; latent_dim=$((DIM - 1)) ;;
    esac
    echo "cell: ${condition} (family=${family} d=${latent_dim})"
    if [[ -f "${CKPT_DIR}/synthetic_graphs/${name}.json" ]]; then
        echo "already done"
        return 0
    fi
    uv run --no-sync python -m apps.synthetic.graph_families \
        --graph latent_kernel --family "${family}" --latent-dim "${latent_dim}" \
        --epochs 500 --decoder-bias --results-name "${name}" ${flags[@]+"${flags[@]}"}
}

LOG_DIR="slurm_logs/latent-kernel_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cells"
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

if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/latent_kernel_m_sweep.sbatch"; then
    exit 0
fi
exit "${STATUS}"
