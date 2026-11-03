#!/bin/bash
# Cost check of a p-biased TNBBeta head start on the axial (antipodal-bimodal) task: one
# array task per seed, loops over p-offset and latent_dim. Exploratory. Usage:
#   sbatch --array=0-2 scripts/slurm/axial_init_shift.sbatch
#SBATCH --job-name=axial-shift
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"
CKPT="$(grep '^TNBBETA_CHECKPOINT_DIR=' .env | cut -d= -f2-)"
OUT_DIR="${OUT_DIR:-${CKPT}/init_symmetry_sweep/axial}"
SEED="${SLURM_ARRAY_TASK_ID}"
STATUS=0
for DIM in 2 5; do
  for P in none 0.75 0.9; do
    DEST="${OUT_DIR}/seed${SEED}_dim${DIM}_p${P}"
    [[ -f "${DEST}/axial_recovery.json" ]] && continue
    set +e
    OMP_NUM_THREADS=2 PYTHONPATH=. uv run --no-sync python -m apps.synthetic.axial_init_shift "${P}" -- \
        --families tnbbeta --seed "${SEED}" --latent-dim "${DIM}" --out-dir "${DEST}" \
        > "${OUT_DIR}_seed${SEED}_dim${DIM}_p${P}.log" 2>&1
    s=$?; set -e
    if (( s == NO_GPU_EXIT_CODE )); then STATUS=$s; break 2; fi
    if (( s != 0 )); then echo "cell ${SEED}/${DIM}/${P} failed (status ${s}); continuing" >&2; fi
  done
done
if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/axial_init_shift.sbatch"; then exit 0; fi
exit "${STATUS}"
