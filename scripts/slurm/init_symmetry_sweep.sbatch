#!/bin/bash
# TNBBeta init-symmetry sweep (apps/synthetic/init_symmetry_sweep.py), exploratory. Usage:
#   sbatch --array=0-7 scripts/slurm/init_symmetry_sweep.sbatch
# Each array task runs PROCS workers sharing its GPU; worker p of task t takes shard
# (t * PROCS + p) of (array size * PROCS). Per-shard JSONL in $OUT_DIR (resumable; a
# finished cell is skipped on resubmission). No --device flag: select_device exits 75 on a
# node without a usable GPU and the script resubmits itself.
#SBATCH --job-name=init-sym
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/no_gpu_retry.sh"

PROCS="${PROCS:-6}"
export TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_ARRAY_TASK_COUNT}}"
NUM_SHARDS=$(( TOTAL_TASKS * PROCS ))
CKPT="$(grep '^TNBBETA_CHECKPOINT_DIR=' .env | cut -d= -f2-)"
OUT_DIR="${OUT_DIR:-${CKPT}/init_symmetry_sweep}"
mkdir -p "${OUT_DIR}"
CONDS="default,p0.6,p0.75,p0.9,eps2,eps8.5,q0.1,q0.9,w10,w30,p0.58,p0.75q0.25,p0.75eps2,p0.75eps8.5,vmf_init,vmf_default"

PIDS=()
for ((p = 0; p < PROCS; p++)); do
    shard=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run --no-sync python \
        -m apps.synthetic.init_symmetry_sweep "${OUT_DIR}/sweep_shard${shard}.jsonl" \
        assortative,bipartite,ring 0.003,tuned 0,1,2 "${CONDS}" "${shard}" "${NUM_SHARDS}" \
        > "${OUT_DIR}/shard_${shard}.log" 2>&1 &
    PIDS+=($!)
done

STATUS=0
for pid in "${PIDS[@]}"; do
    set +e; wait "${pid}"; worker_status=$?; set -e
    if (( worker_status != 0 )); then
        echo "a worker failed with status ${worker_status}" >&2
        if (( worker_status == NO_GPU_EXIT_CODE || STATUS == 0 )); then STATUS=${worker_status}; fi
    fi
done
if resubmit_if_no_gpu "${STATUS}" "${SLURM_SUBMIT_DIR}/scripts/slurm/init_symmetry_sweep.sbatch"; then
    exit 0
fi
exit "${STATUS}"
