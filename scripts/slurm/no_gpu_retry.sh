#!/bin/bash
# Sourced by the sbatch scripts. If a job's training step exited with the node-problem code
# (apps.train.main exits 75 when Slurm allocated a GPU but CUDA fails to start, or when the
# node cannot read the shared storage),
# resubmit it excluding the node it ran on, up to MAX_ATTEMPTS times: just that array task
# inside an array, or the whole job with its arguments otherwise.
#
#   resubmit_if_no_gpu STATUS SCRIPT [SCRIPT_ARGS...]
#     returns 0 after resubmitting (the caller should then `exit 0`), and 1 when STATUS is
#     not the no-GPU code (the caller handles it). Exits 1 itself once attempts run out.

NO_GPU_EXIT_CODE=75
MAX_ATTEMPTS=5

resubmit_if_no_gpu() {
    local status="$1" script="$2"
    shift 2
    if [[ "${status}" -ne "${NO_GPU_EXIT_CODE}" ]]; then
        return 1
    fi

    local attempt="${TNB_ATTEMPT:-1}"
    if (( attempt >= MAX_ATTEMPTS )); then
        echo "No usable GPU after ${attempt} attempts; giving up." >&2
        exit 1
    fi

    # Keep whatever this task already excluded (e.g. from --exclude at submit time).
    local excluded new_exclude
    excluded="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p')"
    if [[ "${excluded}" == "(null)" ]]; then
        excluded=""
    fi
    new_exclude="${excluded:+${excluded},}${SLURMD_NODENAME}"

    local what="" array_flag=()
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        what=" task ${SLURM_ARRAY_TASK_ID}"
        array_flag=(--array="${SLURM_ARRAY_TASK_ID}")
    fi
    echo "Node problem on ${SLURMD_NODENAME} (no usable GPU or unreadable storage);" \
        "resubmitting${what}" \
        "(attempt $((attempt + 1)) of ${MAX_ATTEMPTS}) excluding ${new_exclude}" >&2
    # The +-expansion keeps an empty array valid under `set -u` on older bash.
    sbatch --exclude="${new_exclude}" ${array_flag[@]+"${array_flag[@]}"} \
        --export="ALL,TNB_ATTEMPT=$((attempt + 1))" "${script}" "$@"
    return 0
}
