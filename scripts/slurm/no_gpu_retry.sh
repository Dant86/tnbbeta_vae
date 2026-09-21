#!/bin/bash
# Sourced by the array scripts. If a task's training step exited with the "no usable GPU"
# code (apps.train.main exits 75 when Slurm allocated a GPU but CUDA fails to start),
# resubmit just that array task, excluding the node it ran on, up to MAX_ATTEMPTS times.
#
#   resubmit_if_no_gpu STATUS SCRIPT
#     returns 0 after resubmitting (the caller should then `exit 0`), and 1 when STATUS is
#     not the no-GPU code (the caller handles it). Exits 1 itself once attempts run out.

NO_GPU_EXIT_CODE=75
MAX_ATTEMPTS=5

resubmit_if_no_gpu() {
    local status="$1" script="$2"
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

    echo "No usable GPU on ${SLURMD_NODENAME}; resubmitting task ${SLURM_ARRAY_TASK_ID}" \
        "(attempt $((attempt + 1)) of ${MAX_ATTEMPTS}) excluding ${new_exclude}" >&2
    sbatch --exclude="${new_exclude}" --array="${SLURM_ARRAY_TASK_ID}" \
        --export="ALL,TNB_ATTEMPT=$((attempt + 1))" "${script}"
    return 0
}
