#!/bin/bash
# CPU test/lint job for the vMF sampler audit: AUX=pytest (full suite) or AUX=summarize.
#SBATCH --job-name=vmf-bias-aux
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
case "${AUX:?set AUX=pytest|summarize|analyze}" in
    pytest) uv run --no-sync pytest -q -p no:cacheprovider ;;
    summarize) uv run --no-sync python -m apps.synthetic.vmf_sampler_bias summarize ;;
    analyze) uv run --no-sync python -m apps.synthetic.vmf_exact_rerun analyze ;;
esac
