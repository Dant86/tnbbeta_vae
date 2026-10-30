#!/bin/bash
# Downloads DTD into $TNBBETA_DATA_DIR (set in .env). Run once.
# Submit from the repo root:  sbatch scripts/slurm/download_dtd.sbatch
# NOTE: needs outbound network access. If compute nodes have none, run
#   uv run python -m apps.data.download_dtd
# directly on the login node instead (a ~600 MB download is short).
#SBATCH --job-name=dtd-download
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --no-sync python -m apps.data.download_dtd
