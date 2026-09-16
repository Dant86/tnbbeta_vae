#!/usr/bin/env bash
# Runs the training entrypoint via uv, forwarding all arguments.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python -m apps.train.main "$@"
