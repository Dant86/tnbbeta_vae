#!/usr/bin/env bash
# Formats the given files with ruff, then re-stages them so the formatted
# version (not the pre-format version) is what gets committed.
set -euo pipefail

uv run ruff format -- "$@"
git add -- "$@"
