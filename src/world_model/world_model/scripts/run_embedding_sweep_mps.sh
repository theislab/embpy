#!/usr/bin/env bash
# Compatibility wrapper. The canonical local launcher lives in scripts/local/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/local/run_embedding_sweep_mps.sh" "$@"

