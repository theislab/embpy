#!/usr/bin/env bash
# Compatibility wrapper. The canonical submit launcher lives in scripts/submit/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/submit/submit_all.sh" "$@"

