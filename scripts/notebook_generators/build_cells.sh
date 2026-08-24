#!/usr/bin/env bash
# Build docs/notebooks/cells.ipynb from its part generators.
#
# The generators import nothing from embpy, so any Python 3.11+ interpreter
# works. The default deliberately avoids the project environment: embpy's
# esm3 and helical extras have mutually unsatisfiable pins, so `uv run` on
# the project fails to resolve. `--no-project` sidesteps that.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-uv run --no-project python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PARTS=()
for n in 1 2 3 4 5 6; do
  f="scripts/notebook_generators/cells_part${n}.py"
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  $PY "$f" "$TMP/part${n}.json"
  PARTS+=("$TMP/part${n}.json")
done
$PY scripts/notebook_generators/assemble.py "${PARTS[@]}" docs/notebooks/cells.ipynb
