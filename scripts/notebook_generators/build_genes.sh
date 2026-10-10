#!/usr/bin/env bash
# Build docs/notebooks/genes.ipynb from its part generators.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-.pixi/envs/default/bin/python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PARTS=()
for n in 1 2 3 4 5 6; do
  f="scripts/notebook_generators/genes_part${n}.py"
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  "$PY" "$f" "$TMP/part${n}.json"
  PARTS+=("$TMP/part${n}.json")
done
"$PY" scripts/notebook_generators/assemble.py "${PARTS[@]}" docs/notebooks/genes.ipynb
