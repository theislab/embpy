#!/usr/bin/env bash
# =============================================================================
# submit_embedding_sweep.sh
#
# Train the world model with EVERY usable gene/action embedding, run the
# baselines + cell_eval for each, then chain ONE cross-embedding aggregator
# that stitches all of them into a single comparison (world model vs every
# baseline, across every embedding).
#
# Per embedding it just calls submit/submit_gene_embeddings.sh (so the AnnData attach ->
# train -> baselines -> compare chain, partitions, and overrides are defined
# in exactly one place). The terminal compare job id of each embedding is
# collected and the aggregator is submitted afterany on all of them.
#
# Usage:
#   bash src/world_model/world_model/scripts/submit/submit_embedding_sweep.sh
#   DATASET=both bash .../submit/submit_embedding_sweep.sh
#   EMBEDDINGS="genept gene2vec borzoi_v0 esm2_650M" bash .../submit_embedding_sweep.sh
#   DRYRUN=1 bash .../submit_embedding_sweep.sh        # print plan, submit nothing
#
# Env: DATASET (replogle|nadig|both, default replogle), PARTITION/QOS,
#      CPU_PARTITION/CPU_QOS  -- all forwarded to submit_gene_embeddings.sh.
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
cd "$PROJECT_DIR"
PROJECT_DIR="${PROJECT_DIR%/}"
export PROJECT_DIR
export PATH="$HOME/.pixi/bin:$PATH"
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SLURM_EXPORT_ENV

project_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        printf "%s\n" "$PROJECT_DIR"
    elif [[ "$path" = /* ]]; then
        printf "%s\n" "$path"
    else
        printf "%s/%s\n" "$PROJECT_DIR" "$path"
    fi
}

DRIVER="src/world_model/world_model/scripts/submit/submit_gene_embeddings.sh"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
DATASET="${DATASET:-replogle}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
LOG_ROOT="${LOG_ROOT:-logs}"
SUBMIT_STAMP="${SUBMIT_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
TMP_ROOT="${TMP_ROOT:-runs/_tmp}"
LOG_ROOT="$(project_path "$LOG_ROOT")"
TMP_BASE="$(project_path "$TMP_ROOT")"
export TMPDIR="${TMPDIR:-${TMP_BASE}/submit-${SUBMIT_STAMP}}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-runs/World_Model}"
RUN_ROOT_BASE="$(project_path "$RUN_ROOT_BASE")"
SWEEP_RUN_DIR="${SWEEP_RUN_DIR:-${RUN_ROOT_BASE}/_sweep/${SUBMIT_STAMP}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/World_Model/submissions/embedding_sweep/${SUBMIT_STAMP}}"
SWEEP_RUN_DIR="$(project_path "$SWEEP_RUN_DIR")"
LOG_DIR="$(project_path "$LOG_DIR")"
SWEEP_TMP_DIR="$(project_path "runs/world_model/_sweep")"
mkdir -p "$LOG_DIR" "$SWEEP_RUN_DIR" "$SWEEP_TMP_DIR" "$TMP_BASE" "$TMPDIR"

# Source of truth for "what is usable" = the driver's own catalog. The
# `list` output prints exactly one `EMB=<name>` line per non-convert entry.
if [[ -n "${EMBEDDINGS:-}" ]]; then
    read -r -a EMB_LIST <<<"$EMBEDDINGS"
else
    mapfile -t EMB_LIST < <(
        bash "$DRIVER" list | sed -n 's/^  EMB=\([^ ]*\).*/\1/p'
    )
fi

if [[ ${#EMB_LIST[@]} -eq 0 ]]; then
    echo "No embeddings resolved to sweep." >&2
    exit 1
fi

echo "Sweep plan"
echo "  dataset(s):  $DATASET"
echo "  embeddings:  ${#EMB_LIST[@]}  ->  ${EMB_LIST[*]}"
echo "  logs:        $LOG_DIR"
echo "  run root:    $RUN_ROOT_BASE/<cell>_with_<action>/<dataset>/$SUBMIT_STAMP"
echo "  aggregate:   $SWEEP_RUN_DIR"
echo

if [[ "${DRYRUN:-0}" == "1" ]]; then
    echo "DRYRUN=1 -- would submit the above; nothing sent."
    exit 0
fi

JID_FILE="$(mktemp "${SWEEP_TMP_DIR}/jids.XXXXXX")"
: >"$JID_FILE"

for emb in "${EMB_LIST[@]}"; do
    echo "=== $emb ==="
    SWEEP_JID_FILE="$JID_FILE" DATASET="$DATASET" EMB="$emb" SUBMIT_STAMP="$SUBMIT_STAMP" \
        RUN_ROOT_BASE="$RUN_ROOT_BASE" LOG_ROOT="$LOG_ROOT" \
        bash "$DRIVER" || {
            echo "WARN: submit failed for $emb; continuing." >&2
            continue
        }
done

mapfile -t ALL_JIDS < <(grep -E '^[0-9]+$' "$JID_FILE" || true)
if [[ ${#ALL_JIDS[@]} -eq 0 ]]; then
    echo "No compare jobs were chained; aggregator not submitted." >&2
    exit 1
fi

DEP=$(IFS=:; echo "${ALL_JIDS[*]}")
DS_ARGS="${DATASET}"
[[ "$DATASET" == "both" ]] && DS_ARGS="nadig replogle"

echo
echo "Submitting cross-embedding aggregator afterany:${DEP} ..."
AGG_JID=$(sbatch --parsable \
    --chdir="$PROJECT_DIR" \
    --job-name=wm-sweep-aggregate \
    --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --export=ALL \
    -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err" \
    --dependency=afterany:"${DEP}" \
    --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
        export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
        export PYTHONNOUSERSITE=1; \
        export TMPDIR=\"${TMP_BASE}/aggregate-\${SLURM_JOB_ID:-manual}\"; \
        mkdir -p \"\$TMPDIR\"; \
        trap 'rm -rf \"\$TMPDIR\"' EXIT; \
        pixi run -e gpu -- python -m world_model.scripts.sweeps.aggregate_sweep \
            --runs-root ${RUN_ROOT_BASE} --datasets ${DS_ARGS} --out ${SWEEP_RUN_DIR}")

echo
echo "Submitted sweep:"
printf "  per-embedding compare jobs: %s\n" "${ALL_JIDS[*]}"
printf "  aggregator:                 %s\n" "$AGG_JID"
echo "  results -> ${SWEEP_RUN_DIR}/{comparison_all.csv,world_model_by_embedding.csv,report.md}"
echo
if command -v squeue >/dev/null 2>&1; then
    squeue --me --format="%.18i %.9P %.26j %.8T %.10M %.6D %R" || true
fi
