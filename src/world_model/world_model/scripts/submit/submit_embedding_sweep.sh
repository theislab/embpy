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

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

DRIVER="src/world_model/world_model/scripts/submit/submit_gene_embeddings.sh"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
DATASET="${DATASET:-replogle}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
mkdir -p logs runs/world_model/_sweep

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
echo

if [[ "${DRYRUN:-0}" == "1" ]]; then
    echo "DRYRUN=1 -- would submit the above; nothing sent."
    exit 0
fi

JID_FILE="$(mktemp runs/world_model/_sweep/jids.XXXXXX)"
: >"$JID_FILE"

for emb in "${EMB_LIST[@]}"; do
    echo "=== $emb ==="
    SWEEP_JID_FILE="$JID_FILE" DATASET="$DATASET" EMB="$emb" \
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
    --job-name=wm-sweep-aggregate \
    --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    -o logs/%x_%j.out -e logs/%x_%j.err \
    --dependency=afterany:"${DEP}" \
    --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
        export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
        pixi run -e gpu -- python -m world_model.scripts.sweeps.aggregate_sweep \
            --datasets ${DS_ARGS} --out runs/world_model/_sweep")

echo
echo "Submitted sweep:"
printf "  per-embedding compare jobs: %s\n" "${ALL_JIDS[*]}"
printf "  aggregator:                 %s\n" "$AGG_JID"
echo "  results -> runs/world_model/_sweep/{comparison_all.csv,world_model_by_embedding.csv,report.md}"
echo
if command -v squeue >/dev/null 2>&1; then
    squeue --me --format="%.18i %.9P %.26j %.8T %.10M %.6D %R" || true
fi
