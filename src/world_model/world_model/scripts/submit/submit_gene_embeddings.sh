#!/usr/bin/env bash
# =============================================================================
# submit_gene_embeddings.sh
#
# One driver to train the world model on Nadig and/or Replogle with whatever
# gene / action embedding you want.
#
# Three ways to pick the embedding (all equivalent):
#   1. EMB=genept bash .../submit_gene_embeddings.sh
#   2. uncomment exactly one EMB="..." line in the PICK block below
#   3. run with no selection (or `list`) to print every option + the exact
#      command for each, then copy/paste the one you want:
#        bash .../submit_gene_embeddings.sh list
#
# Every embedding is first attached to an AnnData copy under
# runs/_cache/action_h5ad/<dataset>_<embedding>.h5ad as
# .obsm["X_pert_<embedding>"]. Training then reads action_embedding.source =
# anndata_obsm from that file, so the model consumes exactly the same
# AnnData-aligned artifact you can inspect in notebooks. The source AnnData
# must already contain state embeddings in .obsm["${STATE_OBSM_KEY}"].
#
#   DATASET=nadig | replogle (default) | both
#
# -----------------------------------------------------------------------------
#  PICK block -- uncomment ONE line, or just use EMB=<name> on the CLI.
# -----------------------------------------------------------------------------
# EMB="genept"
# EMB="gene2vec"
# EMB="borzoi_v0"
# EMB="esm2_650M"
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# train.py appends a per-job suffix to output_dir. Chained baseline/compare
# jobs receive the logical output_dir and resolve the newest suffixed run dir
# before reading artifacts.

SELF="src/world_model/world_model/scripts/submit/submit_gene_embeddings.sh"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
LAUNCHER="${SLURM_DIR}/train_embedding.sbatch"
DATASET="${DATASET:-replogle}"          # replogle | nadig | both
# This cluster has NO default partition: every sbatch must name one.
PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
TRAIN_GPU_CONSTRAINT="${TRAIN_GPU_CONSTRAINT:-h100_80gb}"
TRAIN_GRES="${TRAIN_GRES:-gpu:h100:1}"
TRAIN_MEM="${TRAIN_MEM:-128G}"
TRAIN_CPUS="${TRAIN_CPUS:-8}"
# Chain run_baselines + compare/report (cell_eval) after each train job.
WITH_COMPARE="${WITH_COMPARE:-1}"
STATE_OBSM_KEY="${STATE_OBSM_KEY:-X_state}"
mkdir -p logs runs/_cache/action_embeddings runs/_cache/action_h5ad

matrix_value() {
    pixi run -e gpu -- python -m world_model.configs.run_matrix "$@"
}

base_cfg_for() {
    matrix_value base-config "$1"
}

h5ad_for() {
    matrix_value h5ad "$1"
}

# ---------------------------------------------------------------------------
#  CATALOG  --  name | kind | target | description
#    kind=precomputed : target is a single symbol-indexed CSV/NPZ attached to AnnData
#    kind=bio         : target is a MODEL_REGISTRY key (computed + cached)
#    kind=convert     : on disk but NOT in a loadable shape -> one-time
#                       conversion to a symbol-keyed CSV/NPZ required first
#  Any name not in this list is still accepted and treated as a bio_embedder
#  MODEL_REGISTRY key (e.g. esm2_3B, caduceus_ph_131k, hyenadna_large_1m, ...).
# ---------------------------------------------------------------------------
mapfile -t CATALOG < <(matrix_value catalog --target all)

lookup() {  # $1=name -> echoes "kind|target|desc" or returns 1
    local row n k t d
    for row in "${CATALOG[@]}"; do
        IFS='|' read -r n k t d <<<"$row"
        [[ "$n" == "$1" ]] && { echo "$k|$t|$d"; return 0; }
    done
    return 1
}

print_catalog() {
    local row n k t d
    echo "Gene / action embeddings for the world model"
    echo "  precomputed = CSV/NPZ -> adata.obsm | bio = computed+cached -> adata.obsm | convert = needs prep"
    echo
    printf "  %-24s %-11s %s\n" "NAME" "KIND" "DESCRIPTION"
    for row in "${CATALOG[@]}"; do
        IFS='|' read -r n k t d <<<"$row"
        printf "  %-24s %-11s %s\n" "$n" "$k" "$d"
    done
    echo
    echo "Submit a single-dataset run (replogle by default):"
    for row in "${CATALOG[@]}"; do
        IFS='|' read -r n k t d <<<"$row"
        [[ "$k" == convert ]] && continue
        printf "  EMB=%-24s bash %s\n" "$n" "$SELF"
    done
    echo
    echo "  # nadig instead / both:    DATASET=nadig EMB=<name> bash $SELF"
    echo "  #                          DATASET=both  EMB=<name> bash $SELF"
}

# --- selection ------------------------------------------------------------
case "${1:-}" in
    list|--list|-l|help|--help|-h) print_catalog; exit 0 ;;
esac

EMB="${EMB:-}"
if [[ -z "$EMB" ]]; then
    echo "No embedding selected (set EMB=<name>, or uncomment a line in $SELF)."
    echo
    print_catalog
    exit 0
fi

if resolved="$(lookup "$EMB")"; then
    IFS='|' read -r KIND TARGET DESC <<<"$resolved"
else
    KIND="bio"; TARGET="$EMB"
    DESC="custom MODEL_REGISTRY key (not in catalog)"
fi

if [[ "$KIND" == "convert" ]]; then
    {
        echo "ERROR: '$EMB' is on disk but NOT in a shape the AnnData attachment step can read."
        echo "       location: $TARGET"
        echo "       $DESC"
        echo
        echo "  The offline table loader (load_gene_embedding_table) accepts only:"
        echo "    * one symbol-indexed .csv  (gene symbol in column 0, comma-separated)"
        echo "    * one .npz with {symbols, embeddings} arrays"
        echo "  These sources are not that: omics/pops are ENSEMBL-keyed TSV, and the"
        echo "  STRING sets are 1322 per-anchor HDF5 files keyed by Entrez gene id"
        echo "  (an (n_proteins x dim) matrix per file -- a modelling choice is needed"
        echo "  to reduce them to one vector per gene symbol)."
        echo
        echo "  Convert once into a symbol-indexed CSV/NPZ, then add a 'precomputed'"
        echo "  catalog row pointing at the converted file. Ask me to wire the"
        echo "  converter for the one you want."
    } >&2
    exit 3
fi

EMB_PATH=""; EMB_MODEL=""
if [[ "$KIND" == "precomputed" ]]; then
    EMB_PATH="$TARGET"
    if [[ ! -f "$EMB_PATH" ]]; then
        echo "ERROR: precomputed embedding not found: $EMB_PATH" >&2
        exit 1
    fi
else
    EMB_MODEL="$TARGET"
fi

# Pick the pixi env that has the right CUDA kernels for this embedder.
# Default = `gpu` (the everything-else env). A few foundation models
# need exotic SSM / FlashAttn deps that conflict with the rest of the
# gpu stack and therefore live in their own pixi envs (see pixi.toml).
if [[ -n "$EMB_MODEL" ]]; then
    PIXI_ENV="$(matrix_value pixi-env "$EMB_MODEL")"
else
    PIXI_ENV="gpu"
fi

case "$DATASET" in
    replogle|nadig) DATASETS=("$DATASET") ;;
    both)           DATASETS=(nadig replogle) ;;
    *) echo "ERROR: DATASET must be replogle|nadig|both, got '$DATASET'" >&2; exit 2 ;;
esac

echo "Embedding: $EMB  (kind=$KIND) -- $DESC"
[[ "$KIND" == precomputed ]] && echo "  path:  $EMB_PATH"
[[ "$KIND" == bio ]]         && echo "  model: $EMB_MODEL (region=full pool=mean organism=human id=symbol)"
echo "Datasets:  ${DATASETS[*]}"
echo "State obsm: ${STATE_OBSM_KEY}"
echo

for ds in "${DATASETS[@]}"; do
    cfg="$(base_cfg_for "$ds")"
    h5ad="$(h5ad_for "$ds")"
    run_name="single_${ds}_${EMB}"
    out_dir="runs/world_model/${run_name}"
    obsm_key="X_pert_${EMB}"
    action_h5ad="runs/_cache/action_h5ad/${ds}_${EMB}.h5ad"
    action_npz="runs/_cache/action_embeddings/${ds}_${EMB}.npz"

    if [[ "$KIND" == "precomputed" ]]; then
        echo "[$ds] submitting AnnData attach (precomputed table: $EMB, pixi env: $PIXI_ENV) ..."
        attach_jid=$(sbatch --parsable \
            --job-name="wm-attach-${ds}-${EMB}" \
            --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
            --time=02:00:00 --mem=64G --cpus-per-task=4 \
            -o logs/%x_%j.out -e logs/%x_%j.err \
            --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
                export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
                pixi run -e ${PIXI_ENV} -- python -m world_model.scripts.embed_perturbations \
                    --dataset ${ds} --h5ad ${h5ad} --table ${EMB_PATH} \
                    --output ${action_npz} --output-h5ad ${action_h5ad} \
                    --obsm-key ${obsm_key}")
        echo "[$ds]   attach job: $attach_jid  -> ${action_h5ad} obsm[${obsm_key}]"
        echo "[$ds] chaining train after AnnData attach ..."
        jid=$(sbatch --parsable \
            --job-name="wm-${ds}-${EMB}" \
            --partition="$PARTITION" --qos="$QOS" \
            --gres="$TRAIN_GRES" --constraint="$TRAIN_GPU_CONSTRAINT" \
            --mem="$TRAIN_MEM" --cpus-per-task="$TRAIN_CPUS" \
            --dependency=afterok:"${attach_jid}" \
            --export=ALL,EMBPY_PIXI_ENV="${PIXI_ENV}" \
            "$LAUNCHER" "$cfg" \
            "data.h5ad_path=${action_h5ad}" \
            "data.state_obsm_key=${STATE_OBSM_KEY}" \
            "action_embedding.source=anndata_obsm" \
            "action_embedding.obsm_key=${obsm_key}" \
            "action_embedding.model_name=${EMB}" \
            "run_name=${run_name}" \
            "output_dir=${out_dir}")
        echo "[$ds]   train job: $jid  (after ${attach_jid})  -> $out_dir"
    else
        echo "[$ds] submitting AnnData attach (bio_embedder: $EMB_MODEL, pixi env: $PIXI_ENV) ..."
        # Constrain to 80GB GPUs: protein/DNA foundation models (ESM-2 650M/3B,
        # Enformer, Evo2, Caduceus, etc.) blow up on 32GB V100s with OOM. The
        # 80GB A100/H100 nodes have enough headroom for the full token stream.
        # Without explicit -o/-e, sbatch --wrap defaults to slurm-<jid>.out
        # in the cwd (project root). Force into logs/ to match the rest of
        # the pipeline and keep the workspace tidy.
        attach_jid=$(sbatch --parsable \
            --job-name="wm-attach-${ds}-${EMB}" \
            --partition="$PARTITION" --qos="$QOS" \
            --gres=gpu:1 --constraint="a100_80gb|h100_80gb" \
            --time=08:00:00 --mem=64G --cpus-per-task=8 \
            -o logs/%x_%j.out -e logs/%x_%j.err \
            --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
                export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
                export TMPDIR=\"${PROJECT_DIR}/.tmp/job-\$SLURM_JOB_ID\"; \
                mkdir -p \"\$TMPDIR\"; \
                trap 'rm -rf \"\$TMPDIR\"' EXIT; \
                pixi run -e ${PIXI_ENV} -- python -m world_model.scripts.embed_perturbations \
                    --dataset ${ds} --h5ad ${h5ad} --model ${EMB_MODEL} \
                    --output ${action_npz} --output-h5ad ${action_h5ad} \
                    --obsm-key ${obsm_key} \
                    --region full --pooling-strategy mean \
                    --organism human --id-type symbol")
        echo "[$ds]   attach job: $attach_jid  -> ${action_h5ad} obsm[${obsm_key}]"
        echo "[$ds] chaining train after AnnData attach ..."
        jid=$(sbatch --parsable \
            --job-name="wm-${ds}-${EMB}" \
            --partition="$PARTITION" --qos="$QOS" \
            --gres="$TRAIN_GRES" --constraint="$TRAIN_GPU_CONSTRAINT" \
            --mem="$TRAIN_MEM" --cpus-per-task="$TRAIN_CPUS" \
            --dependency=afterok:"${attach_jid}" \
            --export=ALL,EMBPY_PIXI_ENV="${PIXI_ENV}" \
            "$LAUNCHER" "$cfg" \
            "data.h5ad_path=${action_h5ad}" \
            "data.state_obsm_key=${STATE_OBSM_KEY}" \
            "action_embedding.source=anndata_obsm" \
            "action_embedding.obsm_key=${obsm_key}" \
            "action_embedding.model_name=${EMB_MODEL}" \
            "action_embedding.organism=human" \
            "action_embedding.id_type=symbol" \
            "action_embedding.region=full" \
            "action_embedding.pooling_strategy=mean" \
            "run_name=${run_name}" \
            "output_dir=${out_dir}")
        echo "[$ds]   train job: $jid  (after ${attach_jid})  -> $out_dir"
    fi

    if [[ "$WITH_COMPARE" == "1" ]]; then
        base_jid=$(sbatch --parsable \
            --job-name="wm-base-${ds}-${EMB}" \
            --partition="$PARTITION" --qos="$QOS" \
            --dependency=afterok:"${jid}" \
            --export=ALL,RUN_DIR="${out_dir}",EMBPY_PIXI_ENV="${PIXI_ENV}" \
            "${SLURM_DIR}/run_baselines.sbatch")
        cmp_jid=$(sbatch --parsable \
            --job-name="wm-cmp-${ds}-${EMB}" \
            --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
            --dependency=afterok:"${base_jid}" \
            --export=ALL,RUN_DIR="${out_dir}" \
            "${SLURM_DIR}/compare.sbatch")
        echo "[$ds]   baselines: $base_jid  compare/cell_eval: $cmp_jid"
        # Sweep hook: record the terminal job id so submit_embedding_sweep.sh
        # can chain a cross-embedding aggregator after every run finishes.
        [[ -n "${SWEEP_JID_FILE:-}" ]] && echo "$cmp_jid" >>"$SWEEP_JID_FILE"
    fi
    echo
done

if command -v squeue >/dev/null 2>&1; then
    squeue --me --format="%.18i %.9P %.26j %.8T %.10M %.6D %R" || true
fi
