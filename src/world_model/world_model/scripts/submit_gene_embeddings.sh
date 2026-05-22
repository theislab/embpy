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
# precomputed embedding -> training job submitted directly.
# bio_embedder embedding -> a pre-warm job (embed_perturbations) is submitted
#   first and training is chained after it with afterok, so GPU time is not
#   wasted recomputing embeddings during epoch 1 (same pattern as submit_all.sh).
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

# Sweep submissions must have STABLE output_dirs so that the chained
# baselines + compare jobs (which look up RUN_DIR by exact path) can
# find the train artifacts. train.py's _apply_auto_suffix would
# otherwise rewrite output_dir to ``${out_dir}__job<id>__<ts>``,
# breaking the dependency chain. Per-job uniqueness is already
# guaranteed by the (dataset, embedding) keying in run_name.
export EMBPY_NO_AUTO_SUFFIX=1

SELF="src/world_model/world_model/scripts/submit_gene_embeddings.sh"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
CFG_DIR="src/world_model/world_model/configs/experiments"
LAUNCHER="${SLURM_DIR}/train_embedding.sbatch"
GE="data/embeddings/gene_embeddings"
DATASET="${DATASET:-replogle}"          # replogle | nadig | both
# This cluster has NO default partition: every sbatch must name one.
PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
# Chain run_baselines + compare/report (cell_eval) after each train job.
WITH_COMPARE="${WITH_COMPARE:-1}"
mkdir -p logs

declare -A BASE_CFG H5AD
BASE_CFG[replogle]="${CFG_DIR}/single_replogle.yaml"
BASE_CFG[nadig]="${CFG_DIR}/single_nadig.yaml"
H5AD[replogle]="data/datasets/replogle/replogle_2022_k562_essential.h5ad"
H5AD[nadig]="data/datasets/nadig/NadigOConner2024_jurkat.h5ad"

# ---------------------------------------------------------------------------
#  CATALOG  --  name | kind | target | description
#    kind=precomputed : target is a single symbol-indexed CSV (loads directly)
#    kind=bio         : target is a MODEL_REGISTRY key (computed + cached)
#    kind=convert     : on disk but NOT in a loadable shape -> one-time
#                       conversion to a symbol-keyed CSV/NPZ required first
#  Any name not in this list is still accepted and treated as a bio_embedder
#  MODEL_REGISTRY key (e.g. esm2_3B, caduceus_ph_131k, hyenadna_large_1m, ...).
# ---------------------------------------------------------------------------
CATALOG=(
  "genept|precomputed|$GE/genept/embeddings_3072.csv|GenePT GPT-3.5 text emb, 3072d (symbol)"
  "genept_scaled|precomputed|$GE/genept/scaled/embeddings_3072.csv|GenePT z-scored, 3072d"
  "gene2vec|precomputed|$GE/gene2vec/embeddings_d200.csv|Gene2Vec co-expression, 200d"
  "wikicrow|precomputed|$GE/wikicrow/scaled/embeddings_4096.csv|WikiCrow text emb, 4096d"
  "ccle|precomputed|$GE/ccle/expression_1270_symbol.csv|CCLE expression, 1270d (symbol)"
  "ccle_ensembl|precomputed|$GE/ccle/expression_300_ensemblid.csv|CCLE expr 300d (ENSEMBL keyed*)"
  "crispr_gene_effect|precomputed|$GE/crispr_gene_effect/gene_effect.csv|DepMap CRISPR effect, full"
  "crispr_gene_effect_1178|precomputed|$GE/crispr_gene_effect/scaled/gene_effect_1178.csv|DepMap CRISPR effect, 1178d"
  "crispr_gene_effect_205|precomputed|$GE/crispr_gene_effect/scaled/gene_effect_205.csv|DepMap CRISPR effect, 205d"
  "borzoi_v0|bio|borzoi_v0|Borzoi rep-0 DNA, 1536d (cache: full_mean_human)"
  "borzoi_v1|bio|borzoi_v1|Borzoi rep-1 DNA"
  "enformer_human_rough|bio|enformer_human_rough|Enformer (human only)"
  "caduceus_ph_131k|bio|caduceus_ph_131k|Caduceus-PH 131k"
  "caduceus_ps_131k|bio|caduceus_ps_131k|Caduceus-PS 131k"
  "gena_lm_bert_base|bio|gena_lm_bert_base|GENA-LM BERT base"
  "gena_lm_bert_large|bio|gena_lm_bert_large|GENA-LM BERT large"
  "gena_lm_bigbird_base|bio|gena_lm_bigbird_base|GENA-LM BigBird base"
  "hyenadna_large_1m|bio|hyenadna_large_1m|HyenaDNA large 1m"
  "nt_v2_500m|bio|nt_v2_500m|Nucleotide Transformer v2 500m"
  "evo2_7b|bio|evo2_7b|Evo2 7B (needs embpy[evo2])"
  "esm2_650M|bio|esm2_650M|ESM-2 650M protein"
  "esm2_3B|bio|esm2_3B|ESM-2 3B protein"
  "esmc_600m|bio|esmc_600m|ESM-C 600M protein"
  "minilm_l6_v2|bio|minilm_l6_v2|MiniLM-L6-v2 text"
  "omics|convert|$GE/omics/embeddings_d256.tsv|omics 256d -- ENSEMBL-keyed TSV (convert first)"
  "pops|convert|$GE/pops/features_d256.tsv|PoPS 256d -- ENSEMBL-keyed TSV (convert first)"
  "string_functional|convert|data/embeddings/precomputed_embeddings_string/functional_embeddings/functional_emb|STRING functional: 1322 per-anchor Entrez .h5 (~11156x512) (reduce first)"
  "string_node2vec|convert|data/embeddings/precomputed_embeddings_string/node2vec/node2vec|STRING node2vec: 1322 per-anchor Entrez .h5 (~9678x128) (reduce first)"
)

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
    echo "  precomputed = loads directly | bio = computed+cached | convert = needs prep"
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
    echo
    echo "Transfer learning across datasets (pretrain Nadig -> fine-tune Replogle):"
    echo "  bash src/world_model/world_model/scripts/submit_transfer_borzoi.sh            # Borzoi, 10% Replogle"
    echo "  FRACTION=0.05 bash src/world_model/world_model/scripts/submit_transfer_borzoi.sh   # vary fine-tune %"
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
        echo "ERROR: '$EMB' is on disk but NOT in a shape the precomputed provider can read."
        echo "       location: $TARGET"
        echo "       $DESC"
        echo
        echo "  The precomputed loader (load_gene_embedding_table) accepts only:"
        echo "    * one symbol-indexed .csv  (gene symbol in column 0, comma-separated)"
        echo "    * one .npz with {symbols, embeddings} arrays"
        echo "  These sources are not that: omics/pops are ENSEMBL-keyed TSV, and the"
        echo "  STRING sets are 1322 per-anchor HDF5 files keyed by Entrez gene id"
        echo "  (an (n_proteins x dim) matrix per file -- a modelling choice is needed"
        echo "  to reduce them to one vector per gene symbol)."
        echo
        echo "  Convert once into a symbol-indexed CSV, then add a 'precomputed'"
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
declare -A PIXI_ENV_BY_MODEL
PIXI_ENV_BY_MODEL[caduceus_ph_131k]=caduceus
PIXI_ENV_BY_MODEL[caduceus_ps_131k]=caduceus
PIXI_ENV_BY_MODEL[evo2_7b]=evo2
# Precomputed embeddings have no model name; guard the array lookup so an
# empty key doesn't trip `set -u` ("bad array subscript").
if [[ -n "$EMB_MODEL" ]]; then
    PIXI_ENV="${PIXI_ENV_BY_MODEL[$EMB_MODEL]:-gpu}"
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
echo

for ds in "${DATASETS[@]}"; do
    cfg="${BASE_CFG[$ds]}"
    h5ad="${H5AD[$ds]}"
    run_name="single_${ds}_${EMB}"
    out_dir="runs/world_model/${run_name}"

    if [[ "$KIND" == "precomputed" ]]; then
        echo "[$ds] submitting train (precomputed: $EMB, pixi env: $PIXI_ENV) ..."
        jid=$(sbatch --parsable \
            --job-name="wm-${ds}-${EMB}" \
            --partition="$PARTITION" --qos="$QOS" \
            --export=ALL,EMBPY_PIXI_ENV="${PIXI_ENV}" \
            "$LAUNCHER" "$cfg" \
            "action_embedding.source=precomputed" \
            "action_embedding.path=${EMB_PATH}" \
            "data.gene_embedding_path=${EMB_PATH}" \
            "run_name=${run_name}" \
            "output_dir=${out_dir}")
        echo "[$ds]   train job: $jid  -> $out_dir"
    else
        echo "[$ds] submitting pre-warm (bio_embedder: $EMB_MODEL, pixi env: $PIXI_ENV) ..."
        # Constrain to 80GB GPUs: protein/DNA foundation models (ESM-2 650M/3B,
        # Enformer, Evo2, Caduceus, etc.) blow up on 32GB V100s with OOM. The
        # 80GB A100/H100 nodes have enough headroom for the full token stream.
        # Without explicit -o/-e, sbatch --wrap defaults to slurm-<jid>.out
        # in the cwd (project root). Force into logs/ to match the rest of
        # the pipeline and keep the workspace tidy.
        prewarm_jid=$(sbatch --parsable \
            --job-name="wm-prewarm-${ds}-${EMB}" \
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
                    --region full --pooling-strategy mean \
                    --organism human --id-type symbol")
        echo "[$ds]   pre-warm job: $prewarm_jid"
        echo "[$ds] chaining train after pre-warm ..."
        jid=$(sbatch --parsable \
            --job-name="wm-${ds}-${EMB}" \
            --partition="$PARTITION" --qos="$QOS" \
            --dependency=afterok:"${prewarm_jid}" \
            --export=ALL,EMBPY_PIXI_ENV="${PIXI_ENV}" \
            "$LAUNCHER" "$cfg" \
            "action_embedding.source=bio_embedder" \
            "action_embedding.model_name=${EMB_MODEL}" \
            "action_embedding.organism=human" \
            "action_embedding.id_type=symbol" \
            "action_embedding.region=full" \
            "action_embedding.pooling_strategy=mean" \
            "run_name=${run_name}" \
            "output_dir=${out_dir}")
        echo "[$ds]   train job: $jid  (after ${prewarm_jid})  -> $out_dir"
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
