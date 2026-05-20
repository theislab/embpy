#!/usr/bin/env bash
# =============================================================================
# submit_transfer_sweep.sh
#
# Transfer-learning sweep: pretrain on Nadig, fine-tune on a fraction of
# Replogle, EVALUATE on the rest of Replogle -- across multiple action
# embeddings and multiple fine-tune fractions.
#
# DAG per (embedding, fraction):
#     prewarm-nadig-EMB   (shared across fractions per embedding)
#     prewarm-replogle-EMB (shared)
#                |
#                v
#         train (--config transfer_nadig_to_replogle_borzoi.yaml + overrides)
#                |
#                v
#         baselines  (afterok train)
#                |
#                v
#         compare    (afterok baselines)
#
# Per (embedding, fraction) -> 1 train + 1 baselines + 1 compare. Pre-warm
# is shared per (embedding, dataset) so we only run 2*len(EMBEDDINGS)
# pre-warm jobs total.
#
# All paths are predictable because EMBPY_NO_AUTO_SUFFIX=1 is exported
# before submitting -- baselines / compare can find the train artifacts.
#
# Defaults:
#   EMBEDDINGS = "borzoi_v0 enformer_human_rough nt_v2_500m esm2_650M minilm_l6_v2"
#   FRACTIONS  = "0.01 0.05 0.10 0.25 0.50"
#
# Usage:
#   bash src/world_model/world_model/scripts/submit_transfer_sweep.sh
#   FRACTIONS="0.10" bash .../submit_transfer_sweep.sh         # single point
#   EMBEDDINGS="borzoi_v0 esm2_650M" bash .../submit_transfer_sweep.sh
#   SKIP_COMPARE=1 bash .../submit_transfer_sweep.sh           # train only
#   DRYRUN=1 bash .../submit_transfer_sweep.sh                 # plan only
# =============================================================================
set -euo pipefail

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# Predictable paths so baselines / compare jobs can find RUN_DIR.
export EMBPY_NO_AUTO_SUFFIX=1

EMBEDDINGS_DEFAULT="borzoi_v0 enformer_human_rough nt_v2_500m esm2_650M minilm_l6_v2"
FRACTIONS_DEFAULT="0.01 0.05 0.10 0.25 0.50"

EMBEDDINGS="${EMBEDDINGS:-$EMBEDDINGS_DEFAULT}"
FRACTIONS="${FRACTIONS:-$FRACTIONS_DEFAULT}"
CFG_BASE="src/world_model/world_model/configs/experiments/transfer_nadig_to_replogle_borzoi.yaml"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
NADIG_H5AD="data/datasets/nadig/NadigOConner2024_jurkat.h5ad"
REPL_H5AD="data/datasets/replogle/replogle_2022_k562_essential.h5ad"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

mkdir -p logs runs/world_model/_transfer_sweep

# Read embeddings + fractions into arrays.
read -r -a EMB_LIST <<<"$EMBEDDINGS"
read -r -a FRAC_LIST <<<"$FRACTIONS"

if [[ ${#EMB_LIST[@]} -eq 0 ]] || [[ ${#FRAC_LIST[@]} -eq 0 ]]; then
    echo "ERROR: empty EMBEDDINGS or FRACTIONS" >&2
    exit 1
fi

echo "Transfer sweep plan"
echo "  embeddings:  ${#EMB_LIST[@]}  ->  ${EMB_LIST[*]}"
echo "  fractions:   ${#FRAC_LIST[@]}  ->  ${FRAC_LIST[*]}"
echo "  total DAGs:  $(( ${#EMB_LIST[@]} * ${#FRAC_LIST[@]} ))"
echo "  pre-warms:   $(( 2 * ${#EMB_LIST[@]} ))  (Nadig + Replogle per embedding)"
echo

if [[ "${DRYRUN:-0}" == "1" ]]; then
    echo "DRYRUN=1 -- nothing submitted."
    exit 0
fi

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
prewarm_for() {  # $1=dataset_label, $2=h5ad, $3=model -> echoes jid
    # Constrain to 80GB GPUs: protein/DNA foundation models (ESM-2 650M/3B,
    # Enformer, Evo2, Caduceus, ...) OOM on 32GB V100s. 80GB A100/H100 nodes
    # have the headroom needed.
    sbatch --parsable \
        --job-name="wm-prewarm-$1-$3" \
        --partition="$PARTITION" --qos="$QOS" \
        --gres=gpu:1 --constraint="a100_80gb|h100_80gb" \
        --time=08:00:00 --mem=64G --cpus-per-task=8 \
        -o logs/%x_%j.out -e logs/%x_%j.err \
        --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
            export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
            pixi run -e gpu -- python -m world_model.scripts.embed_perturbations \
                --dataset $1 --h5ad $2 --model $3 \
                --region full --pooling-strategy mean \
                --organism human --id-type symbol"
}

train_for() {  # $1=emb, $2=fraction, $3=pw_nadig_jid, $4=pw_repl_jid -> echoes jid
    local p_int run_name out_dir
    p_int=$(python -c "print(int(float('$2')*100))")
    run_name="transfer_${1}_p$(printf '%03d' "$p_int")"
    out_dir="runs/world_model/${run_name}"
    sbatch --parsable \
        --job-name="wm-transfer-${1}-p$(printf '%03d' "$p_int")" \
        --partition="$PARTITION" --qos="$QOS" \
        --gres=gpu:1 --time=24:00:00 --mem=64G --cpus-per-task=8 \
        --dependency=afterok:"$3":"$4" \
        -o logs/%x_%j.out -e logs/%x_%j.err \
        --export=ALL,EMBPY_NO_AUTO_SUFFIX=1 \
        --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
            export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
            export TMPDIR=${PROJECT_DIR}/runs/_tmp; mkdir -p \$TMPDIR; \
            pixi run -e gpu -- python -m world_model.scripts.train \
                --config '${CFG_BASE}' \
                'action_embedding.model_name=${1}' \
                'transfer.finetune_fraction=$2' \
                'run_name=${run_name}' \
                'output_dir=${out_dir}'"
}

baselines_after() {  # $1=run_dir, $2=train_jid -> echoes jid
    sbatch --parsable \
        --partition="$PARTITION" --qos="$QOS" \
        --dependency=afterok:"$2" \
        --export=ALL,RUN_DIR="$1",EMBPY_NO_AUTO_SUFFIX=1 \
        "${SLURM_DIR}/run_baselines.sbatch"
}

compare_after() {  # $1=run_dir, $2=base_jid -> echoes jid
    sbatch --parsable \
        --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
        --dependency=afterok:"$2" \
        --export=ALL,RUN_DIR="$1",EMBPY_NO_AUTO_SUFFIX=1 \
        "${SLURM_DIR}/compare.sbatch"
}

# ----------------------------------------------------------------------
# 1. Pre-warm caches per embedding (shared across all fractions)
# ----------------------------------------------------------------------
declare -A PW_NADIG PW_REPL
for emb in "${EMB_LIST[@]}"; do
    echo "Pre-warming caches for ${emb} ..."
    PW_NADIG[$emb]=$(prewarm_for nadig    "$NADIG_H5AD" "$emb")
    PW_REPL[$emb]=$(prewarm_for replogle "$REPL_H5AD"  "$emb")
    echo "  prewarm-nadig    ${PW_NADIG[$emb]}"
    echo "  prewarm-replogle ${PW_REPL[$emb]}"
done
echo

# ----------------------------------------------------------------------
# 2. (embedding x fraction) DAGs: train -> baselines -> compare
# ----------------------------------------------------------------------
CMP_JIDS=()
ROWS=()
for emb in "${EMB_LIST[@]}"; do
    for frac in "${FRAC_LIST[@]}"; do
        p_int=$(python -c "print(int(float('$frac')*100))")
        run_name="transfer_${emb}_p$(printf '%03d' "$p_int")"
        out_dir="runs/world_model/${run_name}"

        TRAIN_JID=$(train_for "$emb" "$frac" "${PW_NADIG[$emb]}" "${PW_REPL[$emb]}")

        if [[ "$SKIP_COMPARE" != "1" ]]; then
            BASE_JID=$(baselines_after "$out_dir" "$TRAIN_JID")
            CMP_JID=$(compare_after "$out_dir" "$BASE_JID")
            CMP_JIDS+=("$CMP_JID")
            ROWS+=("$emb p=${frac} train=$TRAIN_JID base=$BASE_JID cmp=$CMP_JID")
        else
            ROWS+=("$emb p=${frac} train=$TRAIN_JID (no compare)")
        fi
    done
done

# ----------------------------------------------------------------------
# 3. Cross-sweep aggregator (depends on every compare)
# ----------------------------------------------------------------------
if [[ "$SKIP_COMPARE" != "1" ]] && [[ ${#CMP_JIDS[@]} -gt 0 ]]; then
    DEP=$(IFS=:; echo "${CMP_JIDS[*]}")
    AGG_JID=$(sbatch --parsable \
        --job-name=wm-transfer-aggregate \
        --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
        --cpus-per-task=2 --mem=8G --time=00:30:00 \
        -o logs/%x_%j.out -e logs/%x_%j.err \
        --dependency=afterany:"${DEP}" \
        --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
            export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
            pixi run -e gpu -- python -m world_model.scripts.aggregate_sweep \
                --datasets replogle --out runs/world_model/_transfer_sweep")
    echo "Aggregator: $AGG_JID  (afterany ${#CMP_JIDS[@]} compares)"
    echo
fi

# ----------------------------------------------------------------------
# 4. Summary
# ----------------------------------------------------------------------
echo "Submitted transfer sweep:"
for row in "${ROWS[@]}"; do
    echo "  $row"
done
echo
echo "  output_dirs: runs/world_model/transfer_<emb>_p<NNN>/"
echo "  aggregator output: runs/world_model/_transfer_sweep/"
echo

if command -v squeue >/dev/null 2>&1; then
    squeue --me --format="%.18i %.9P %.36j %.8T %.10M %.6D %R" || true
fi
