#!/usr/bin/env bash
# =============================================================================
# submit_transfer_borzoi.sh
#
# Transfer-learning experiment: pretrain the world model on the FULL Nadig
# dataset using Borzoi DNA action embeddings, then fine-tune / transfer to
# p% of Replogle and evaluate. Same Borzoi cache feeds both phases.
#
# DAG:
#   prewarm-nadig  ─┐
#                   ├─(afterok)─> transfer-borzoi ─(afterok)─> baselines ─> compare
#   prewarm-repl   ─┘
#
# Both pre-warm jobs write into the SAME cache file
# (runs/_cache/action_embeddings/borzoi_v0/full_mean_human.npz); symbols are
# merged, so running both just unions the two gene panels. Already-cached
# symbols are skipped, so re-running is cheap.
#
# Usage:
#   bash src/world_model/world_model/scripts/submit_transfer_borzoi.sh
#   FRACTION=0.05 bash .../submit_transfer_borzoi.sh        # vary fine-tune %
#   SKIP_COMPARE=1 bash .../submit_transfer_borzoi.sh       # train only
# =============================================================================
set -euo pipefail

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# Pin output_dir to ``runs/world_model/${RUN_NAME}`` (no per-job
# timestamp suffix) so the chained baselines + compare jobs can find
# the train artifacts by exact path. See submit_gene_embeddings.sh
# for the same rationale.
export EMBPY_NO_AUTO_SUFFIX=1

SLURM_DIR="src/world_model/world_model/scripts/slurm"
CFG="src/world_model/world_model/configs/experiments/transfer_nadig_to_replogle_borzoi.yaml"
FRACTION="${FRACTION:-0.10}"
# This cluster has NO default partition: every sbatch must name one.
# GPU work -> gpu_p / gpu_normal ; CPU-only compare -> cpu_p / cpu_normal.
PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
NADIG_H5AD="data/datasets/nadig/NadigOConner2024_jurkat.h5ad"
REPL_H5AD="data/datasets/replogle/replogle_2022_k562_essential.h5ad"
P_INT=$(python -c "print(int(float('${FRACTION}')*100))")
RUN_NAME="transfer_nadig_to_replogle_borzoi_p$(printf '%03d' "$P_INT")"
OUT_DIR="runs/world_model/${RUN_NAME}"
mkdir -p logs

prewarm() {  # $1=dataset label  $2=h5ad
    sbatch --parsable \
        --job-name="wm-prewarm-$1-borzoi" \
        --partition="$PARTITION" --qos="$QOS" \
        --gres=gpu:1 --constraint="a100_80gb|h100_80gb" \
        --time=08:00:00 --mem=64G --cpus-per-task=8 \
        -o logs/%x_%j.out -e logs/%x_%j.err \
        --wrap="set -euo pipefail; cd ${PROJECT_DIR}; \
            export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
            export TMPDIR=\"${PROJECT_DIR}/.tmp/job-\$SLURM_JOB_ID\"; \
            mkdir -p \"\$TMPDIR\"; \
            trap 'rm -rf \"\$TMPDIR\"' EXIT; \
            pixi run -e gpu -- python -m world_model.scripts.embed_perturbations \
                --dataset $1 --h5ad $2 --model borzoi_v0 \
                --region full --pooling-strategy mean \
                --organism human --id-type symbol"
}

echo "Pre-warming Borzoi cache for Nadig (jurkat) ..."
PW_NADIG=$(prewarm nadig "$NADIG_H5AD")
echo "  -> $PW_NADIG"

echo "Pre-warming Borzoi cache for Replogle (k562 essential) ..."
PW_REPL=$(prewarm replogle "$REPL_H5AD")
echo "  -> $PW_REPL"

echo "Submitting transfer training (fraction=${FRACTION}) after pre-warm ..."
TRAIN_JID=$(FRACTION="${FRACTION}" sbatch --parsable \
    --partition="$PARTITION" --qos="$QOS" \
    --dependency=afterok:"${PW_NADIG}":"${PW_REPL}" \
    "${SLURM_DIR}/transfer_nadig_to_replogle_borzoi.sbatch" "${CFG}")
echo "  -> $TRAIN_JID  (run: ${RUN_NAME})"

if [[ "${SKIP_COMPARE:-0}" != "1" ]]; then
    echo "Chaining baselines after training ..."
    BASE_JID=$(sbatch --parsable \
        --partition="$PARTITION" --qos="$QOS" \
        --dependency=afterok:"${TRAIN_JID}" \
        --export=ALL,RUN_DIR="${OUT_DIR}" \
        "${SLURM_DIR}/run_baselines.sbatch")
    echo "  -> $BASE_JID"
    echo "Chaining compare/report after baselines ..."
    CMP_JID=$(sbatch --parsable \
        --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
        --dependency=afterok:"${BASE_JID}" \
        --export=ALL,RUN_DIR="${OUT_DIR}" \
        "${SLURM_DIR}/compare.sbatch")
    echo "  -> $CMP_JID"
fi

echo
echo "Submitted transfer-Borzoi DAG:"
printf "  prewarm-nadig    %s\n" "${PW_NADIG}"
printf "  prewarm-replogle %s\n" "${PW_REPL}"
printf "  transfer-train   %s\n" "${TRAIN_JID}"
[[ "${SKIP_COMPARE:-0}" != "1" ]] && printf "  baselines        %s\n" "${BASE_JID:-}"
[[ "${SKIP_COMPARE:-0}" != "1" ]] && printf "  compare/report   %s\n" "${CMP_JID:-}"
echo "  output_dir       ${OUT_DIR}"
echo
if command -v squeue >/dev/null 2>&1; then
    squeue --me --format="%.18i %.9P %.28j %.8T %.10M %.6D %R" || true
fi
