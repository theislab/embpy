#!/bin/bash
# -----------------------------------------------------------------------------
# Transfer: pretrain on Nadig + fine-tune on p% of Replogle, Borzoi action
# embeddings shared across both phases.
#
# Submit via submit_transfer_borzoi.sh (it pre-warms the shared Borzoi cache
# for both gene panels and chains this job afterok). Direct use:
#   FRACTION=0.10 sbatch \
#     src/world_model/world_model/scripts/slurm/transfer_nadig_to_replogle_borzoi.sbatch
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-transfer-borzoi
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"
export TMPDIR="${TMPDIR:-$PROJECT_DIR/runs/_tmp}"
mkdir -p "$TMPDIR" logs

CONFIG="${1:-${CONFIG:-src/world_model/world_model/configs/experiments/transfer_nadig_to_replogle_borzoi.yaml}}"
FRACTION="${FRACTION:-0.10}"
P_INT=$(python -c "print(int(float('${FRACTION}')*100))")
RUN_NAME="transfer_nadig_to_replogle_borzoi_p$(printf '%03d' "$P_INT")"
OUTPUT_DIR="runs/world_model/${RUN_NAME}"

echo "Config:    $CONFIG"
echo "Fraction:  $FRACTION"
echo "Run name:  $RUN_NAME"

exec pixi run -e gpu -- python -m world_model.scripts.train \
    --config "$CONFIG" \
    "transfer.finetune_fraction=${FRACTION}" \
    "run_name=${RUN_NAME}" \
    "output_dir=${OUTPUT_DIR}"
