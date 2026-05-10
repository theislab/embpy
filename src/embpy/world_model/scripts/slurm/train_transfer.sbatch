#!/bin/bash
# -----------------------------------------------------------------------------
# Pretrain on Nadig + fine-tune on p% of Replogle.
# Usage:
#   sbatch --export=ALL,FRACTION=0.10 \
#          src/embpy/world_model/scripts/slurm/train_transfer.sbatch
# or:
#   FRACTION=0.05 sbatch src/embpy/world_model/scripts/slurm/train_transfer.sbatch
# Env overrides:
#   PARTITION=gpu_p QOS=gpu_normal sbatch ...
# Defaults to fraction 0.10 if FRACTION is not set.
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-transfer
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# #SBATCH --partition=${PARTITION:-gpu_p}
# #SBATCH --qos=${QOS:-gpu_normal}

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

CONFIG="${1:-${CONFIG:-src/embpy/world_model/configs/experiments/transfer.yaml}}"
FRACTION="${FRACTION:-0.10}"
RUN_NAME="transfer_nadig_to_replogle_p$(printf '%03d' $(python -c "print(int(float('${FRACTION}')*100))"))"
OUTPUT_DIR="outputs/world_model/${RUN_NAME}"

echo "Config: $CONFIG"
echo "Fraction: $FRACTION"
echo "Run name: $RUN_NAME"

mkdir -p logs

exec pixi run -e gpu -- python -m embpy.world_model.scripts.train \
    --config "$CONFIG" \
    "transfer.finetune_fraction=${FRACTION}" \
    "run_name=${RUN_NAME}" \
    "output_dir=${OUTPUT_DIR}"
