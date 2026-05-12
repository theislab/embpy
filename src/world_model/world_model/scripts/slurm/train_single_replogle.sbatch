#!/bin/bash
# -----------------------------------------------------------------------------
# Train the world model on Replogle (single-dataset setup).
# Usage:
#   sbatch src/world_model/world_model/scripts/slurm/train_single_replogle.sbatch \
#          [config_path]
# Env overrides:
#   PARTITION=gpu_p QOS=gpu_normal sbatch ...
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-replogle
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=${PARTITION:-gpu_p}
#SBATCH --qos=h100_80gb

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

CONFIG="${1:-${CONFIG:-src/world_model/world_model/configs/experiments/single_replogle.yaml}}"
echo "Config: $CONFIG"

mkdir -p logs

exec pixi run -e gpu -- python -m world_model.scripts.train --config "$CONFIG"
