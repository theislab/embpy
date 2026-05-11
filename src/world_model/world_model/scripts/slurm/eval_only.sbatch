#!/bin/bash
# -----------------------------------------------------------------------------
# Re-run evaluation (model + baselines + plots + report) against an
# existing world-model checkpoint. No training.
#
# Usage:
#   CONFIG=... CKPT=... sbatch src/world_model/scripts/slurm/eval_only.sbatch
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# #SBATCH --partition=${PARTITION:-gpu_p}
# #SBATCH --qos=${QOS:-gpu_normal}

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

CONFIG="${1:-${CONFIG:?CONFIG env var required}}"
CKPT="${2:-${CKPT:?CKPT env var required}}"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"

mkdir -p logs

exec pixi run -e gpu -- python -m world_model.scripts.eval \
    --config "$CONFIG" --checkpoint "$CKPT"
