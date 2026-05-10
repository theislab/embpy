#!/bin/bash
# -----------------------------------------------------------------------------
# Fit and evaluate every baseline against the same train/test split as a
# given world-model run. Optionally also evaluates a checkpoint.
#
# Usage:
#   CONFIG=... CKPT=... sbatch src/embpy/world_model/scripts/slurm/run_baselines.sbatch
# When invoked by submit_all.sh, RUN_DIR is set instead and we infer
# CONFIG / CKPT from it.
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-baselines
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

if [[ -n "${RUN_DIR:-}" ]]; then
    CONFIG="${CONFIG:-${RUN_DIR}/config.yaml}"
    if [[ -z "${CKPT:-}" ]]; then
        # Pick the latest *_final.pt or *_finetune_final.pt under RUN_DIR.
        CKPT=$(ls -1t "${RUN_DIR}"/*_final.pt "${RUN_DIR}"/finetune/*_final.pt 2>/dev/null | head -n 1 || true)
    fi
fi

CONFIG="${1:-${CONFIG:?CONFIG env var (or first positional arg) required}}"
CKPT="${2:-${CKPT:-}}"
echo "Config: $CONFIG"
echo "Checkpoint: ${CKPT:-<none>}"

mkdir -p logs

if [[ -n "${CKPT:-}" ]]; then
    exec pixi run -e gpu -- python -m embpy.world_model.scripts.run_baselines \
        --config "$CONFIG" --checkpoint "$CKPT"
else
    exec pixi run -e gpu -- python -m embpy.world_model.scripts.run_baselines \
        --config "$CONFIG"
fi
