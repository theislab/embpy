#!/bin/bash
# -----------------------------------------------------------------------------
# Re-run evaluation (model + baselines + plots + report) against an
# existing world-model checkpoint. No training.
#
# Usage:
#   CONFIG=... CKPT=... sbatch src/world_model/world_model/scripts/slurm/eval_only.sbatch
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-eval
#SBATCH -o logs/slurm/eval_only/%x_%j.out
#SBATCH -e logs/slurm/eval_only/%x_%j.err
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

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
cd "$PROJECT_DIR"
PROJECT_DIR="${PROJECT_DIR%/}"
export PROJECT_DIR

CONFIG="${1:-${CONFIG:?CONFIG env var required}}"
CKPT="${2:-${CKPT:?CKPT env var required}}"
CELL_EVAL_OVERRIDES="${CELL_EVAL_OVERRIDES:-eval.require_cell_eval=true eval.cell_eval_profile=full eval.cell_eval_num_threads=${SLURM_CPUS_PER_TASK:-8}}"
read -r -a CELL_EVAL_OVERRIDE_ARGS <<<"$CELL_EVAL_OVERRIDES"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Cell-eval overrides: ${CELL_EVAL_OVERRIDE_ARGS[*]}"

mkdir -p logs

exec pixi run -e gpu -- python -m world_model.scripts.eval \
    --config "$CONFIG" --checkpoint "$CKPT" \
    "${CELL_EVAL_OVERRIDE_ARGS[@]}"
