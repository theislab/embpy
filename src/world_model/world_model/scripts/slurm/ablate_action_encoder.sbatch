#!/bin/bash
# -----------------------------------------------------------------------------
# Run the action-encoder ablation sweep.
# -----------------------------------------------------------------------------
# Usage (single job, walks the grid sequentially):
#   BASE_CONFIG=src/world_model/world_model/configs/experiments/single_replogle.yaml \
#   GRID=src/world_model/world_model/configs/experiments/ablation_action_encoder.yaml \
#   OUTPUT_ROOT=runs/ablation_action_replogle \
#   sbatch src/world_model/world_model/scripts/slurm/ablate_action_encoder.sbatch
#
# Usage (array job, one spec per array task):
#   sbatch --array=0-4 src/.../slurm/ablate_action_encoder.sbatch
#   When SLURM_ARRAY_TASK_ID is set, the script picks the corresponding
#   grid key and runs --only that single spec.
#
# Optional env: ONLY=borzoi,esm2_650m  SKIP=minilm  SEED=0
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-ablate-action
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
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

mkdir -p logs

BASE_CONFIG="${BASE_CONFIG:-src/world_model/world_model/configs/experiments/single_replogle.yaml}"
GRID="${GRID:-src/world_model/world_model/configs/experiments/ablation_action_encoder.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/ablation_action_replogle}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
SEED_ARG=""
if [[ -n "${SEED:-}" ]]; then
    SEED_ARG="--seed ${SEED}"
fi

# When this is an array job, resolve --only to a single spec.key.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SPEC_KEY=$(pixi run -e gpu -- python -c "
import sys
from world_model.evaluation.ablation import load_grid
specs = load_grid(sys.argv[1])
i = int(sys.argv[2])
if i < 0 or i >= len(specs):
    raise SystemExit(f'SLURM_ARRAY_TASK_ID={i} out of range for grid of size {len(specs)}')
print(specs[i].key)
" "$GRID" "$SLURM_ARRAY_TASK_ID")
    echo "Array task ${SLURM_ARRAY_TASK_ID} -> spec ${SPEC_KEY}"
    ONLY="${SPEC_KEY}"
fi

EXTRA_ARGS=()
if [[ -n "$ONLY" ]]; then
    EXTRA_ARGS+=(--only "$ONLY")
fi
if [[ -n "$SKIP" ]]; then
    EXTRA_ARGS+=(--skip "$SKIP")
fi

echo "BASE_CONFIG=${BASE_CONFIG}"
echo "GRID=${GRID}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "ONLY=${ONLY:-(none)}"
echo "SKIP=${SKIP:-(none)}"

exec pixi run -e gpu -- python -m world_model.scripts.ablate_action_encoder \
    --base-config "$BASE_CONFIG" \
    --grid "$GRID" \
    --output-root "$OUTPUT_ROOT" \
    ${SEED_ARG} \
    "${EXTRA_ARGS[@]}"
