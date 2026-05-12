#!/bin/bash
# -----------------------------------------------------------------------------
# Leave-one-encoder-out (LOEO) transfer sweep.
#
# Two array stages chained externally with --dependency=afterok::
#   STAGE=A array=0..|grid|-1   pretrain reuse, one job per encoder X.
#   STAGE=B array=0..|cells|-1  off-diagonal cells, one job per (X, Y).
#
# Both stages share BASE_CONFIG, GRID, STRATEGY and OUTPUT_ROOT. Stage B
# requires Stage A's checkpoints to already exist.
#
# Usage (Stage A, array of pretrains):
#   STAGE=A BASE_CONFIG=.../transfer.yaml GRID=.../ablation_action_encoder.yaml \
#       STRATEGY=reset_adapter OUTPUT_ROOT=outputs/lone_replogle \
#       sbatch --array=0-4 src/.../slurm/leave_one_encoder_out.sbatch
#
# Usage (Stage B, array of off-diagonal cells; depends on Stage A):
#   STAGE=B BASE_CONFIG=... GRID=... STRATEGY=reset_adapter \
#       OUTPUT_ROOT=outputs/lone_replogle \
#       sbatch --array=0-19 --dependency=afterok:<jobid_A> \
#           src/.../slurm/leave_one_encoder_out.sbatch
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-lone
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

BASE_CONFIG="${BASE_CONFIG:-src/world_model/world_model/configs/experiments/transfer.yaml}"
GRID="${GRID:-src/world_model/world_model/configs/experiments/ablation_action_encoder.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/lone_replogle}"
STRATEGY="${STRATEGY:-reset_adapter}"
STAGE="${STAGE:-A}"
SEED_ARG=""
if [[ -n "${SEED:-}" ]]; then
    SEED_ARG="--seed ${SEED}"
fi

if [[ "${STAGE}" == "A" ]]; then
    # ----- Pretrain reuse: one job per encoder X.
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "STAGE=A requires SLURM_ARRAY_TASK_ID (use sbatch --array=0-N)." >&2
        exit 2
    fi
    X_KEY=$(pixi run -e gpu -- python -c "
import sys
from world_model.evaluation.ablation import load_grid
specs = load_grid(sys.argv[1])
i = int(sys.argv[2])
if i < 0 or i >= len(specs):
    raise SystemExit(f'SLURM_ARRAY_TASK_ID={i} out of range for grid of size {len(specs)}')
print(specs[i].key)
" "$GRID" "$SLURM_ARRAY_TASK_ID")
    echo "Stage A: pretrain X=${X_KEY}"
    exec pixi run -e gpu -- python -m world_model.scripts.leave_one_encoder_out \
        --base-config "$BASE_CONFIG" \
        --grid "$GRID" \
        --strategy "$STRATEGY" \
        --output-root "$OUTPUT_ROOT" \
        --only "${X_KEY}:${X_KEY}" \
        ${SEED_ARG}
fi

if [[ "${STAGE}" == "B" ]]; then
    # ----- Fine-tune sweep: one job per off-diagonal (X, Y) cell.
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "STAGE=B requires SLURM_ARRAY_TASK_ID." >&2
        exit 2
    fi
    PAIR=$(pixi run -e gpu -- python -c "
import sys
from world_model.evaluation.ablation import load_grid
keys = [s.key for s in load_grid(sys.argv[1])]
pairs = [(x, y) for x in keys for y in keys if x != y]
i = int(sys.argv[2])
if i < 0 or i >= len(pairs):
    raise SystemExit(f'SLURM_ARRAY_TASK_ID={i} out of range for {len(pairs)} off-diagonal cells')
x, y = pairs[i]
print(f'{x}:{y}')
" "$GRID" "$SLURM_ARRAY_TASK_ID")
    echo "Stage B: cell ${PAIR}"
    exec pixi run -e gpu -- python -m world_model.scripts.leave_one_encoder_out \
        --base-config "$BASE_CONFIG" \
        --grid "$GRID" \
        --strategy "$STRATEGY" \
        --output-root "$OUTPUT_ROOT" \
        --only "${PAIR}" \
        ${SEED_ARG}
fi

echo "Unknown STAGE=${STAGE}; expected A or B." >&2
exit 2
