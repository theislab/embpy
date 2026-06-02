#!/bin/bash
# -----------------------------------------------------------------------------
# Re-run evaluation (model + baselines + plots + report) on CPU against an
# existing world-model checkpoint. No training.
#
# This launcher is intended for dependency chains after GPU training:
# training releases the GPU, then this CPU job computes the full cell-eval
# report from the final checkpoint.
#
# Usage:
#   RUN_DIR=... sbatch src/world_model/world_model/scripts/slurm/eval_only_cpu.sbatch
#   sbatch src/world_model/world_model/scripts/slurm/eval_only_cpu.sbatch <run_dir>
#   CONFIG=... CKPT=... sbatch src/world_model/world_model/scripts/slurm/eval_only_cpu.sbatch
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-eval-cpu
#SBATCH -o logs/slurm/eval_only_cpu/%x_%j.out
#SBATCH -e logs/slurm/eval_only_cpu/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
cd "$PROJECT_DIR"
PROJECT_DIR="${PROJECT_DIR%/}"
export PROJECT_DIR

PIXI_ENV="${EMBPY_PIXI_ENV:-gpu}"
mkdir -p logs/slurm/eval_only_cpu

if [[ -n "${1:-}" && -z "${RUN_DIR:-}" && -z "${CONFIG:-}" ]]; then
    RUN_DIR="$1"
    shift
fi

if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR=$(pixi run -e "$PIXI_ENV" -- python -c "from world_model.utils.run_identity import resolve_latest_suffixed_run_dir; print(resolve_latest_suffixed_run_dir('${RUN_DIR}'))")
    CONFIG="${CONFIG:-${RUN_DIR}/config.yaml}"
    if [[ -z "${CKPT:-}" ]]; then
        CKPT=$(ls -1t "${RUN_DIR}"/*_final.pt 2>/dev/null | head -n 1 || true)
    fi
fi

CONFIG="${CONFIG:?CONFIG env var, first positional run dir, or RUN_DIR env var required}"
CKPT="${CKPT:?CKPT env var required when RUN_DIR is not provided}"

CELL_EVAL_OVERRIDES="${CELL_EVAL_OVERRIDES:-train.device=cpu eval.require_cell_eval=true eval.cell_eval_profile=full eval.cell_eval_num_threads=${SLURM_CPUS_PER_TASK:-4} eval.save_predictions=false}"
read -r -a CELL_EVAL_OVERRIDE_ARGS <<<"$CELL_EVAL_OVERRIDES"

echo "Run dir:    ${RUN_DIR:-<not provided>}"
echo "Config:     $CONFIG"
echo "Checkpoint: $CKPT"
echo "Pixi env:   $PIXI_ENV"
echo "Output dir: $(dirname "$CONFIG")"
echo "Overrides:  ${CELL_EVAL_OVERRIDE_ARGS[*]} $*"

exec pixi run -e "$PIXI_ENV" -- python -m world_model.scripts.eval \
    --config "$CONFIG" --checkpoint "$CKPT" \
    "${CELL_EVAL_OVERRIDE_ARGS[@]}" "$@"
