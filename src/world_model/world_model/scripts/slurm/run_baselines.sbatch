#!/bin/bash
# -----------------------------------------------------------------------------
# Fit and evaluate every baseline against the same train/test split as a
# given world-model run. Optionally also evaluates a checkpoint.
#
# Usage:
#   CONFIG=... CKPT=... sbatch src/world_model/world_model/scripts/slurm/run_baselines.sbatch
# When invoked by submit_gene_embeddings.sh, RUN_DIR is set instead and we infer
# CONFIG / CKPT from it.
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-baselines
#SBATCH -o logs/slurm/run_baselines/%x_%j.out
#SBATCH -e logs/slurm/run_baselines/%x_%j.err
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

export TMPDIR="${TMPDIR:-$PROJECT_DIR/runs/_tmp/baselines-${SLURM_JOB_ID:-manual}}"
mkdir -p "$TMPDIR"

# Pixi env propagated from the submit driver. Caduceus / Evo2 sweep runs
# set EMBPY_PIXI_ENV=caduceus|evo2; everything else stays on `gpu`.
PIXI_ENV="${EMBPY_PIXI_ENV:-gpu}"

if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR=$(pixi run -e "$PIXI_ENV" -- python -c "from world_model.utils.run_identity import resolve_latest_suffixed_run_dir; print(resolve_latest_suffixed_run_dir('${RUN_DIR}'))")
    CONFIG="${CONFIG:-${RUN_DIR}/config.yaml}"
    if [[ -z "${CKPT:-}" ]]; then
        # Pick the latest single-dataset final checkpoint under RUN_DIR.
        CKPT=$(ls -1t "${RUN_DIR}"/*_final.pt 2>/dev/null | head -n 1 || true)
    fi
fi

CONFIG="${1:-${CONFIG:?CONFIG env var (or first positional arg) required}}"
CKPT="${2:-${CKPT:-}}"
echo "Config: $CONFIG"
echo "Checkpoint: ${CKPT:-<none>}"
echo "Pixi env: $PIXI_ENV"

mkdir -p logs/slurm/run_baselines

if [[ -n "${CKPT:-}" ]]; then
    exec pixi run -e "$PIXI_ENV" -- python -m world_model.scripts.run_baselines \
        --config "$CONFIG" --checkpoint "$CKPT"
else
    exec pixi run -e "$PIXI_ENV" -- python -m world_model.scripts.run_baselines \
        --config "$CONFIG"
fi
