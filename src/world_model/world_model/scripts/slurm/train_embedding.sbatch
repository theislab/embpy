#!/bin/bash
# -----------------------------------------------------------------------------
# Generic single-dataset training launcher with CLI-override passthrough.
#
# Unlike train_single_{nadig,replogle}.sbatch (which only forward a config
# path), this one forwards EVERY positional arg after the config straight
# into `python -m world_model.scripts.train` as a dotted override -- exactly
# the mechanism train_transfer.sbatch already relies on. That lets the
# submit_gene_embeddings.sh driver swap the action/gene embedding without a
# bespoke YAML per embedding.
#
# Usage:
#   sbatch src/world_model/world_model/scripts/slurm/train_embedding.sbatch \
#          <config_path> [key.sub=value ...]
#
# Hardware is generic (1 GPU) so the job is not stuck behind the H100 queue;
# override on the sbatch line if you need a specific node, e.g.:
#   sbatch --constraint=h100_80gb --gres=gpu:h100:1 --partition=gpu_p \
#          --qos=gpu_normal .../train_embedding.sbatch <config> ...
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-emb
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

if [[ $# -lt 1 ]]; then
    echo "usage: train_embedding.sbatch <config_path> [key.sub=value ...]" >&2
    exit 2
fi

CONFIG="$1"; shift
echo "Config:    $CONFIG"
echo "Overrides: $*"

exec pixi run -e gpu -- python -m world_model.scripts.train \
    --config "$CONFIG" "$@"
