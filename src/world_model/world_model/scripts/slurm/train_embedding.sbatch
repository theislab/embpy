#!/bin/bash
# -----------------------------------------------------------------------------
# Generic single-dataset training launcher with CLI-override passthrough.
#
# This forwards EVERY positional arg after the config straight
# into `python -m world_model.scripts.train` as a dotted override. That lets
# submit_gene_embeddings.sh swap the dataset/action embedding without a
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
#SBATCH -o logs/slurm/train_embedding/%x_%j.out
#SBATCH -e logs/slurm/train_embedding/%x_%j.err
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

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
cd "$PROJECT_DIR"
PROJECT_DIR="${PROJECT_DIR%/}"
export PROJECT_DIR

export TMPDIR="${TMPDIR:-$PROJECT_DIR/runs/_tmp}"
mkdir -p "$TMPDIR" logs/slurm/train_embedding

if [[ $# -lt 1 ]]; then
    echo "usage: train_embedding.sbatch <config_path> [key.sub=value ...]" >&2
    exit 2
fi

CONFIG="$1"; shift
# Pixi env to run in. Defaults to `gpu`; attachment jobs for models with exotic
# CUDA kernels (Caduceus, Evo2) set EMBPY_PIXI_ENV at submit time so the
# whole DAG (attach -> train -> baselines -> compare) shares one env.
PIXI_ENV="${EMBPY_PIXI_ENV:-gpu}"
echo "Config:    $CONFIG"
echo "Overrides: $*"
echo "Pixi env:  $PIXI_ENV"

exec pixi run -e "$PIXI_ENV" -- python -m world_model.scripts.train \
    --config "$CONFIG" "$@"
