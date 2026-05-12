#!/bin/bash
# -----------------------------------------------------------------------------
# Train the world model on Replogle (single-dataset setup).
#
# Hardware target: one H100 80GB on supergpu14-32.
#   - partition  = gpu_p           (only partition that exposes the H100 nodes)
#   - qos        = gpu_normal      (user-level QOS; h100_80gb is a node FEATURE
#                                   on this cluster, not a QOS)
#   - constraint = h100_80gb       (forces the scheduler to a supergpu14-32 node)
#   - gres       = gpu:h100:1      (typed gres = "1 x H100", not any GPU)
#
# Usage:
#   sbatch src/world_model/world_model/scripts/slurm/train_single_replogle.sbatch \
#          [config_path]
#
# To swap hardware (e.g. drop to A100 80GB for queue speed), override on the
# sbatch command line (SBATCH directives below are parsed before the shell,
# so $VAR substitution does NOT work in them):
#   sbatch --constraint=a100_80gb --gres=gpu:1 \
#          src/world_model/world_model/scripts/slurm/train_single_replogle.sbatch ...
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-replogle
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# Compute-node /tmp is a small local disk (a few GB). If the state-backbone
# cache ever misses, StackWrapper writes a ~10 GB intermediate h5ad before
# handing it to arc-stack, which blows out /tmp. Redirect to Lustre so the
# fallback path stays survivable. Pre-warmed runs never touch this.
export TMPDIR="${TMPDIR:-$PROJECT_DIR/runs/_tmp}"
mkdir -p "$TMPDIR"

CONFIG="${1:-${CONFIG:-src/world_model/world_model/configs/experiments/single_replogle.yaml}}"
echo "Config: $CONFIG"

mkdir -p logs

exec pixi run -e gpu -- python -m world_model.scripts.train --config "$CONFIG"
