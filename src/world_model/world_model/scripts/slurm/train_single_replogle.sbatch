#!/bin/bash
# -----------------------------------------------------------------------------
# Train the world model on Replogle (single-dataset setup).
# Targets: gpu_p partition, h100_80gb QOS (one full H100 80GB).
#
# Usage:
#   sbatch src/world_model/world_model/scripts/slurm/train_single_replogle.sbatch \
#          [config_path]
#
# To override partition/QOS pass them on the sbatch command line (SBATCH
# directives below are parsed BEFORE the shell, so $VAR substitution does
# not work there -- that is why we keep literal defaults here):
#   sbatch --partition=gpu_p --qos=gpu_normal \
#          src/world_model/world_model/scripts/slurm/train_single_replogle.sbatch ...
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-replogle
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=h100_80gb

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# Compute-node /tmp is a small local disk (a few GB). If the state-backbone
# cache ever misses, StackWrapper writes a ~10 GB intermediate h5ad before
# handing it to arc-stack, which blows out /tmp. Redirect to Lustre so the
# fallback path stays survivable. Pre-warmed runs never touch this.
export TMPDIR="${TMPDIR:-$PROJECT_DIR/outputs/_tmp}"
mkdir -p "$TMPDIR"

CONFIG="${1:-${CONFIG:-src/world_model/world_model/configs/experiments/single_replogle.yaml}}"
echo "Config: $CONFIG"

mkdir -p logs

exec pixi run -e gpu -- python -m world_model.scripts.train --config "$CONFIG"
