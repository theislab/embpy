#!/bin/bash
# Compatibility wrapper for Nadig single-dataset training.
# Canonical launcher: train_embedding.sbatch <config> [overrides...]
#SBATCH --job-name=wm-nadig
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${CONFIG:-src/world_model/world_model/configs/experiments/single_nadig.yaml}}"
if [[ $# -gt 0 ]]; then
    shift
fi

exec "${SCRIPT_DIR}/train_embedding.sbatch" "$CONFIG" "$@"
