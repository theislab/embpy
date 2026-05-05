#!/bin/bash
# -----------------------------------------------------------------------------
# Timing test for the BioEmbedder single-cell wrapper cache.
#
# Launches one GPU job that calls `embed_cells(scgpt)` three times in a row
# on the same BioEmbedder instance and prints cold vs. warm timings.
# If the cache is working, warm calls are ~10-100x faster than the cold one.
# -----------------------------------------------------------------------------

#SBATCH --job-name=embpy-cache-test
#SBATCH -o logs/test_cache_scgpt_%j.out
#SBATCH -e logs/test_cache_scgpt_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# The test uses the helical-gpu env because it has scGPT installed.
echo "----- nvidia-smi -----"
nvidia-smi || true

echo "----- running cache test -----"
exec pixi run -e helical-gpu -- python submission_scripts/test_cache_scgpt.py
