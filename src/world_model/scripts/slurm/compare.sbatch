#!/bin/bash
# -----------------------------------------------------------------------------
# Render comparison.csv + comparison.png + report.md from a finished run.
# Runs after run_baselines.sbatch has produced baselines.csv.
#
# Usage:
#   RUN_DIR=outputs/world_model/single_replogle \
#     sbatch src/world_model/scripts/slurm/compare.sbatch
# Env overrides:
#   PARTITION=cpu_p sbatch ... (no GPU needed; comparison is pure pandas)
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-compare
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# #SBATCH --partition=${PARTITION:-cpu_p}
# #SBATCH --qos=${QOS:-cpu_normal}

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

RUN_DIR="${1:-${RUN_DIR:?RUN_DIR env var or first positional arg required}}"
echo "Run dir: $RUN_DIR"

mkdir -p logs

pixi run -e gpu -- python -m world_model.scripts.compare --run-dir "$RUN_DIR"
exec pixi run -e gpu -- python -m world_model.scripts.make_report --run-dir "$RUN_DIR"
