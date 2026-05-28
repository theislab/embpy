#!/bin/bash
# -----------------------------------------------------------------------------
# Render comparison.csv + comparison.png + report.md from a finished run.
# Runs after run_baselines.sbatch has produced baselines.csv.
#
# Usage:
#   RUN_DIR=runs/world_model/single_replogle \
#     sbatch src/world_model/world_model/scripts/slurm/compare.sbatch
# Env overrides:
#   PARTITION=cpu_p sbatch ... (no GPU needed; comparison is pure pandas)
# -----------------------------------------------------------------------------
#SBATCH --job-name=wm-compare
#SBATCH -o logs/slurm/compare/%x_%j.out
#SBATCH -e logs/slurm/compare/%x_%j.err
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
RUN_DIR=$(pixi run -e gpu -- python -c "from world_model.utils.run_identity import resolve_latest_suffixed_run_dir; print(resolve_latest_suffixed_run_dir('${RUN_DIR}'))")
echo "Run dir: $RUN_DIR"

mkdir -p logs/slurm/compare

pixi run -e gpu -- python -m world_model.scripts.compare --run-dir "$RUN_DIR"
exec pixi run -e gpu -- python -m world_model.scripts.make_report --run-dir "$RUN_DIR"
