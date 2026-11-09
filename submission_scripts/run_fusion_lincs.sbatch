#!/usr/bin/env bash
#SBATCH --job-name=embpy-fusion-stage1
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
: "${EMBPY_PYTHON:?Set EMBPY_PYTHON to the cluster environment Python}"
: "${EMBPY_SOURCE_ROOT:?Set EMBPY_SOURCE_ROOT to the checkout or source snapshot}"
export PYTHONPATH="${EMBPY_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
"$EMBPY_PYTHON" -u -m embpy.scripts.benchmark_fusion_lincs "$@"
