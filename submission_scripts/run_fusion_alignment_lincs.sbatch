#!/usr/bin/env bash
#SBATCH --job-name=embpy-fusion-stage2
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00

set -euo pipefail
: "${EMBPY_PYTHON:?Set EMBPY_PYTHON to the cluster Python}"
: "${EMBPY_SOURCE_ROOT:?Set EMBPY_SOURCE_ROOT to the checkout or source snapshot}"
export PYTHONPATH="${EMBPY_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
"$EMBPY_PYTHON" -u -m embpy.scripts.benchmark_fusion_alignment_lincs "$@"
