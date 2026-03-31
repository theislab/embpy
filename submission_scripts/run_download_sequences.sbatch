#!/bin/bash
#SBATCH --job-name=dl-seqs
#SBATCH --output=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/download_sequences_%A_%a.out
#SBATCH --error=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/download_sequences_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --partition=cpu_p
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --array=0-4

# ===========================================================================
# Pre-download gene/protein sequences -- one step per array task
# ===========================================================================
# Task 0: Full genomic DNA
# Task 1: Exon DNA
# Task 2: Intron DNA
# Task 3: Canonical protein sequences
# Task 4: All protein isoforms
#
# Output:
#   data/datasets/genome/          -- DNA .npz files
#   data/datasets/proteome/canonical/  -- canonical protein .npz
#   data/datasets/proteome/isoforms/   -- isoform .npz
#
# Usage:
#   sbatch run_download_sequences.sbatch
# ===========================================================================

set -eo pipefail

DATA_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/datasets"
SCRIPT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts"

mkdir -p "${SCRIPT_DIR}/logs"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started at $(date) ==="
echo "Node: $(hostname)"

set +u
eval "$(conda shell.bash hook)"
conda activate /ictstr01/home/icb/goncalo.pinto/tools/apps/mamba/envs/embpy
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

CONFIGS=(
    "dna --region full"
    "dna --region exons"
    "dna --region introns"
    "protein"
    "protein_isoforms"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Running: --type ${CONFIG} ==="

python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type ${CONFIG}

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete at $(date) ==="
