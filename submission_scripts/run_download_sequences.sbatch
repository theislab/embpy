#!/bin/bash
#SBATCH --job-name=dl-seqs
#SBATCH --output=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/download_sequences_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/download_sequences_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=cpu_p
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

# ===========================================================================
# Pre-download all gene sequences (DNA + protein)
# ===========================================================================
# Run this ONCE before submitting embedding jobs.
# Downloads ~20k sequences from Ensembl/UniProt and saves to disk.
# Subsequent embedding jobs load from disk (seconds vs hours).
#
# Usage:
#   sbatch run_download_sequences.sbatch
# ===========================================================================

set -eo pipefail

DATA_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings"
SCRIPT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts"

mkdir -p "${SCRIPT_DIR}/logs"

echo "=== Job ${SLURM_JOB_ID} started at $(date) ==="
echo "Node: $(hostname)"

set +u
eval "$(/ictstr01/home/icb/goncalo.pinto/tools/apps/mamba/bin/conda shell.bash hook 2>/dev/null)"
conda activate embpy
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "=== Step 1: Download full genomic DNA sequences ==="
python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type dna \
    --region full

echo "=== Step 2: Download exon sequences ==="
python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type dna \
    --region exons

echo "=== Step 3: Download intron sequences ==="
python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type dna \
    --region introns

echo "=== Step 4: Download canonical protein sequences ==="
python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type protein

echo "=== Step 5: Download all protein isoforms ==="
python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type protein_isoforms

echo "=== All downloads complete at $(date) ==="
