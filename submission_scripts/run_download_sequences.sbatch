#!/bin/bash
#SBATCH --job-name=dl-seqs
#SBATCH --output=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/download_sequences_%A_%a.out
#SBATCH --error=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/download_sequences_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --partition=cpu_p
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --array=0-4

# ===========================================================================
# Production sequence download pipeline
# ===========================================================================
# Uses bulk FTP download from Ensembl + local pysam extraction.
# No REST API calls for DNA -- genome FASTA is downloaded once (~900 MB)
# and sequences are extracted locally via indexed random access.
#
# Task 0: Full genomic DNA (via pysam + genome FASTA)
# Task 1: Exon DNA (via pysam + pyensembl gene structure)
# Task 2: Intron DNA (via pysam + pyensembl gene structure)
# Task 3: Canonical protein sequences (via UniProt bulk FASTA)
# Task 4: All protein isoforms (via UniProt REST API -- already works)
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
export POLARS_SKIP_CPU_CHECK=1

CONFIGS=(
    "dna --region full"
    "dna --region exons"
    "dna --region introns"
    "protein"
    "protein_isoforms"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Running: --type ${CONFIG} ==="
echo "=== Method: bulk FTP download + local extraction ==="

python -u "${SCRIPT_DIR}/download_sequences.py" \
    --data-dir "${DATA_DIR}" \
    --type ${CONFIG}

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete at $(date) ==="
