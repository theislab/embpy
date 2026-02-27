#!/bin/bash
#SBATCH --job-name=embpy-gene
#SBATCH --output=logs/gene_embed_%j.out
#SBATCH --error=logs/gene_embed_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

# ===========================================================================
# Gene (DNA) Embedding SLURM Submission Script
# ===========================================================================
# Usage:
#   sbatch run_gene_embedding.sbatch
#
# Override variables by exporting them before sbatch:
#   export INPUT_MODE=biomart MART_FILE=/path/to/genes.csv GENOME_DIR=/path/to/genome
#   sbatch run_gene_embedding.sbatch
# ===========================================================================

set -euo pipefail
mkdir -p logs

# ---- Configuration (override via environment variables) --------------------
INPUT_MODE="${INPUT_MODE:-biomart}"              # biomart | single | all | adata
MODEL="${MODEL:-enformer_human_rough}"           # DNA model name
POOLING="${POOLING:-mean}"                       # mean | max | cls
DEVICE="${DEVICE:-auto}"                         # auto | cpu | cuda
OUTPUT_FORMAT="${OUTPUT_FORMAT:-npz}"            # npz | csv
BIOTYPE="${BIOTYPE:-protein_coding}"             # protein_coding | all
ORGANISM="${ORGANISM:-human}"

# Biomart mode
MART_FILE="${MART_FILE:-}"                       # Path to BioMart CSV
GENOME_DIR="${GENOME_DIR:-}"                     # Path to chromosome FASTA dir

# Single mode
IDENTIFIER="${IDENTIFIER:-}"                     # Gene name, Ensembl ID, or DNA seq

# AnnData mode
ADATA_PATH="${ADATA_PATH:-}"                     # Path to .h5ad file
GENE_COLUMN="${GENE_COLUMN:-}"                   # Column name in adata.var

# Output
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy/output/genes}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/${MODEL}_${INPUT_MODE}_${POOLING}.${OUTPUT_FORMAT}}"

# ---- Environment setup -----------------------------------------------------
echo "=== Job ${SLURM_JOB_ID} started at $(date) ==="
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"

# Activate your conda / venv / module environment here:
# source activate embpy_env
# module load cuda/12.1

# ---- Build command ----------------------------------------------------------
CMD="python -u $(dirname "$0")/embed_genes.py"
CMD+=" --input-mode ${INPUT_MODE}"
CMD+=" --model ${MODEL}"
CMD+=" --pooling ${POOLING}"
CMD+=" --device ${DEVICE}"
CMD+=" --output ${OUTPUT_FILE}"
CMD+=" --output-format ${OUTPUT_FORMAT}"
CMD+=" --biotype ${BIOTYPE}"
CMD+=" --organism ${ORGANISM}"

[[ -n "${MART_FILE}" ]]   && CMD+=" --mart-file ${MART_FILE}"
[[ -n "${GENOME_DIR}" ]]  && CMD+=" --genome-dir ${GENOME_DIR}"
[[ -n "${IDENTIFIER}" ]]  && CMD+=" --identifier '${IDENTIFIER}'"
[[ -n "${ADATA_PATH}" ]]  && CMD+=" --adata-path ${ADATA_PATH}"
[[ -n "${GENE_COLUMN}" ]] && CMD+=" --gene-column ${GENE_COLUMN}"

echo "Command: ${CMD}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# ---- Run --------------------------------------------------------------------
eval ${CMD}

echo "=== Job finished at $(date) ==="
