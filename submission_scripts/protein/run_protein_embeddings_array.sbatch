#!/bin/bash
#SBATCH --job-name=prot-emb
#SBATCH --output=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/prot_emb_%A_%a.out
#SBATCH --error=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/prot_emb_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --array=0-65

# ===========================================================================
# Full Proteome Protein Embedding Array Job
# ===========================================================================
# Generates protein embeddings for ALL protein-coding genes using
# multiple protein language models, pooling strategies, and isoform modes.
#
# Usage:
#   sbatch run_protein_embeddings_array.sbatch
#
# Output structure:
#   data/embeddings/protein_embeddings/
#     esm2_650M/
#       canonical_mean.npz
#       canonical_max.npz
#       canonical_cls.npz
#       all_mean.npz
#       ...
#     esmc_300m/
#       ...
# ===========================================================================

set -eo pipefail

BASE_OUTPUT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings/protein_embeddings"
SCRIPT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/protein"
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/.checkpoints"
LOG_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# Models x Poolings x Isoforms
# 11 models x 3 poolings x 2 isoform modes = 66 tasks (array 0-65)
MODELS=(
    "esm2_8M"
    "esm2_35M"
    "esm2_150M"
    "esm2_650M"
    "esm2_3B"
    "esmc_300m"
    "esmc_600m"
    "esm3_small"
    "esm1b"
    "prot_t5_xl"
    "prot_t5_xl_half"
)

POOLINGS=("mean" "max" "cls")
ISOFORMS=("canonical" "all")

NUM_MODELS=${#MODELS[@]}
NUM_POOLINGS=${#POOLINGS[@]}
NUM_ISOFORMS=${#ISOFORMS[@]}

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_POOLINGS * NUM_ISOFORMS) ))
REMAINING=$(( SLURM_ARRAY_TASK_ID % (NUM_POOLINGS * NUM_ISOFORMS) ))
POOLING_IDX=$(( REMAINING / NUM_ISOFORMS ))
ISOFORM_IDX=$(( REMAINING % NUM_ISOFORMS ))

MODEL="${MODELS[$MODEL_IDX]}"
POOLING="${POOLINGS[$POOLING_IDX]}"
ISOFORM="${ISOFORMS[$ISOFORM_IDX]}"

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL}"
OUTPUT_FILE="${OUTPUT_DIR}/${ISOFORM}_${POOLING}.npz"

# Skip if already exists
if [[ -f "${OUTPUT_FILE}" ]]; then
    echo "Output already exists: ${OUTPUT_FILE}"
    echo "Skipping..."
    exit 0
fi

echo "=== Job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID}) started at $(date) ==="
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Configuration: model=${MODEL}, pooling=${POOLING}, isoform=${ISOFORM}"
echo "Output: ${OUTPUT_FILE}"

# Activate conda environment
set +u
eval "$(/ictstr01/home/icb/goncalo.pinto/tools/apps/mamba/bin/conda shell.bash hook 2>/dev/null)"
conda activate embpy
set -u

mkdir -p "${OUTPUT_DIR}"

python -u "${SCRIPT_DIR}/embed_full_proteome.py" \
    --model "${MODEL}" \
    --pooling "${POOLING}" \
    --isoform "${ISOFORM}" \
    --output "${OUTPUT_FILE}" \
    --organism human \
    --biotype protein_coding \
    --device auto \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings"

echo "=== Job finished at $(date) ==="
