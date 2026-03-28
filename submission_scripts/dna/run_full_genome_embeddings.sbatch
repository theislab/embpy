#!/bin/bash
#SBATCH --job-name=genome-emb
#SBATCH --output=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/genome_emb_%A_%a.out
#SBATCH --error=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/genome_emb_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --array=0-47

# ===========================================================================
# Full Genome DNA Embedding Array Job
# ===========================================================================
# Generates embeddings for ALL protein-coding genes in the human genome
# across multiple DNA models, pooling strategies, and gene regions.
#
# Usage:
#   sbatch run_full_genome_embeddings.sbatch
#
# Output directory structure:
#   data/dna_embeddings/
#     enformer_human_rough/
#       full_mean.npz
#       full_max.npz
#       exons_mean.npz
#       exons_max.npz
#       introns_mean.npz
#       introns_max.npz
#     borzoi_v0/
#       ...
#     evo2_7b/
#       ...
# ===========================================================================

set -euo pipefail

# ---- Configuration ---------------------------------------------------------
BASE_OUTPUT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/dna_embeddings"
SCRIPT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/dna"
LOG_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs"

mkdir -p "${LOG_DIR}"

# Define all configurations as arrays
# Models: enformer, borzoi (x2), evo2 (x4), evo1 (x5), nt (x8), gena (x4), hyenadna (x5), caduceus (x2), ntv3 (x5)
# For this array job, we focus on the main models with region/pooling combos

MODELS=(
    "enformer_human_rough"
    "borzoi_v0"
    "borzoi_v1"
    "evo2_7b"
    "evo2_7b_base"
    "evo1_8k"
    "evo1_131k"
    "nt_v2_500m"
)

POOLINGS=("mean" "max")
REGIONS=("full" "exons" "introns")

# Calculate total combinations: 8 models x 2 poolings x 3 regions = 48
# Array index 0-47

# Parse array task ID into model/pooling/region indices
NUM_MODELS=${#MODELS[@]}
NUM_POOLINGS=${#POOLINGS[@]}
NUM_REGIONS=${#REGIONS[@]}

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_POOLINGS * NUM_REGIONS) ))
REMAINING=$(( SLURM_ARRAY_TASK_ID % (NUM_POOLINGS * NUM_REGIONS) ))
POOLING_IDX=$(( REMAINING / NUM_REGIONS ))
REGION_IDX=$(( REMAINING % NUM_REGIONS ))

MODEL="${MODELS[$MODEL_IDX]}"
POOLING="${POOLINGS[$POOLING_IDX]}"
REGION="${REGIONS[$REGION_IDX]}"

# Output path
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL}"
OUTPUT_FILE="${OUTPUT_DIR}/${REGION}_${POOLING}.npz"

# Skip if already exists
if [[ -f "${OUTPUT_FILE}" ]]; then
    echo "Output already exists: ${OUTPUT_FILE}"
    echo "Skipping..."
    exit 0
fi

# ---- Environment setup -----------------------------------------------------
echo "=== Job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID}) started at $(date) ==="
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Configuration: model=${MODEL}, pooling=${POOLING}, region=${REGION}"
echo "Output: ${OUTPUT_FILE}"

# Activate conda environment (set +u needed for conda activation scripts)
set +u
eval "$(/ictstr01/home/icb/goncalo.pinto/tools/apps/mamba/bin/conda shell.bash hook 2>/dev/null)"
conda activate embpy
set -u

# ---- Run Python script -----------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

python -u "${SCRIPT_DIR}/embed_full_genome.py" \
    --model "${MODEL}" \
    --pooling "${POOLING}" \
    --region "${REGION}" \
    --output "${OUTPUT_FILE}" \
    --organism human \
    --biotype protein_coding \
    --device auto \
    --data-dir "/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data"

echo "=== Job finished at $(date) ==="
