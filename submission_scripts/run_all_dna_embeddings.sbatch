#!/bin/bash
#SBATCH --job-name=all-dna-emb
#SBATCH --output=logs/all_dna_emb_%j.out
#SBATCH --error=logs/all_dna_emb_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

# ===========================================================================
# Full DNA Embedding Pipeline - All Models/Poolings/Regions
# ===========================================================================
# Runs all DNA embedding configurations sequentially in a single job.
# Use this if you prefer fewer jobs running longer.
#
# For parallel execution, use run_full_genome_embeddings.sbatch instead.
#
# Usage:
#   sbatch run_all_dna_embeddings.sbatch
# ===========================================================================

set -euo pipefail
mkdir -p logs

BASE_OUTPUT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/dna_embeddings"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/.checkpoints"

echo "=== Job ${SLURM_JOB_ID} started at $(date) ==="
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"

# Activate conda environment (adjust as needed)
# source /path/to/conda/bin/activate embpy

# Define configurations
MODELS=(
    "enformer_human_rough"
    "borzoi_v0"
    "borzoi_v1"
    "evo2_7b"
    "evo2_7b_base"
    "evo1_8k"
    "evo1_131k"
    "nt_v2_500m"
    "nt_v2_250m"
    "nt_v2_100m"
    "gena_lm_bert_base"
    "hyenadna_medium_160k"
    "caduceus_ph_131k"
)

POOLINGS=("mean" "max")
REGIONS=("full" "exons" "introns")

TOTAL_CONFIGS=$(( ${#MODELS[@]} * ${#POOLINGS[@]} * ${#REGIONS[@]} ))
CURRENT=0

for MODEL in "${MODELS[@]}"; do
    for POOLING in "${POOLINGS[@]}"; do
        for REGION in "${REGIONS[@]}"; do
            CURRENT=$((CURRENT + 1))
            OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL}"
            OUTPUT_FILE="${OUTPUT_DIR}/${REGION}_${POOLING}.npz"

            echo ""
            echo "=========================================="
            echo "[$CURRENT/$TOTAL_CONFIGS] Model: $MODEL | Pooling: $POOLING | Region: $REGION"
            echo "=========================================="

            if [[ -f "${OUTPUT_FILE}" ]]; then
                echo "Already exists: ${OUTPUT_FILE}, skipping..."
                continue
            fi

            mkdir -p "${OUTPUT_DIR}"

            python -u "${SCRIPT_DIR}/embed_full_genome.py" \
                --model "${MODEL}" \
                --pooling "${POOLING}" \
                --region "${REGION}" \
                --output "${OUTPUT_FILE}" \
                --organism human \
                --biotype protein_coding \
                --device auto \
                --checkpoint-dir "${CHECKPOINT_DIR}" \
                || echo "WARNING: Failed for ${MODEL}/${REGION}/${POOLING}"

            echo "Completed at $(date)"
        done
    done
done

echo ""
echo "=== All configurations completed at $(date) ==="
