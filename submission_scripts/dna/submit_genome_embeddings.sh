#!/bin/bash
# Submit only the array tasks whose output files do not yet exist.
# Usage: bash submit_genome_embeddings.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_full_genome_embeddings.sbatch"
BASE_OUTPUT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings/gene_embeddings/dna"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

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

NUM_POOLINGS=${#POOLINGS[@]}
NUM_REGIONS=${#REGIONS[@]}

MISSING_IDS=()

for model_idx in "${!MODELS[@]}"; do
    for pooling_idx in "${!POOLINGS[@]}"; do
        for region_idx in "${!REGIONS[@]}"; do
            task_id=$(( model_idx * NUM_POOLINGS * NUM_REGIONS + pooling_idx * NUM_REGIONS + region_idx ))
            output_file="${BASE_OUTPUT_DIR}/${MODELS[$model_idx]}/${REGIONS[$region_idx]}_${POOLINGS[$pooling_idx]}.npz"

            if [[ ! -f "${output_file}" ]]; then
                MISSING_IDS+=("${task_id}")
                echo "  [MISSING] task ${task_id}: ${MODELS[$model_idx]} / ${REGIONS[$region_idx]} / ${POOLINGS[$pooling_idx]}"
            else
                echo "  [DONE]    task ${task_id}: ${MODELS[$model_idx]} / ${REGIONS[$region_idx]} / ${POOLINGS[$pooling_idx]}"
            fi
        done
    done
done

if [[ ${#MISSING_IDS[@]} -eq 0 ]]; then
    echo ""
    echo "All 48 tasks are complete. Nothing to submit."
    exit 0
fi

ARRAY_SPEC=$(IFS=,; echo "${MISSING_IDS[*]}")

echo ""
echo "${#MISSING_IDS[@]} of 48 tasks still need to run."
echo "Array spec: --array=${ARRAY_SPEC}"

if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    echo "[DRY RUN] Would run: sbatch --array=${ARRAY_SPEC} ${SBATCH_FILE}"
else
    echo ""
    echo "Submitting..."
    sbatch --array="${ARRAY_SPEC}" "${SBATCH_FILE}"
fi
