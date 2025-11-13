#!/bin/bash
#SBATCH --job-name=embpy_gen
#SBATCH --output=err/embpy_gen_output_%A_%a.txt
#SBATCH --error=err/embpy_gen_error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=500Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=0

# ------- paths & knobs (edit if needed) -------
INPUT_DIR="/lustre/groups/ml01/workspace/goncalo/pinto/embpy/data/missing_genes_per_embeddings"
OUT_DIR_BASE="${INPUT_DIR}/embeddings_out"
DEVICE="cuda"     # use the GPU you requested
ENV_NAME="embpy"

# --- Model-specific knobs ---

# Enformer/Borzoi
ENFORMER_ID="EleutherAI/enformer-official-rough"
BORZOI_ID="johahi/borzoi-replicate-0"

# ESMC
ESMC_SCRIPT_PATH="submission_scripts/embed_all_genes.py" 
ESMC_POOLING="mean"               # Pooling for --pooling

# ----------------------------------------------

# Which model to run: enformer | borzoi | esmc_300m | esmc_600m (default enformer)
MODEL="${1:-enformer}"

# ----------------------------------------------

# Ensure log dir exists
mkdir -p err

# Init shell + env
source "$HOME/.bashrc"
mamba activate "$ENV_NAME"

echo "▶ Starting ${MODEL} embeddings on $(hostname) at $(date)"
echo "   GPU(s): $CUDA_VISIBLE_DEVICES"
echo "   Python: $(which python)"
python - <<'PY'
import torch
print("   PyTorch:", torch.__version__)
print("   CUDA available:", torch.cuda.is_available())
print("   Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("   Current device:", torch.cuda.current_device())
    print("   Device name:", torch.cuda.get_device_name(0))
PY

# Pick input glob by model
# Note: ESMC does not use a GLOB
if [[ "$MODEL" == "enformer" ]]; then
  GLOB="*enformer*.fasta"
elif [[ "$MODEL" == "borzoi" ]]; then
  GLOB="*borzoi*.fasta"
elif [[ "$MODEL" == "esmc_300m" || "$MODEL" == "esmc_600m" ]]; then
  echo "→ ESMC model selected, no input FASTA glob needed (uses pyensembl)."
  GLOB="N/A" # Set to N/A
else
  echo "ERROR: Unknown MODEL '$MODEL'. Use 'enformer', 'borzoi', 'esmc_300m', or 'esmc_600m'." >&2
  exit 2
fi

# Resolve output dir per model
OUT_DIR="${OUT_DIR_BASE}/${MODEL}"
mkdir -p "$OUT_DIR"

# Quick sanity: list files that match
# --- MODIFIED: Skip this check for ESMC ---
if [[ "$MODEL" != "esmc_300m" && "$MODEL" != "esmc_600m" ]]; then
  echo "→ Looking for FASTA files in: $INPUT_DIR with pattern: $GLOB"
  MATCH_COUNT=$(ls -1 ${INPUT_DIR}/${GLOB} 2>/dev/null | wc -l)
  echo "   Matching files: $MATCH_COUNT"
  if [[ "$MATCH_COUNT" -eq 0 ]]; then
    echo "ERROR: No input FASTA files found matching '${GLOB}' in ${INPUT_DIR}" >&2
    exit 3
  fi
fi

# Run
if [[ "$MODEL" == "enformer" ]]; then
  echo "→ Running Enformer"
  python -u submission_scripts/enformer_embed.py \
    --input_dir "$INPUT_DIR" \
    --out_dir   "$OUT_DIR" \
    --device    "$DEVICE" \
    --enformer_id "$ENFORMER_ID" \
    --glob "$GLOB"

elif [[ "$MODEL" == "borzoi" ]]; then
  echo "→ Running Borzoi"
  python -u submission_scripts/borzoi_embed.py \
    --input_dir "$INPUT_DIR" \
    --out_dir   "$OUT_DIR" \
    --device    "$DEVICE" \
    --borzoi_id "$BORZOI_ID" \
    --glob "$GLOB"

# --- NEW BLOCKS FOR ESMC ---
elif [[ "$MODEL" == "esmc_300m" ]]; then
  echo "→ Running ESMC 300M (all human genes from pyensembl)"
  OUTPUT_FILENAME="${OUT_DIR}/esmc_gene_embeddings.csv"
  echo "   Output File: ${OUTPUT_FILENAME}"

  python -u "$ESMC_SCRIPT_PATH" \
    --models "esmc_300m" \
    --pooling "$ESMC_POOLING" \
    --output "${OUTPUT_FILENAME}"

elif [[ "$MODEL" == "esmc_600m" ]]; then
  echo "→ Running ESMC 600M (all human genes from pyensembl)"
  OUTPUT_FILENAME="${OUT_DIR}/esmc_gene_embeddings.csv"
  echo "   Output File: ${OUTPUT_FILENAME}"

  python -u "$ESMC_SCRIPT_PATH" \
    --models "esmc_600m" \
    --pooling "$ESMC_POOLING" \
    --output "${OUTPUT_FILENAME}"
fi

echo "✅ Finished at $(date)"