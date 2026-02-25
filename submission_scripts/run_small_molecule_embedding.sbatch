#!/bin/bash
#SBATCH --job-name=mol_embeddings
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mol_embeddings_%j.log
#SBATCH --error=logs/mol_embeddings_%j.log

# ── Create log directory ────────────────────────────────────────────────────
mkdir -p logs

# ── Activate environment (EDIT THIS) ───────────────────────────────────────
# source activate your_env
# conda activate your_env

# ── Run embedding generation ───────────────────────────────────────────────
python scripts/small_molecule_embedding_generation.py \
    --smiles-file data/unique_smiles.parquet \
    --output-dir  data/embeddings \
    --batch-size  64 \
    --device      auto
