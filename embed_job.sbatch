#!/bin/bash
#SBATCH --job-name=embpy_gen
#SBATCH --output=err/embpy_gen_output_%A.txt
#SBATCH --error=err/embpy_gen_error_%A.txt
#SBATCH --ntasks=2
#SBATCH --time=3:00:00
#SBATCH --mem=500Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=0
# #SBATCH --constraint="a100_80gb|a100_40gb|h100_80gb"
# #SBATCH --array=0-2

# If you ever want to run multiple seeds or variants, you could enable an array:
# #SBATCH --array=0-2

# Ensure log dir exists
mkdir -p err

# Init shell + env
source "$HOME/.bashrc"
mamba activate embpy

echo "▶ Starting MoLFormer embeddings on $(hostname) at $(date)"
echo "   GPUs: $CUDA_VISIBLE_DEVICES"
echo "   Python: $(which python)"
python -c "import torch; print('   PyTorch CUDA:', torch.cuda.is_available())"

# Paths
PYFILE=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/run_generation_molformer.py
CSV_IN=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/rdkit_200.csv
OUTDIR=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/output

# Run (choose poolings as you like: cls mean max)
python -u "$PYFILE" \
  --csv "$CSV_IN" \
  --smiles-col smiles \
  --model molformer_base \
  --poolings cls mean max \
  --outdir "$OUTDIR"

echo "✅ Finished at $(date)"
