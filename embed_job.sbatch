#!/bin/bash
#SBATCH --job-name=embpy_gen
#SBATCH --output=err/embpy_gen_output_%A.txt
#SBATCH --error=err/embpy_gen_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=500Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --nice=0
# #SBATCH --constraint="a100_80gb|a100_40gb|h100_80gb"
# #SBATCH --array=0-2

# If you ever want to run multiple seeds or variants, you could enable an array:
# #SBATCH --array=0-2

# Initialize your shell and activate the embpy env
source $HOME/.bashrc
mamba activate embpy

echo "▶ Starting embeddings on $(hostname) at $(date)"
echo "   GPUs: $CUDA_VISIBLE_DEVICES"
echo "   Python: $(which python)"

# Run your generation script
python -u /lustre/groups/ml01/workspace/goncalo.pinto/embpy/run_generation_borzoi.py \
    --ccle       /lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/ccle_300.csv \
    --mart       /lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/mart_export-5.txt \
    --chrom_dir  /lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/chromosomes \
    --out_prefix /lustre/groups/ml01/workspace/goncalo.pinto/embpy/output/enformer_embeddings

echo "✅ Embeddings finished on $(hostname) at $(date)"
