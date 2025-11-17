#!/bin/bash
#SBATCH --job-name=esmc_genome
#SBATCH --output=logs/esmc_genome_%j.out
#SBATCH --error=logs/esmc_genome_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=4

mkdir -p logs
source $HOME/.bashrc
mamba activate embpy

echo "Running ESMC Genome Embedding..."
python -u submission_scripts/esmc_embed.py
echo "Done."