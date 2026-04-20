#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM launcher: JupyterLab running inside the pixi `gpu` env on a GPU node.
#
# Usage (from the repo root, where pixi.toml lives):
#
#   sbatch submission_scripts/jupyter_pixi.sbatch
#
# After it starts, the connection info is printed to slurm_jupyter_<JOBID>.txt
# (URL + token + exact SSH tunnel command for your laptop).
#
# One-time setup on the login node BEFORE the first sbatch:
#
#   cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy
#   pixi install -e gpu        # resolves + downloads ~1 GB, only done once
# -----------------------------------------------------------------------------

#SBATCH --job-name=embpy-jpt
#SBATCH -o slurm_jupyter_%j.txt
#SBATCH -e slurm_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# #SBATCH --constraint=a100_80gb

# -----------------------------------------------------------------------------
# Environment hygiene: do NOT source conda/mamba here. pixi is fully isolated
# and any leftover `mamba activate` would shadow pixi's python.
# -----------------------------------------------------------------------------
set -euo pipefail

# Make pixi available in non-interactive SLURM shells.
export PATH="$HOME/.pixi/bin:$PATH"

# Keep user-site-packages from leaking into the pixi env.
export PYTHONNOUSERSITE=1

# Project containing pixi.toml (edit if you cloned it elsewhere).
PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# Pick a random port and a strong token.
PORT=$(/usr/bin/shuf -i 8000-9000 -n 1)
export JUPYTER_TOKEN=$(openssl rand -hex 32)

# -----------------------------------------------------------------------------
# Write connection instructions to the output log BEFORE Jupyter starts.
# -----------------------------------------------------------------------------
LOG="slurm_jupyter_${SLURM_JOB_ID}.txt"
cat <<END >> "$LOG"

################################################################################
                 JupyterLab (pixi gpu env) -- connection info
################################################################################

Compute node : ${HOSTNAME}
Port         : ${PORT}
Token        : ${JUPYTER_TOKEN}
Project dir  : ${PROJECT_DIR}
Pixi env     : gpu  (.pixi/envs/gpu/)

-- Step 1: from your laptop, open an SSH tunnel ------------------------------
   ssh -L ${PORT}:${HOSTNAME}:${PORT} ${USER}@hpc-build01.scidom.de

-- Step 2: open in your browser ----------------------------------------------
   http://localhost:${PORT}/?token=${JUPYTER_TOKEN}

-- Step 3: when done ---------------------------------------------------------
   scancel ${SLURM_JOB_ID}

END

# -----------------------------------------------------------------------------
# Smoke-check GPU is visible to the pixi env before launching Jupyter.
# -----------------------------------------------------------------------------
echo "----- nvidia-smi -----" >> "$LOG"
nvidia-smi >> "$LOG" 2>&1 || true
echo "----- torch sees:  -----" >> "$LOG"
pixi run -e gpu python -c \
  "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')" \
  >> "$LOG" 2>&1

# -----------------------------------------------------------------------------
# Launch JupyterLab inside the pixi gpu env. No LD_LIBRARY_PATH hack needed --
# conda-forge's pytorch-cuda package plus pixi's activation script take care
# of CUDA_HOME, LD_LIBRARY_PATH, and NVIDIA toolkit paths automatically.
# -----------------------------------------------------------------------------
exec pixi run -e gpu jupyter lab \
    --port=${PORT} \
    --no-browser \
    --ip=0.0.0.0 \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --notebook-dir="${PROJECT_DIR}"
