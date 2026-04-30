#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM launcher: JupyterLab running inside the pixi `gpu` env on a GPU node.
#
# Usage (from the repo root, where pixi.toml lives):
#
#   sbatch submission_scripts/jupyter_pixi.sbatch
#
# After it starts, the connection info (URL + token) is written to:
#   submission_scripts/logs/jupyter_pixi_<JOBID>.out
#
# To connect from VSCode / Cursor (Remote-SSH on the login node):
#   1. `cat` the .out file to get the URL.
#   2. Open any .ipynb, click the kernel picker (top right).
#   3. "Select Another Kernel..." -> "Existing Jupyter Server..."
#   4. Paste the URL exactly as printed (uses the COMPUTE NODE hostname).
#   Cursor's Remote-SSH routes the traffic automatically -- no ssh -L needed.
#
# One-time setup on the login node BEFORE the first sbatch:
#   cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy
#   pixi install -e gpu        # resolves + downloads ~1 GB, only done once
# -----------------------------------------------------------------------------

#SBATCH --job-name=embpy-jpt
#SBATCH -o logs/jupyter_pixi_%j.out
#SBATCH -e logs/jupyter_pixi_%j.err
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
# Write connection instructions to stdout (SLURM captures into the .out file).
# -----------------------------------------------------------------------------
cat <<END

################################################################################
                 JupyterLab (pixi gpu env) -- connection info
################################################################################

Compute node : ${HOSTNAME}
Port         : ${PORT}
Token        : ${JUPYTER_TOKEN}
Project dir  : ${PROJECT_DIR}
Pixi env     : gpu  (.pixi/envs/gpu/)

--- CONNECT FROM VSCode / Cursor (Remote-SSH to the login node) --------------

  1. Open any .ipynb, click the kernel picker (top right).
  2. "Select Another Kernel..." -> "Existing Jupyter Server..."
  3. Paste this URL exactly (compute node hostname, NOT localhost):

       http://${HOSTNAME}:${PORT}/?token=${JUPYTER_TOKEN}

  Cursor's Remote-SSH connection reaches ${HOSTNAME} automatically -- no
  manual ssh tunnel needed.

--- CONNECT FROM A BROWSER ON YOUR LAPTOP ------------------------------------

  1. From your laptop, open an SSH tunnel to the compute node:

       ssh -L ${PORT}:${HOSTNAME}:${PORT} ${USER}@hpc-build01.scidom.de

  2. Open this URL in your browser:

       http://localhost:${PORT}/?token=${JUPYTER_TOKEN}

--- WHEN DONE ----------------------------------------------------------------

  scancel ${SLURM_JOB_ID}

END

# -----------------------------------------------------------------------------
# Smoke-check GPU is visible to the pixi env before launching Jupyter.
# -----------------------------------------------------------------------------
echo "----- nvidia-smi -----"
nvidia-smi || true
echo "----- torch sees:  -----"
pixi run -e gpu python -c \
  "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')"

# -----------------------------------------------------------------------------
# Launch JupyterLab inside the pixi gpu env.
#
# Note: `pixi run -e gpu jupyter ...` would invoke the `jupyter` TASK defined
# in pixi.toml (which already supplies `lab --no-browser --ip 0.0.0.0`) and
# duplicate our flags. Use `pixi run -e gpu -- jupyter-lab ...` to run the
# binary directly instead of the task.
#
# No LD_LIBRARY_PATH hack needed -- conda-forge's pytorch-cuda package plus
# pixi's activation script take care of CUDA_HOME, LD_LIBRARY_PATH, and NVIDIA
# toolkit paths automatically.
# -----------------------------------------------------------------------------
exec pixi run -e gpu -- jupyter-lab \
    --port=${PORT} \
    --no-browser \
    --ip=0.0.0.0 \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --notebook-dir="${PROJECT_DIR}"
