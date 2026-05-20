#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM launcher: JupyterLab in the pixi `evo2` env on an A100 GPU node.
#
# This is a sibling of jupyter_pixi.sbatch but pinned to:
#   * pixi env  = evo2  (Evo 2 + Vortex + Transformer Engine + flash-attn)
#   * GPU       = A100 80 GB (SM 8.0; needed for Vortex BF16 inference)
#   * RAM       = 120 G  (Evo 2 7B in BF16 needs ~14 GB VRAM but ~60-80 GB RAM
#                          for tokenizer caches and Vortex graph compilation)
#
# Usage:
#   sbatch submission_scripts/jupyter_pixi_evo2.sbatch
#
# Connection info (URL + token) is written to:
#   submission_scripts/logs/jupyter_pixi_evo2_<JOBID>.out
#
# One-time setup on the login node BEFORE the first sbatch:
#   cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy
#   pixi install -e evo2
# -----------------------------------------------------------------------------

#SBATCH --job-name=embpy-evo2-jpt
#SBATCH -o logs/jupyter_pixi_evo2_%j.out
#SBATCH -e logs/jupyter_pixi_evo2_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --time=12:00:00

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
                 JupyterLab (pixi evo2 env, A100 80GB) -- connection info
################################################################################

Compute node : ${HOSTNAME}
Port         : ${PORT}
Token        : ${JUPYTER_TOKEN}
Project dir  : ${PROJECT_DIR}
Pixi env     : evo2  (.pixi/envs/evo2/)

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
# Verifies that:
#   1. nvidia-smi sees an A100
#   2. torch can use CUDA
#   3. evo2 imports successfully (catches missing flash-attn / TE wheels early)
# -----------------------------------------------------------------------------
echo "----- nvidia-smi -----"
nvidia-smi || true
echo "----- torch sees:  -----"
pixi run -e evo2 python -c \
  "import torch; cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None; \
   print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()} sm={cap}')"
echo "----- evo2 imports: -----"
pixi run -e evo2 python -c \
  "from evo2 import Evo2; print('evo2 import: OK')" || \
  echo "WARNING: evo2 failed to import. Re-run 'pixi install -e evo2' on the login node."

# -----------------------------------------------------------------------------
# Launch JupyterLab inside the pixi evo2 env.
# -----------------------------------------------------------------------------
exec pixi run -e evo2 -- jupyter-lab \
    --port=${PORT} \
    --no-browser \
    --ip=0.0.0.0 \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --notebook-dir="${PROJECT_DIR}"
