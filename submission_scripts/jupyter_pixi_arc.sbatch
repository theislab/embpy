#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM launcher: JupyterLab running inside the pixi `arc-gpu` env on a GPU
# node.
#
# WHY A SEPARATE SCRIPT?
#   The Arc Institute models (arc-state, arc-stack) require a newer scientific
#   stack (pytorch>=2.7, scipy>=1.15, transformers>=4.52, numpy>=2.2) than the
#   helical foundation-models stack can tolerate. They therefore live in their
#   own pixi environment (`arc-gpu`) with its own launcher script.
#
#   Notebook 08 (single-cell foundation models) section 13c -- STATE encode+
#   decode and Stack in-context generation -- is the only part of the repo
#   that needs this env. The other sections of notebook 08 run in the
#   `helical-gpu` env (see jupyter_pixi_helical.sbatch).
#
# Usage (from the repo root, where pixi.toml lives):
#
#   sbatch submission_scripts/jupyter_pixi_arc.sbatch
#
# After it starts, the connection info (URL + token) is written to:
#   submission_scripts/logs/jupyter_pixi_arc_<JOBID>.out
#
# One-time setup on the login node BEFORE the first sbatch:
#   cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy
#   CONDA_OVERRIDE_CUDA=12.0 pixi install -e arc-gpu
#   # (CONDA_OVERRIDE_CUDA lets the solver pick CUDA-enabled variants when
#   #  the login node itself is CPU-only.)
# -----------------------------------------------------------------------------

#SBATCH --job-name=embpy-arc
#SBATCH -o logs/jupyter_pixi_arc_%j.out
#SBATCH -e logs/jupyter_pixi_arc_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --constraint=a100_80gb

# -----------------------------------------------------------------------------
# Environment hygiene: do NOT source conda/mamba here. pixi is fully isolated
# and any leftover `mamba activate` would shadow pixi's python.
# -----------------------------------------------------------------------------
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

PORT=$(/usr/bin/shuf -i 8000-9000 -n 1)
export JUPYTER_TOKEN=$(openssl rand -hex 32)

cat <<END

################################################################################
             JupyterLab (pixi arc-gpu env) -- connection info
################################################################################

Compute node : ${HOSTNAME}
Port         : ${PORT}
Token        : ${JUPYTER_TOKEN}
Project dir  : ${PROJECT_DIR}
Pixi env     : arc-gpu  (.pixi/envs/arc-gpu/)

--- CONNECT FROM VSCode / Cursor (Remote-SSH to the login node) --------------

  1. Open any .ipynb, click the kernel picker (top right).
  2. "Select Another Kernel..." -> "Existing Jupyter Server..."
  3. Paste this URL exactly (compute node hostname, NOT localhost):

       http://${HOSTNAME}:${PORT}/?token=${JUPYTER_TOKEN}

--- CONNECT FROM A BROWSER ON YOUR LAPTOP ------------------------------------

  1. From your laptop, open an SSH tunnel to the compute node:

       ssh -L ${PORT}:${HOSTNAME}:${PORT} ${USER}@hpc-build01.scidom.de

  2. Open this URL in your browser:

       http://localhost:${PORT}/?token=${JUPYTER_TOKEN}

--- WHEN DONE ----------------------------------------------------------------

  scancel ${SLURM_JOB_ID}

END

# -----------------------------------------------------------------------------
# Smoke-check GPU is visible to the arc-gpu env and arc-state / arc-stack
# imports cleanly. `|| echo ...` keeps `set -e` from killing the whole job if
# the smoke test hiccups on a version-lookup corner case -- Jupyter itself
# will still launch.
# -----------------------------------------------------------------------------
echo "----- nvidia-smi -----"
nvidia-smi || true
echo "----- torch + arc-state + arc-stack see:  -----"
pixi run -e arc-gpu python -c "
import torch
from importlib.metadata import version
def _v(pkg):
    try:
        return version(pkg)
    except Exception:
        return '(installed but version lookup failed)'
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} '
      f'gpus={torch.cuda.device_count()} '
      f'arc-state={_v(\"arc-state\")} arc-stack={_v(\"arc-stack\")}')
" || echo "(smoke test failed -- Jupyter will still launch)"

# -----------------------------------------------------------------------------
# Launch JupyterLab inside the pixi arc-gpu env.
# -----------------------------------------------------------------------------
exec pixi run -e arc-gpu -- jupyter-lab \
    --port=${PORT} \
    --no-browser \
    --ip=0.0.0.0 \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --notebook-dir="${PROJECT_DIR}"
