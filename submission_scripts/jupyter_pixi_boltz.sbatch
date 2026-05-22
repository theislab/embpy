#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM launcher: JupyterLab running inside the pixi `boltz` env on a GPU node.
#
# WHY A SEPARATE SCRIPT?
#   Boltz-2 (structure-informed protein embeddings) hard-pins numpy<2.0,
#   scipy==1.13.1, scikit-learn==1.6.1 etc., which conflict with the modern
#   numpy-2.x stack in the main `gpu` env. It therefore lives in its own pixi
#   environment (`boltz`) with its own launcher. Notebook
#   04_boltz2_structure_embeddings.ipynb is the part of the repo that needs it
#   (BioEmbedder model keys: boltz2 / boltz2_pairwise / boltz2_both).
#
# Usage (from the repo root, where pixi.toml lives):
#
#   sbatch submission_scripts/jupyter_pixi_boltz.sbatch
#
# After it starts, the connection info (URL + token) is written to:
#   submission_scripts/logs/jupyter_pixi_boltz_<JOBID>.out
#
# To connect from VSCode / Cursor (Remote-SSH on the login node):
#   1. `cat` the .out file to get the URL.
#   2. Open notebook 04, click the kernel picker (top right).
#   3. "Select Another Kernel..." -> "Existing Jupyter Server..."
#   4. Paste the URL exactly as printed (uses the COMPUTE NODE hostname).
#
# One-time setup on the login node BEFORE the first sbatch:
#   cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy
#   pixi install -e boltz      # resolves boltz[cuda] + pinned deps (~2 GB)
#   # First notebook run also downloads the Boltz-2 checkpoint (~2 GB) and the
#   # CCD dictionary (~150 MB) into the HuggingFace / boltz cache.
# -----------------------------------------------------------------------------

#SBATCH --job-name=embpy-boltz
#SBATCH -o logs/jupyter_pixi_boltz_%j.out
#SBATCH -e logs/jupyter_pixi_boltz_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# #SBATCH --constraint=a100_80gb

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

PORT=$(/usr/bin/shuf -i 8000-9000 -n 1)
export JUPYTER_TOKEN=$(openssl rand -hex 32)

cat <<END

################################################################################
                JupyterLab (pixi boltz env) -- connection info
################################################################################

Compute node : ${HOSTNAME}
Port         : ${PORT}
Token        : ${JUPYTER_TOKEN}
Project dir  : ${PROJECT_DIR}
Pixi env     : boltz  (.pixi/envs/boltz/)

--- CONNECT FROM VSCode / Cursor (Remote-SSH to the login node) --------------

  1. Open notebook 04, click the kernel picker (top right).
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

echo "----- nvidia-smi -----"
nvidia-smi || true
echo "----- torch + boltz see: -----"
pixi run -e boltz python -c \
  "import torch, boltz, numpy; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()} boltz={boltz.__version__} numpy={numpy.__version__}')" \
  || echo "(smoke test failed -- Jupyter will still launch)"

# Launch JupyterLab inside the pixi boltz env. Use `-- jupyter-lab` (not the
# `jupyter` pixi task) so our flags are not duplicated.
exec pixi run -e boltz -- jupyter-lab \
    --port=${PORT} \
    --no-browser \
    --ip=0.0.0.0 \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --notebook-dir="${PROJECT_DIR}"