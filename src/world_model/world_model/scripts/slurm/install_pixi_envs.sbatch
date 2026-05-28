#!/bin/bash
# -----------------------------------------------------------------------------
# Install one or more pixi environments on a GPU node.
#
# `pixi install` for envs that ship CUDA-built PyPI wheels (mamba-ssm,
# causal-conv1d, flash-attn, transformer_engine, ...) needs a GPU node so
# the system-requirements `cuda = "12"` constraint resolves cleanly, and
# enough memory/CPU/time for pip to fall back to compiling from source if
# no matching wheel is published.
#
# Usage:
#   sbatch --export=ALL,ENVS="caduceus evo2" \
#          src/world_model/world_model/scripts/slurm/install_pixi_envs.sbatch
#
# ENVS is a space-separated list of environment names from pixi.toml.
# -----------------------------------------------------------------------------
#SBATCH --job-name=pixi-install
#SBATCH -o logs/slurm/install_pixi_envs/%x_%j.out
#SBATCH -e logs/slurm/install_pixi_envs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

PROJECT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy"
cd "$PROJECT_DIR"

# Critical: redirect TMPDIR off the node's /tmp (which has been filling up
# during parallel pixi installs). Lustre is slower per-syscall but has
# orders of magnitude more space available.
export TMPDIR="${PROJECT_DIR}/.tmp/pixi-install-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
# Keep TMPDIR around if the build fails so the user can inspect logs;
# only delete on a clean exit.
trap '[ "$?" -eq 0 ] && rm -rf "$TMPDIR" || echo "Build dir retained: $TMPDIR"' EXIT

# pip is happier with a few extra knobs when building CUDA kernels:
#   * MAX_JOBS lets ninja parallelize but not exhaust 128G RAM.
#   * TORCH_CUDA_ARCH_LIST narrows compile to Ampere+Hopper (matches the
#     a100_80gb|h100_80gb constraint above).
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export PYTHONNOUSERSITE=1

ENVS="${ENVS:-caduceus evo2}"
echo "=== pixi install for envs: $ENVS ==="
echo "  TMPDIR              = $TMPDIR"
echo "  TORCH_CUDA_ARCH_LIST= $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS            = $MAX_JOBS"
echo "  GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

for env in $ENVS; do
    echo
    echo "============================================"
    echo "  installing pixi env: $env"
    echo "============================================"
    # `pixi install` resolves + downloads + installs and is idempotent.
    # If pixi.lock already contains a resolution for this env, the install
    # is a fast copy from the global pixi cache.
    time pixi install -e "$env"
    echo
    echo "  smoke test: 'pixi run -e $env python -c \"import torch; print(torch.cuda.is_available())\"'"
    pixi run -e "$env" -- python -c "import torch; print(f'env={\"$env\"} torch={torch.__version__} cuda={torch.cuda.is_available()}')"
done

# Per-env import smoke tests for the things this env actually needs to load
# at inference time.
if echo "$ENVS" | grep -qw caduceus; then
    echo
    echo "=== caduceus smoke test ==="
    pixi run -e caduceus -- python -c "
import mamba_ssm, causal_conv1d, transformers
print('mamba_ssm   :', mamba_ssm.__version__)
print('causal_conv1d:', causal_conv1d.__version__)
print('transformers:', transformers.__version__)
"
fi

if echo "$ENVS" | grep -qw evo2; then
    echo
    echo "=== evo2 smoke test ==="
    pixi run -e evo2 -- python -c "
import evo2, mamba_ssm
print('evo2       :', evo2.__version__)
print('mamba_ssm  :', mamba_ssm.__version__)
"
fi

echo
echo "=== ALL ENVS INSTALLED OK ==="
