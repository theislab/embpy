#!/bin/bash
#SBATCH --job-name=mol_emb
#SBATCH --output=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/mol_emb_%A_%a.out
#SBATCH --error=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs/mol_emb_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --array=0-19

# ── Configuration ──────────────────────────────────────────────────────
SMILES_FILE="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings/rdkit_200.csv"
OUTPUT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings/drug_embeddings"
SCRIPT_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/molecule"
LOG_DIR="/lustre/groups/ml01/workspace/goncalo.pinto/embpy/submission_scripts/logs"
BATCH_SIZE=64

mkdir -p "${LOG_DIR}"

# ── Activate conda environment ────────────────────────────────────────
set +u
eval "$(conda shell.bash hook)"
conda activate /ictstr01/home/icb/goncalo.pinto/tools/apps/mamba/envs/embpy
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ── Model + pooling configurations ────────────────────────────────────
# Transformer models: mean, max, cls pooling (3 x 3 = 9 jobs)
# Fingerprint models: flat pooling only (8 x 1 = 8 jobs)
# GNN models: mean pooling only (3 x 1 = 3 jobs)
# Total: 20 array tasks (0-19)

CONFIGS=(
    # Transformer models -- 3 pooling strategies each
    "chemberta2MTR    mean"
    "chemberta2MTR    max"
    "chemberta2MTR    cls"
    "chemberta2MLM    mean"
    "chemberta2MLM    max"
    "chemberta2MLM    cls"
    "molformer_base   mean"
    "molformer_base   max"
    "molformer_base   cls"
    # Fingerprint models -- flat only
    "rdkit_fp         flat"
    "morgan_fp        flat"
    "morgan_count_fp  flat"
    "maccs_fp         flat"
    "atom_pair_fp     flat"
    "atom_pair_count_fp flat"
    "torsion_fp       flat"
    "torsion_count_fp flat"
    # GNN models -- graph-level readout (no token pooling)
    "minimol          graph"
    "mhg_gnn          graph"
    "mole             graph"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL=$(echo "${CONFIG}" | awk '{print $1}')
POOLING=$(echo "${CONFIG}" | awk '{print $2}')

echo "======================================================"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL               = ${MODEL}"
echo "POOLING              = ${POOLING}"
echo "SMILES_FILE          = ${SMILES_FILE}"
echo "OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "BATCH_SIZE           = ${BATCH_SIZE}"
echo "======================================================"

python -u "${SCRIPT_DIR}/embed_small_molecule.py" \
    --smiles-file "${SMILES_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --pooling "${POOLING}" \
    --batch-size "${BATCH_SIZE}" \
    --device auto
