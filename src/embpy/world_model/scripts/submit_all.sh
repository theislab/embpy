#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Submit the three training setups, the baseline run, and the final
# compare / report job with one command. Uses sbatch --dependency=afterok:
# to chain baselines + report after each training job finishes.
#
# Usage (from the repo root):
#   bash src/embpy/world_model/scripts/submit_all.sh
#
# Override the SLURM partition / qos via env:
#   PARTITION=gpu_p QOS=gpu_normal bash .../submit_all.sh
# Override the transfer fraction:
#   FRACTION=0.05 bash .../submit_all.sh
# Skip a setup (default: run all three):
#   SKIP_NADIG=1 SKIP_REPLOGLE=1 SKIP_TRANSFER=1 bash .../submit_all.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SLURM_DIR="src/embpy/world_model/scripts/slurm"
mkdir -p logs

# Map setup -> output_dir: hard-coded to match the values in
# configs/experiments/*.yaml so we do not have to parse YAML in bash.
declare -A SETUP_OUT
SETUP_OUT[single_nadig]="outputs/world_model/single_nadig"
SETUP_OUT[single_replogle]="outputs/world_model/single_replogle"

FRACTION="${FRACTION:-0.10}"
P_INT=$(python -c "print(int(float('${FRACTION}')*100))")
TRANSFER_RUN_NAME="transfer_nadig_to_replogle_p$(printf '%03d' $P_INT)"
SETUP_OUT[transfer]="outputs/world_model/${TRANSFER_RUN_NAME}"

declare -A TRAIN_JOB

if [[ "${SKIP_NADIG:-0}" != "1" ]]; then
    echo "Submitting single_nadig ..."
    TRAIN_JOB[single_nadig]=$(sbatch --parsable "${SLURM_DIR}/train_single_nadig.sbatch")
fi

if [[ "${SKIP_REPLOGLE:-0}" != "1" ]]; then
    echo "Submitting single_replogle ..."
    TRAIN_JOB[single_replogle]=$(sbatch --parsable "${SLURM_DIR}/train_single_replogle.sbatch")
fi

if [[ "${SKIP_TRANSFER:-0}" != "1" ]]; then
    echo "Submitting transfer (fraction=${FRACTION}) ..."
    TRAIN_JOB[transfer]=$(FRACTION="${FRACTION}" sbatch --parsable "${SLURM_DIR}/train_transfer.sbatch")
fi

declare -A BASELINE_JOB
declare -A COMPARE_JOB

for setup in "${!TRAIN_JOB[@]}"; do
    train_jid="${TRAIN_JOB[$setup]}"
    out_dir="${SETUP_OUT[$setup]}"
    echo "Chaining baselines (after ${train_jid}) for ${setup} ..."
    BASELINE_JOB[$setup]=$(sbatch --parsable \
        --dependency=afterok:"${train_jid}" \
        --export=ALL,RUN_DIR="${out_dir}" \
        "${SLURM_DIR}/run_baselines.sbatch")
    echo "Chaining compare/report (after ${BASELINE_JOB[$setup]}) for ${setup} ..."
    COMPARE_JOB[$setup]=$(sbatch --parsable \
        --dependency=afterok:"${BASELINE_JOB[$setup]}" \
        --export=ALL,RUN_DIR="${out_dir}" \
        "${SLURM_DIR}/compare.sbatch")
done

echo
echo "Submitted job ids:"
for setup in "${!TRAIN_JOB[@]}"; do
    printf "  %-18s train=%s baselines=%s compare=%s\n" \
        "$setup" "${TRAIN_JOB[$setup]}" "${BASELINE_JOB[$setup]}" "${COMPARE_JOB[$setup]}"
done
