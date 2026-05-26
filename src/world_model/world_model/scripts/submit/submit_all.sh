#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Submit the three training setups, the baseline run, and the final
# compare / report job with one command. Uses sbatch --dependency=afterok:
# to chain baselines + report after each training job finishes.
#
# Usage (default behaviour):
#   bash src/world_model/world_model/scripts/submit/submit_all.sh
#
# Override the SLURM partition / qos via env:
#   PARTITION=gpu_p QOS=gpu_normal bash .../submit_all.sh
# Override the transfer fraction:
#   FRACTION=0.05 bash .../submit_all.sh
# Run multiple seed replicates (default: one seed, 0):
#   SEEDS="0 1 2" bash .../submit_all.sh
# Skip a setup (default: run all three):
#   SKIP_NADIG=1 SKIP_REPLOGLE=1 SKIP_TRANSFER=1 bash .../submit_all.sh
#
# Action-encoder ablation sweep (alternative entry point):
#   ./submit_all.sh --ablate-action-encoder \
#       --base-config configs/experiments/single_replogle.yaml \
#       --grid configs/grids/ablation_action_encoder.yaml \
#       [--array]
#   Submits one pre-warm cache job per spec, then the ablation runner
#   (or array of N runner tasks), then the aggregator + report.
# -----------------------------------------------------------------------------
set -euo pipefail

SLURM_DIR="src/world_model/world_model/scripts/slurm"
mkdir -p logs


# ---------------------------------------------------------------------
# Action-encoder ablation branch
# ---------------------------------------------------------------------

ablate_action_encoder() {
    local base_config="src/world_model/world_model/configs/experiments/single_replogle.yaml"
    local grid="src/world_model/world_model/configs/grids/ablation_action_encoder.yaml"
    local output_root="runs/ablation_action_replogle"
    local use_array=0
    local only=""
    local skip=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-config)  base_config="$2"; shift 2 ;;
            --grid)         grid="$2"; shift 2 ;;
            --output-root)  output_root="$2"; shift 2 ;;
            --only)         only="$2"; shift 2 ;;
            --skip)         skip="$2"; shift 2 ;;
            --array)        use_array=1; shift ;;
            *) echo "Unknown ablation arg: $1" >&2; exit 2 ;;
        esac
    done

    echo "Action-encoder ablation"
    echo "  base-config = ${base_config}"
    echo "  grid        = ${grid}"
    echo "  output-root = ${output_root}"

    # Read grid keys + h5ad path with one Python call so we don't shell-parse YAML.
    mapfile -t spec_keys < <(pixi run -e gpu -- python -c "
import sys
from world_model.evaluation.ablation import load_grid
for s in load_grid(sys.argv[1]):
    print(s.key)
" "$grid")
    n_specs=${#spec_keys[@]}
    if [[ ${n_specs} -eq 0 ]]; then
        echo "Grid is empty; nothing to submit." >&2
        exit 2
    fi

    H5AD=$(pixi run -e gpu -- python -c "
from world_model.configs import load_yaml_config
print(load_yaml_config('${base_config}').data.h5ad_path)
")

    declare -a prewarm_jids=()
    for key in "${spec_keys[@]}"; do
        # Map grid key to the model_name via Python so this shell stays lean.
        model_name=$(pixi run -e gpu -- python -c "
import sys
from world_model.evaluation.ablation import load_grid
for s in load_grid(sys.argv[1]):
    if s.key == sys.argv[2]:
        print(s.model_name); break
" "$grid" "$key")
        echo "Submitting pre-warm for spec=${key} model=${model_name} ..."
        prewarm_jid=$(sbatch --parsable \
            --job-name="wm-prewarm-${key}" \
            --dependency=singleton \
            --export=ALL,MODEL="${model_name}",H5AD="${H5AD}",GRID_KEY="${key}" \
            -o logs/%x_%j.out -e logs/%x_%j.err \
            --wrap="set -euo pipefail; \
                cd ${PWD}; \
                export TMPDIR=\"${PWD}/.tmp/job-\$SLURM_JOB_ID\"; \
                mkdir -p \"\$TMPDIR\"; \
                trap 'rm -rf \"\$TMPDIR\"' EXIT; \
                pixi run -e gpu -- python -m world_model.scripts.embed_perturbations \
                    --dataset replogle --h5ad ${H5AD} --model ${model_name}")
        prewarm_jids+=("$prewarm_jid")
    done

    prewarm_dep=$(IFS=:; echo "${prewarm_jids[*]}")
    only_arg=""; skip_arg=""
    if [[ -n "${only}" ]]; then only_arg="ONLY=${only}"; fi
    if [[ -n "${skip}" ]]; then skip_arg="SKIP=${skip}"; fi

    if [[ ${use_array} -eq 1 ]]; then
        echo "Submitting ablation array (0-$((n_specs - 1))) after pre-warm ..."
        runner_jid=$(sbatch --parsable \
            --array=0-$((n_specs - 1)) \
            --dependency=afterok:${prewarm_dep} \
            --export=ALL,BASE_CONFIG="${base_config}",GRID="${grid}",OUTPUT_ROOT="${output_root}",${only_arg},${skip_arg} \
            "${SLURM_DIR}/ablate_action_encoder.sbatch")
    else
        echo "Submitting ablation runner (single job) after pre-warm ..."
        runner_jid=$(sbatch --parsable \
            --dependency=afterok:${prewarm_dep} \
            --export=ALL,BASE_CONFIG="${base_config}",GRID="${grid}",OUTPUT_ROOT="${output_root}",${only_arg},${skip_arg} \
            "${SLURM_DIR}/ablate_action_encoder.sbatch")
    fi

    echo "Submitting aggregator after runner ${runner_jid} ..."
    agg_jid=$(sbatch --parsable \
        --job-name=wm-ablate-aggregate \
        --dependency=afterok:"${runner_jid}" \
        --export=ALL,OUTPUT_ROOT="${output_root}",GRID="${grid}" \
        -o logs/%x_%j.out -e logs/%x_%j.err \
        --wrap="set -euo pipefail; \
            cd ${PWD}; \
            pixi run -e gpu -- python -m world_model.evaluation.ablation.aggregate \
                --output-root ${output_root} --grid ${grid}")

    echo "Submitting final make_report after aggregator ${agg_jid} ..."
    report_jid=$(sbatch --parsable \
        --job-name=wm-ablate-report \
        --dependency=afterok:"${agg_jid}" \
        --export=ALL,RUN_DIR="${output_root}" \
        "${SLURM_DIR}/compare.sbatch" || true)

    echo
    echo "Submitted ablation DAG:"
    printf "  pre-warm    %s\n" "${prewarm_jids[@]}"
    printf "  runner      %s\n" "${runner_jid}"
    printf "  aggregate   %s\n" "${agg_jid}"
    printf "  make_report %s\n" "${report_jid}"
    echo
    if command -v squeue >/dev/null 2>&1; then
        squeue --me --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" || true
    fi
}


# ---------------------------------------------------------------------
# Leave-one-encoder-out (LOEO) branch
# ---------------------------------------------------------------------

lone() {
    local base_config="src/world_model/world_model/configs/experiments/transfer.yaml"
    local grid="src/world_model/world_model/configs/grids/ablation_action_encoder.yaml"
    local output_root="runs/lone_replogle"
    local strategy="reset_adapter"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-config) base_config="$2"; shift 2 ;;
            --grid)        grid="$2"; shift 2 ;;
            --output-root) output_root="$2"; shift 2 ;;
            --strategy)    strategy="$2"; shift 2 ;;
            *) echo "Unknown lone arg: $1" >&2; exit 2 ;;
        esac
    done

    echo "Leave-one-encoder-out (strategy=${strategy})"
    echo "  base-config = ${base_config}"
    echo "  grid        = ${grid}"
    echo "  output-root = ${output_root}"

    mapfile -t spec_keys < <(pixi run -e gpu -- python -c "
import sys
from world_model.evaluation.ablation import load_grid
for s in load_grid(sys.argv[1]):
    print(s.key)
" "$grid")
    n_specs=${#spec_keys[@]}
    n_offdiag=$(( n_specs * (n_specs - 1) ))
    if [[ ${n_specs} -lt 2 ]]; then
        echo "Need at least 2 encoders in the grid; found ${n_specs}." >&2
        exit 2
    fi

    echo "Submitting Stage A array (0-$((n_specs - 1))) ..."
    a_jid=$(sbatch --parsable \
        --array=0-$((n_specs - 1)) \
        --export=ALL,STAGE=A,BASE_CONFIG="${base_config}",GRID="${grid}",OUTPUT_ROOT="${output_root}",STRATEGY="${strategy}" \
        "${SLURM_DIR}/leave_one_encoder_out.sbatch")

    if [[ ${n_offdiag} -gt 0 ]]; then
        echo "Submitting Stage B array (0-$((n_offdiag - 1))) after ${a_jid} ..."
        b_jid=$(sbatch --parsable \
            --array=0-$((n_offdiag - 1)) \
            --dependency=afterok:"${a_jid}" \
            --export=ALL,STAGE=B,BASE_CONFIG="${base_config}",GRID="${grid}",OUTPUT_ROOT="${output_root}",STRATEGY="${strategy}" \
            "${SLURM_DIR}/leave_one_encoder_out.sbatch")
    else
        b_jid=""
    fi

    echo "Submitting LOEO aggregator after Stage B ..."
    dep="${b_jid:-${a_jid}}"
    agg_jid=$(sbatch --parsable \
        --job-name=wm-lone-aggregate \
        --dependency=afterok:"${dep}" \
        --export=ALL,OUTPUT_ROOT="${output_root}",GRID="${grid}",STRATEGY="${strategy}" \
        -o logs/%x_%j.out -e logs/%x_%j.err \
        --wrap="set -euo pipefail; \
            cd ${PWD}; \
            pixi run -e gpu -- python -m world_model.scripts.sweeps.leave_one_encoder_out \
                --base-config ${base_config} --grid ${grid} \
                --strategy ${strategy} --output-root ${output_root} --dry-run; \
            pixi run -e gpu -- python -c \
                'from world_model.scripts.sweeps.leave_one_encoder_out import _summary_long_for_strategy, _render_strategy_heatmaps; \
                 from world_model.evaluation.ablation import resolve_grid; \
                 from pathlib import Path; \
                 grid = resolve_grid(grid_path=\"${grid}\"); \
                 long = _summary_long_for_strategy(Path(\"${output_root}\"), grid, \"${strategy}\"); \
                 long.to_csv(Path(\"${output_root}\")/\"${strategy}\"/\"summary_long.csv\", index=False); \
                 _render_strategy_heatmaps(long, grid, Path(\"${output_root}\")/\"${strategy}\"/\"heatmaps\")'")

    echo
    echo "Submitted LOEO DAG:"
    printf "  Stage A     %s\n" "${a_jid}"
    printf "  Stage B     %s\n" "${b_jid}"
    printf "  aggregate   %s\n" "${agg_jid}"
    echo
    if command -v squeue >/dev/null 2>&1; then
        squeue --me --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" || true
    fi
}


# Branch on --ablate-action-encoder / --lone before falling through to the
# default train-baselines-compare DAG.
if [[ "${1:-}" == "--ablate-action-encoder" ]]; then
    shift
    ablate_action_encoder "$@"
    exit 0
fi

if [[ "${1:-}" == "--lone" ]]; then
    shift
    lone "$@"
    exit 0
fi


# ---------------------------------------------------------------------
# Default branch: train + baselines + compare for the three setups
# ---------------------------------------------------------------------

# Map setup -> output_dir: hard-coded to match the values in
# configs/experiments/*.yaml so we do not have to parse YAML in bash.
declare -A SETUP_OUT
SETUP_OUT[single_nadig]="runs/world_model/single_nadig"
SETUP_OUT[single_replogle]="runs/world_model/single_replogle"

FRACTION="${FRACTION:-0.10}"
SEEDS="${SEEDS:-0}"
read -r -a SEED_LIST <<<"${SEEDS}"
P_INT=$(python -c "print(int(float('${FRACTION}')*100))")
TRANSFER_RUN_NAME="transfer_nadig_to_replogle_p$(printf '%03d' $P_INT)"
SETUP_OUT[transfer]="runs/world_model/${TRANSFER_RUN_NAME}"

declare -A TRAIN_JOB

seed_suffix() {
    local seed="$1"
    if [[ "${#SEED_LIST[@]}" -eq 1 && "$seed" == "0" ]]; then
        echo ""
    else
        echo "_seed${seed}"
    fi
}

for seed in "${SEED_LIST[@]}"; do
    suffix="$(seed_suffix "$seed")"
    if [[ "${SKIP_NADIG:-0}" != "1" ]]; then
        setup="single_nadig${suffix}"
        out_dir="${SETUP_OUT[single_nadig]}${suffix}"
        echo "Submitting ${setup} (seed=${seed}) ..."
        TRAIN_JOB[$setup]=$(sbatch --parsable "${SLURM_DIR}/train_single_nadig.sbatch" \
            "seed=${seed}" "split.seed=${seed}" \
            "run_name=${setup}" "output_dir=${out_dir}")
        SETUP_OUT[$setup]="$out_dir"
    fi

    if [[ "${SKIP_REPLOGLE:-0}" != "1" ]]; then
        setup="single_replogle${suffix}"
        out_dir="${SETUP_OUT[single_replogle]}${suffix}"
        echo "Submitting ${setup} (seed=${seed}) ..."
        TRAIN_JOB[$setup]=$(sbatch --parsable "${SLURM_DIR}/train_single_replogle.sbatch" \
            "seed=${seed}" "split.seed=${seed}" \
            "run_name=${setup}" "output_dir=${out_dir}")
        SETUP_OUT[$setup]="$out_dir"
    fi

    if [[ "${SKIP_TRANSFER:-0}" != "1" ]]; then
        setup="transfer${suffix}"
        run_name="${TRANSFER_RUN_NAME}${suffix}"
        out_dir="${SETUP_OUT[transfer]}${suffix}"
        echo "Submitting ${setup} (fraction=${FRACTION}, seed=${seed}) ..."
        TRAIN_JOB[$setup]=$(FRACTION="${FRACTION}" sbatch --parsable "${SLURM_DIR}/train_transfer.sbatch" \
            "seed=${seed}" "split.seed=${seed}" \
            "run_name=${run_name}" "output_dir=${out_dir}")
        SETUP_OUT[$setup]="$out_dir"
    fi
done

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
