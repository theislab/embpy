#!/usr/bin/env bash
# =============================================================================
# submit_incontext_stability_experiments.sh
#
# Submit the in-context stability experiment matrix:
#   baseline
#   latent_norm
#   residual_delta
#   latent_norm_residual
#   action_sim_support
#   aux_A / aux_B / aux_C / aux_D
#
# The script expects an AnnData that already contains:
#   - state embeddings in obsm[$STATE_OBSM_KEY], usually X_stack
#   - action embeddings in obsm[$ACTION_OBSM_KEY], e.g. X_pert_borzoi_v0
#
# Examples:
#   DRYRUN=1 DATASET=nadig EMB=borzoi_v0 STATE_OBSM_KEY=X_stack \
#     bash src/world_model/world_model/scripts/submit/submit_incontext_stability_experiments.sh
#
#   DATASET=nadig EMB=borzoi_v0 STATE_OBSM_KEY=X_stack \
#     bash src/world_model/world_model/scripts/submit/submit_incontext_stability_experiments.sh
#
# Ready AnnData resolution order:
#   1. H5AD_PATH, if set
#   2. H5AD_TEMPLATE with "{dataset}" inside, if set
#   3. runs/_cache/action_h5ad/<dataset>_<EMB>.h5ad
#   4. latest stable ready AnnData under
#      runs/_cache/world_model_ready_h5ad/stable_gene_embeddings/
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
cd "$PROJECT_DIR"
PROJECT_DIR="${PROJECT_DIR%/}"
export PROJECT_DIR
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SLURM_EXPORT_ENV

project_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        printf "%s\n" "$PROJECT_DIR"
    elif [[ "$path" = /* ]]; then
        printf "%s\n" "$path"
    else
        printf "%s/%s\n" "$PROJECT_DIR" "$path"
    fi
}

matrix_value() {
    pixi run -e "$MATRIX_PIXI_ENV" -- python -m world_model.configs.run_matrix "$@"
}

base_cfg_for() {
    matrix_value base-config "$1"
}

h5ad_for() {
    local ds="$1"
    if [[ -n "${H5AD_PATH:-}" ]]; then
        printf "%s\n" "$(project_path "$H5AD_PATH")"
    elif [[ -n "${H5AD_TEMPLATE:-}" ]]; then
        printf "%s\n" "$(project_path "${H5AD_TEMPLATE/\{dataset\}/$ds}")"
    else
        autodetect_h5ad_for "$ds"
    fi
}

autodetect_h5ad_for() {
    local ds="$1"
    local candidate
    candidate="$(project_path "runs/_cache/action_h5ad/${ds}_${EMB}.h5ad")"
    if [[ -s "$candidate" ]]; then
        printf "%s\n" "$candidate"
        return 0
    fi

    local latest=""
    local p
    for p in "$PROJECT_DIR"/runs/_cache/world_model_ready_h5ad/stable_gene_embeddings/*/"${ds}_stack_all_gene_embeddings.h5ad"; do
        [[ -s "$p" ]] || continue
        latest="$p"
    done
    if [[ -n "$latest" ]]; then
        printf "%s\n" "$latest"
        return 0
    fi

    for candidate in \
        "$(project_path "runs/_cache/world_model_ready_h5ad/stable_gene_embeddings/${ds}_stack_all_gene_embeddings.h5ad")" \
        "$(project_path "runs/_cache/world_model_ready_h5ad/${ds}_stack_all_gene_embeddings.h5ad")" \
        "$(project_path "runs/_cache/world_model_ready_h5ad/${ds}_stack_existing_gene_embeddings.h5ad")"
    do
        if [[ -s "$candidate" ]]; then
            printf "%s\n" "$candidate"
            return 0
        fi
    done

    printf "%s\n" "$(project_path "runs/_cache/action_h5ad/${ds}_${EMB}.h5ad")"
}

submit_sbatch() {
    if [[ "$DRYRUN" == "1" ]]; then
        printf '+ sbatch --chdir %q' "$PROJECT_DIR" >&2
        local arg
        for arg in "$@"; do
            printf ' %q' "$arg" >&2
        done
        printf '\n\n' >&2
        echo "DRYRUN-$RANDOM"
    else
        sbatch --parsable --chdir="$PROJECT_DIR" "$@"
    fi
}

link_job_logs() {
    local root="$1"
    local name="$2"
    local jid="$3"
    [[ "$DRYRUN" == "1" ]] && return 0
    local job_dir="${root}/jobs/${jid}"
    mkdir -p "$job_dir"
    ln -sfn "../../${name}_${jid}.out" "${job_dir}/stdout"
    ln -sfn "../../${name}_${jid}.err" "${job_dir}/stderr"
}

variant_overrides() {
    case "$1" in
        baseline)
            echo "dynamics.latent_normalization=none dynamics.prediction_mode=absolute data.incontext_support_strategy=random loss.info_nce=0.05 loss.action_counterfactual=0.05"
            ;;
        latent_norm)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=absolute data.incontext_support_strategy=random loss.info_nce=0.05 loss.action_counterfactual=0.05"
            ;;
        residual_delta)
            echo "dynamics.latent_normalization=none dynamics.prediction_mode=residual_delta data.incontext_support_strategy=random loss.info_nce=0.05 loss.action_counterfactual=0.05"
            ;;
        latent_norm_residual)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=residual_delta data.incontext_support_strategy=random loss.info_nce=0.05 loss.action_counterfactual=0.05"
            ;;
        action_sim_support)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=residual_delta data.incontext_support_strategy=action_similarity loss.info_nce=0.05 loss.action_counterfactual=0.05"
            ;;
        aux_A)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=residual_delta data.incontext_support_strategy=random loss.info_nce=0.00 loss.action_counterfactual=0.00"
            ;;
        aux_B)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=residual_delta data.incontext_support_strategy=random loss.info_nce=0.05 loss.action_counterfactual=0.00"
            ;;
        aux_C)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=residual_delta data.incontext_support_strategy=random loss.info_nce=0.00 loss.action_counterfactual=0.05"
            ;;
        aux_D)
            echo "dynamics.latent_normalization=layer_norm dynamics.prediction_mode=residual_delta data.incontext_support_strategy=random loss.info_nce=0.05 loss.action_counterfactual=0.05"
            ;;
        *)
            echo "ERROR: unknown variant '$1'." >&2
            exit 2
            ;;
    esac
}

DATASET="${DATASET:-nadig}"
EMB="${EMB:-borzoi_v0}"
STATE_OBSM_KEY="${STATE_OBSM_KEY:-X_stack}"
ACTION_OBSM_KEY="${ACTION_OBSM_KEY:-X_pert_${EMB}}"
QUERY_ACTION_OBSM_KEY="${QUERY_ACTION_OBSM_KEY:-}"
PERTURBATION_KEY="${PERTURBATION_KEY:-perturbation}"
MATRIX_PIXI_ENV="${MATRIX_PIXI_ENV:-gpu}"
TRAIN_PIXI_ENV="${TRAIN_PIXI_ENV:-gpu}"

PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
TRAIN_GPU_CONSTRAINT="${TRAIN_GPU_CONSTRAINT:-h100_80gb}"
TRAIN_GRES="${TRAIN_GRES:-gpu:h100:1}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
TRAIN_MEM="${TRAIN_MEM:-128G}"
TRAIN_CPUS="${TRAIN_CPUS:-8}"

SUPPORT_SIZE="${SUPPORT_SIZE:-16}"
DYNAMICS_MAX_SEQUENCE_LENGTH="${DYNAMICS_MAX_SEQUENCE_LENGTH:-16}"
SEQUENCE_BUCKET_KEY="${SEQUENCE_BUCKET_KEY:-auto}"
STATE_BACKBONE_KIND="${STATE_BACKBONE_KIND:-stack}"
TRAIN_DEFAULT_OVERRIDES="${TRAIN_DEFAULT_OVERRIDES:-data.batch_size=16 data.num_workers=0 data.pin_memory=false eval.save_predictions=false eval.n_control_samples_per_pert=8 optim.lr=0.0001 optim.grad_clip=0.5 loss.info_nce_temperature=0.1}"
REQUIRE_CELL_EVAL="${REQUIRE_CELL_EVAL:-true}"
CELL_EVAL_PROFILE="${CELL_EVAL_PROFILE:-full}"
CELL_EVAL_NUM_THREADS="${CELL_EVAL_NUM_THREADS:-$TRAIN_CPUS}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-$TRAIN_DEFAULT_OVERRIDES}"
VARIANTS="${VARIANTS:-baseline latent_norm residual_delta latent_norm_residual action_sim_support aux_A aux_B aux_C aux_D}"

DRYRUN="${DRYRUN:-0}"
SUBMIT_STAMP="${SUBMIT_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
SUBMITTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
TMP_ROOT="${TMP_ROOT:-runs/_tmp}"
TMP_BASE="$(project_path "$TMP_ROOT")"
export TMPDIR="${TMPDIR:-${TMP_BASE}/stability-submit-${SUBMIT_STAMP}}"

WORKFLOW_LABEL="${WORKFLOW_LABEL:-stability_${STATE_OBSM_KEY#X_}_with_${EMB}}"
RUN_ROOT_BASE="$(project_path "${RUN_ROOT_BASE:-runs/World_Model}")"
LOG_ROOT="$(project_path "${LOG_ROOT:-logs}")"
LOG_BASE="${LOG_BASE:-${LOG_ROOT}/World_Model}"
LOG_BASE="$(project_path "$LOG_BASE")"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${LOG_BASE}/submissions/${WORKFLOW_LABEL}/${SUBMIT_STAMP}}"
SUBMIT_LOG_DIR="$(project_path "$SUBMIT_LOG_DIR")"
SUBMIT_LOG="${SUBMIT_LOG:-${SUBMIT_LOG_DIR}/submit_incontext_stability_${SUBMIT_STAMP}.log}"
SUBMIT_LOG="$(project_path "$SUBMIT_LOG")"

mkdir -p "$SUBMIT_LOG_DIR" "$TMP_BASE" "$TMPDIR"

if [[ "${TEE_SUBMIT_LOG:-1}" == "1" ]]; then
    exec > >(tee -a "$SUBMIT_LOG") 2>&1
fi

case "$DATASET" in
    nadig|replogle) DATASETS=("$DATASET") ;;
    both)           DATASETS=(nadig replogle) ;;
    *) echo "ERROR: DATASET must be nadig|replogle|both, got '$DATASET'." >&2; exit 2 ;;
esac

read -r -a VARIANT_LIST <<<"$VARIANTS"
read -r -a TRAIN_EXTRA_ARGS <<<"$TRAIN_OVERRIDES"

echo "In-context stability experiment submission"
echo "  project:      $PROJECT_DIR"
echo "  datasets:     ${DATASETS[*]}"
echo "  variants:     ${VARIANT_LIST[*]}"
echo "  state obsm:   $STATE_OBSM_KEY"
echo "  action obsm:  $ACTION_OBSM_KEY"
if [[ -n "$QUERY_ACTION_OBSM_KEY" ]]; then
    echo "  query action: $QUERY_ACTION_OBSM_KEY"
fi
echo "  run root:     ${RUN_ROOT_BASE}/${WORKFLOW_LABEL}/<dataset>/${SUBMIT_STAMP}"
echo "  logs:         ${LOG_BASE}/${WORKFLOW_LABEL}/<dataset>/${SUBMIT_STAMP}"
echo "  submit log:   $SUBMIT_LOG"
echo "  cell-eval:    require=${REQUIRE_CELL_EVAL} profile=${CELL_EVAL_PROFILE} threads=${CELL_EVAL_NUM_THREADS}"
echo "  dry run:      $DRYRUN"
echo

TRAIN_LAUNCHER="src/world_model/world_model/scripts/slurm/train_embedding.sbatch"

for ds in "${DATASETS[@]}"; do
    cfg="$(project_path "$(base_cfg_for "$ds")")"
    h5ad="$(h5ad_for "$ds")"
    echo "[$ds] ready h5ad: $h5ad"
    if [[ "$DRYRUN" != "1" && ! -s "$h5ad" ]]; then
        echo "ERROR: ready AnnData not found for dataset '$ds': $h5ad" >&2
        echo "Pass H5AD_PATH=/path/file.h5ad or H5AD_TEMPLATE='/path/{dataset}_file.h5ad'." >&2
        exit 1
    fi

    dataset_run_root="$(project_path "${RUN_ROOT:-${RUN_ROOT_BASE}/${WORKFLOW_LABEL}/${ds}/${SUBMIT_STAMP}}")"
    dataset_log_dir="$(project_path "${LOG_DIR:-${LOG_BASE}/${WORKFLOW_LABEL}/${ds}/${SUBMIT_STAMP}}")"
    manifest="${dataset_log_dir}/manifest.tsv"
    mkdir -p "$dataset_run_root" "$dataset_log_dir"
    printf "submitted_at\tdataset\tvariant\tjob_id\toutput_path\tstdout\tstderr\n" >"$manifest"

    for variant in "${VARIANT_LIST[@]}"; do
        run_name="incontext_${ds}_${EMB}_${variant}"
        out_dir="${dataset_run_root}/${run_name}"
        read -r -a VARIANT_EXTRA_ARGS <<<"$(variant_overrides "$variant")"

        train_args=(
            --job-name="wm-${ds}-${variant}"
            --partition="$PARTITION"
            --qos="$QOS"
            --gres="$TRAIN_GRES"
            --time="$TRAIN_TIME"
            --mem="$TRAIN_MEM"
            --cpus-per-task="$TRAIN_CPUS"
            -o "${dataset_log_dir}/%x_%j.out"
            -e "${dataset_log_dir}/%x_%j.err"
            --export=ALL,EMBPY_PIXI_ENV="${TRAIN_PIXI_ENV}"
        )
        if [[ -n "$TRAIN_GPU_CONSTRAINT" ]]; then
            train_args+=(--constraint="$TRAIN_GPU_CONSTRAINT")
        fi
        train_args+=(
            "$TRAIN_LAUNCHER"
            "$cfg"
            "data.h5ad_path=${h5ad}"
            "data.state_obsm_key=${STATE_OBSM_KEY}"
            "data.perturbation_key=${PERTURBATION_KEY}"
            "data.context_mode=incontext_set"
            "data.incontext_support_size=${SUPPORT_SIZE}"
            "data.sequence_bucket_key=${SEQUENCE_BUCKET_KEY}"
            "dynamics.kind=incontext_set"
            "dynamics.max_sequence_length=${DYNAMICS_MAX_SEQUENCE_LENGTH}"
            "state_backbone.kind=${STATE_BACKBONE_KIND}"
            "action_embedding.source=anndata_obsm"
            "action_embedding.obsm_key=${ACTION_OBSM_KEY}"
            "action_embedding.model_name=${EMB}"
            "run_name=${run_name}"
            "output_dir=${out_dir}"
            "eval.require_cell_eval=${REQUIRE_CELL_EVAL}"
            "eval.cell_eval_profile=${CELL_EVAL_PROFILE}"
            "eval.cell_eval_num_threads=${CELL_EVAL_NUM_THREADS}"
            "${TRAIN_EXTRA_ARGS[@]}"
            "${VARIANT_EXTRA_ARGS[@]}"
        )
        if [[ -n "$QUERY_ACTION_OBSM_KEY" ]]; then
            train_args+=(
                "query_action_embedding.source=anndata_obsm"
                "query_action_embedding.obsm_key=${QUERY_ACTION_OBSM_KEY}"
                "query_action_embedding.model_name=${QUERY_ACTION_MODEL_NAME:-$EMB}"
            )
        fi

        jid=$(submit_sbatch "${train_args[@]}")
        echo "[$ds] $variant job: $jid -> $out_dir"
        link_job_logs "$dataset_log_dir" "wm-${ds}-${variant}" "$jid"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$SUBMITTED_AT" "$ds" "$variant" "$jid" "$out_dir" \
            "${dataset_log_dir}/jobs/${jid}/stdout" \
            "${dataset_log_dir}/jobs/${jid}/stderr" >>"$manifest"
    done
    echo "[$ds] manifest: $manifest"
    echo
done

if command -v squeue >/dev/null 2>&1 && [[ "$DRYRUN" != "1" ]]; then
    squeue --me --format="%.18i %.9P %.32j %.10T %.10M %.6D %R" || true
fi
