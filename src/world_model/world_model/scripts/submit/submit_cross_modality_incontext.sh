#!/usr/bin/env bash
# =============================================================================
# submit_cross_modality_incontext.sh
#
# AnnData-first cross-modality in-context workflow:
#   support/context action embedding: ESM-2 650M -> obsm[X_pert_esm2_650M]
#   held-out query action embedding:  SubCell    -> obsm[X_pert_subcell_mae_rybg]
#
# The final training input is one .h5ad per dataset containing:
#   - state embeddings in obsm[X_stack]
#   - embpy payloads in uns["embpy"]["perturbations"]
#   - per-cell action matrices in obsm for both modalities
#
# Examples:
#   DRYRUN=1 DATASET=nadig bash src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh
#   DATASET=both STACK_CHECKPOINT=/path/bc_large.ckpt STACK_GENELIST=/path/basecount.pkl \
#     bash src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh
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

SELF="src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
TRAIN_LAUNCHER="${SLURM_DIR}/train_embedding.sbatch"

DATASET="${DATASET:-both}"
STATE_OBSM_KEY="${STATE_OBSM_KEY:-X_stack}"
SUPPORT_OBSM_KEY="${SUPPORT_OBSM_KEY:-X_pert_esm2_650M}"
QUERY_OBSM_KEY="${QUERY_OBSM_KEY:-X_pert_subcell_mae_rybg}"
PERTURBATION_KEY="${PERTURBATION_KEY:-perturbation}"

STACK_CHECKPOINT="${STACK_CHECKPOINT:-}"
STACK_GENELIST="${STACK_GENELIST:-}"
STACK_GENE_NAME_COL="${STACK_GENE_NAME_COL:-}"
STACK_BATCH_SIZE="${STACK_BATCH_SIZE:-64}"
STACK_CACHE_DIR="${STACK_CACHE_DIR:-runs/_cache/state_backbone}"

MORPHOLOGY_DATASET="${MORPHOLOGY_DATASET:-hpa}"
MORPHOLOGY_SOURCE="${MORPHOLOGY_SOURCE:-subcell}"
MORPHOLOGY_LOCAL_DIR="${MORPHOLOGY_LOCAL_DIR:-data/embeddings/morphology_cache}"
MORPHOLOGY_MAX_IMAGES="${MORPHOLOGY_MAX_IMAGES:-5}"
MORPHOLOGY_AGGREGATION="${MORPHOLOGY_AGGREGATION:-mean}"
MORPHOLOGY_WORKERS="${MORPHOLOGY_WORKERS:-8}"
MORPHOLOGY_PLATE_TYPE="${MORPHOLOGY_PLATE_TYPE:-}"
JUMP_PROFILES_DIR="${JUMP_PROFILES_DIR:-}"

PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
STACK_GPU_CONSTRAINT="${STACK_GPU_CONSTRAINT:-}"
ACTION_GPU_CONSTRAINT="${ACTION_GPU_CONSTRAINT:-a100_80gb|h100_80gb}"
TRAIN_GPU_CONSTRAINT="${TRAIN_GPU_CONSTRAINT:-h100_80gb}"
TRAIN_GRES="${TRAIN_GRES:-gpu:h100:1}"
TRAIN_MEM="${TRAIN_MEM:-128G}"
TRAIN_CPUS="${TRAIN_CPUS:-8}"

STACK_TIME="${STACK_TIME:-24:00:00}"
STACK_MEM="${STACK_MEM:-128G}"
STACK_CPUS="${STACK_CPUS:-8}"
ESM_TIME="${ESM_TIME:-08:00:00}"
ESM_MEM="${ESM_MEM:-64G}"
ESM_CPUS="${ESM_CPUS:-8}"
SUBCELL_TIME="${SUBCELL_TIME:-24:00:00}"
SUBCELL_MEM="${SUBCELL_MEM:-96G}"
SUBCELL_CPUS="${SUBCELL_CPUS:-8}"

STATE_PIXI_ENV="${STATE_PIXI_ENV:-arc-gpu}"
ESM_PIXI_ENV="${ESM_PIXI_ENV:-gpu}"
SUBCELL_PIXI_ENV="${SUBCELL_PIXI_ENV:-gpu}"
TRAIN_PIXI_ENV="${TRAIN_PIXI_ENV:-gpu}"
MATRIX_PIXI_ENV="${MATRIX_PIXI_ENV:-gpu}"

DRYRUN="${DRYRUN:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
REUSE_STATE="${REUSE_STATE:-1}"
REUSE_ESM="${REUSE_ESM:-1}"
REUSE_SUBCELL="${REUSE_SUBCELL:-1}"
FAIL_ON_UNRESOLVED="${FAIL_ON_UNRESOLVED:-0}"
STRICT_RESOLVER="${STRICT_RESOLVER:-0}"

STATE_H5AD_DIR="${STATE_H5AD_DIR:-runs/_cache/state_h5ad}"
STATE_NPZ_DIR="${STATE_NPZ_DIR:-runs/_cache/state_embeddings}"
CROSSMOD_H5AD_DIR="${CROSSMOD_H5AD_DIR:-runs/_cache/cross_modality_incontext_h5ad}"
TMP_ROOT="${TMP_ROOT:-runs/_tmp}"
LOG_ROOT="${LOG_ROOT:-logs}"
SUBMIT_STAMP="${SUBMIT_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
SUBMITTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
STACK_CACHE_DIR="$(project_path "$STACK_CACHE_DIR")"
MORPHOLOGY_LOCAL_DIR="$(project_path "$MORPHOLOGY_LOCAL_DIR")"
if [[ -n "$JUMP_PROFILES_DIR" ]]; then
    JUMP_PROFILES_DIR="$(project_path "$JUMP_PROFILES_DIR")"
fi
STATE_H5AD_DIR="$(project_path "$STATE_H5AD_DIR")"
STATE_NPZ_DIR="$(project_path "$STATE_NPZ_DIR")"
CROSSMOD_H5AD_DIR="$(project_path "$CROSSMOD_H5AD_DIR")"
LOG_ROOT="$(project_path "$LOG_ROOT")"
TMP_BASE="$(project_path "$TMP_ROOT")"
export TMPDIR="${TMPDIR:-${TMP_BASE}/submit-${SUBMIT_STAMP}}"
CELL_EMBEDDING_LABEL="${CELL_EMBEDDING_LABEL:-${STATE_OBSM_KEY#X_}}"
ACTION_LABEL="${ACTION_LABEL:-esm2_650M_to_subcell_mae_rybg}"
WORKFLOW_LABEL="${WORKFLOW_LABEL:-${CELL_EMBEDDING_LABEL}_with_${ACTION_LABEL}}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-runs/World_Model}"
LOG_BASE="${LOG_BASE:-${LOG_ROOT}/World_Model}"
RUN_ROOT_BASE="$(project_path "$RUN_ROOT_BASE")"
LOG_BASE="$(project_path "$LOG_BASE")"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${LOG_BASE}/submissions/${WORKFLOW_LABEL}/${SUBMIT_STAMP}}"
SUBMIT_LOG="${SUBMIT_LOG:-${SUBMIT_LOG_DIR}/submit_cross_modality_incontext_${SUBMIT_STAMP}.log}"
SUBMIT_LOG_DIR="$(project_path "$SUBMIT_LOG_DIR")"
SUBMIT_LOG="$(project_path "$SUBMIT_LOG")"

mkdir -p "$SUBMIT_LOG_DIR" "$STATE_H5AD_DIR" "$STATE_NPZ_DIR" "$CROSSMOD_H5AD_DIR" "$TMP_BASE" "$TMPDIR"

if [[ "${TEE_SUBMIT_LOG:-1}" == "1" ]]; then
    exec > >(tee -a "$SUBMIT_LOG") 2>&1
fi

matrix_value() {
    pixi run -e "$MATRIX_PIXI_ENV" -- python -m world_model.configs.run_matrix "$@"
}

h5ad_for() {
    matrix_value h5ad "$1"
}

base_cfg_for() {
    matrix_value base-config "$1"
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

job_stdout() {
    echo "${1}/jobs/${3}/stdout"
}

job_stderr() {
    echo "${1}/jobs/${3}/stderr"
}

link_job_logs() {
    local root="$1"
    local name="$2"
    local jid="$3"
    [[ "$DRYRUN" == "1" ]] && return 0
    local job_dir="${root}/jobs/${jid}"
    mkdir -p "$job_dir"
    ln -sfn "../${name}_${jid}.out" "${job_dir}/stdout"
    ln -sfn "../${name}_${jid}.err" "${job_dir}/stderr"
}

manifest_add() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$SUBMITTED_AT" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >>"$MANIFEST"
}

case "${1:-}" in
    help|--help|-h)
        sed -n '1,45p' "$SELF"
        exit 0
        ;;
esac

case "$DATASET" in
    nadig|replogle) DATASETS=("$DATASET") ;;
    both)           DATASETS=(nadig replogle) ;;
    *) echo "ERROR: DATASET must be nadig|replogle|both, got '$DATASET'." >&2; exit 2 ;;
esac

fail_args=""
if [[ "$FAIL_ON_UNRESOLVED" == "1" ]]; then
    fail_args="${fail_args} --fail-on-unresolved"
fi
if [[ "$STRICT_RESOLVER" == "1" ]]; then
    fail_args="${fail_args} --strict-resolver"
fi

echo "Cross-modality in-context world-model submission"
echo "  project:        $PROJECT_DIR"
echo "  datasets:       ${DATASETS[*]}"
echo "  state:          obsm[$STATE_OBSM_KEY]"
echo "  support action: obsm[$SUPPORT_OBSM_KEY] (esm2_650M)"
echo "  query action:   obsm[$QUERY_OBSM_KEY] (subcell_mae_rybg; ${MORPHOLOGY_DATASET}/${MORPHOLOGY_SOURCE})"
echo "  final h5ads:    $CROSSMOD_H5AD_DIR"
echo "  workflow:       $WORKFLOW_LABEL"
echo "  run root:       ${RUN_ROOT_BASE}/${WORKFLOW_LABEL}/<dataset>/${SUBMIT_STAMP}"
echo "  logs:           ${LOG_BASE}/${WORKFLOW_LABEL}/<dataset>/${SUBMIT_STAMP}"
echo "  submit log:     $SUBMIT_LOG"
echo "  dry run:        $DRYRUN"
echo

MANIFEST_OVERRIDE="${MANIFEST:-}"
SUBMISSION_INFO_OVERRIDE="${SUBMISSION_INFO:-}"

for ds in "${DATASETS[@]}"; do
    dataset_run_root="${RUN_ROOT:-${RUN_ROOT_BASE}/${WORKFLOW_LABEL}/${ds}/${SUBMIT_STAMP}}"
    dataset_log_dir="${LOG_DIR:-${LOG_BASE}/${WORKFLOW_LABEL}/${ds}/${SUBMIT_STAMP}}"
    dataset_run_root="$(project_path "$dataset_run_root")"
    dataset_log_dir="$(project_path "$dataset_log_dir")"
    MANIFEST="${MANIFEST_OVERRIDE:-${dataset_log_dir}/manifest.tsv}"
    SUBMISSION_INFO="${SUBMISSION_INFO_OVERRIDE:-${dataset_log_dir}/submission.txt}"
    mkdir -p "$dataset_run_root" "$dataset_log_dir"
    cat >"$SUBMISSION_INFO" <<EOF
submitted_at=${SUBMITTED_AT}
submit_stamp=${SUBMIT_STAMP}
project_dir=${PROJECT_DIR}
script=${SELF}
dataset=${ds}
action_label=${ACTION_LABEL}
cell_embedding_label=${CELL_EMBEDDING_LABEL}
workflow_label=${WORKFLOW_LABEL}
state_obsm_key=${STATE_OBSM_KEY}
support_obsm_key=${SUPPORT_OBSM_KEY}
query_obsm_key=${QUERY_OBSM_KEY}
run_root=${dataset_run_root}
log_dir=${dataset_log_dir}
manifest=${MANIFEST}
dryrun=${DRYRUN}
EOF
    printf "submitted_at\tdataset\tphase\tembedding\tjob_id\tdependency\toutput_path\tstdout\tstderr\n" >"$MANIFEST"

    source_h5ad="$(project_path "$(h5ad_for "$ds")")"
    state_npz="${STATE_NPZ_DIR}/${ds}_stack.npz"
    state_h5ad="${STATE_H5AD_DIR}/${ds}_stack.h5ad"
    esm_h5ad="${CROSSMOD_H5AD_DIR}/${ds}_stack_esm2_650M.h5ad"
    ready_h5ad="${CROSSMOD_H5AD_DIR}/${ds}_stack_esm2_650M_subcell_mae_rybg.h5ad"

    stack_jid=""
    if [[ "$REUSE_STATE" == "1" && -s "$state_h5ad" ]]; then
        echo "[$ds] reusing state AnnData: $state_h5ad"
        manifest_add "$ds" "state" "stack" "REUSED" "none" "$state_h5ad" "" ""
    else
        if [[ -z "$STACK_CHECKPOINT" || -z "$STACK_GENELIST" ]]; then
            if [[ "$DRYRUN" == "1" ]]; then
                STACK_CHECKPOINT="${STACK_CHECKPOINT:-/path/to/bc_large.ckpt}"
                STACK_GENELIST="${STACK_GENELIST:-/path/to/basecount_1000per_15000max.pkl}"
            else
                echo "ERROR: $state_h5ad is missing and STACK_CHECKPOINT/STACK_GENELIST were not provided." >&2
                exit 2
            fi
        fi
        if [[ "$DRYRUN" != "1" ]]; then
            STACK_CHECKPOINT="$(project_path "$STACK_CHECKPOINT")"
            STACK_GENELIST="$(project_path "$STACK_GENELIST")"
            [[ -f "$STACK_CHECKPOINT" ]] || { echo "ERROR: STACK_CHECKPOINT not found: $STACK_CHECKPOINT" >&2; exit 1; }
            [[ -f "$STACK_GENELIST" ]] || { echo "ERROR: STACK_GENELIST not found: $STACK_GENELIST" >&2; exit 1; }
        fi
        stack_gene_col_arg=""
        if [[ -n "$STACK_GENE_NAME_COL" ]]; then
            stack_gene_col_arg=" --stack-gene-name-col ${STACK_GENE_NAME_COL}"
        fi
        stack_cmd="set -euo pipefail; cd ${PROJECT_DIR}; export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export PYTHONNOUSERSITE=1; export TMPDIR=\"${TMP_BASE}/stack-\${SLURM_JOB_ID:-manual}\"; mkdir -p \"\$TMPDIR\"; trap 'rm -rf \"\$TMPDIR\"' EXIT; pixi run -e ${STATE_PIXI_ENV} -- python -m world_model.scripts.encode_cells --kind stack --adata ${source_h5ad} --output ${state_npz} --output-h5ad ${state_h5ad} --obsm-key ${STATE_OBSM_KEY} --device cuda --batch-size ${STACK_BATCH_SIZE} --cache-dir ${STACK_CACHE_DIR} --stack-checkpoint ${STACK_CHECKPOINT} --stack-genelist ${STACK_GENELIST}${stack_gene_col_arg}"
        stack_args=(--job-name="wm-xmod-stack-${ds}" --partition="$PARTITION" --qos="$QOS" --gres=gpu:1 --export=ALL)
        if [[ -n "$STACK_GPU_CONSTRAINT" ]]; then stack_args+=(--constraint="$STACK_GPU_CONSTRAINT"); fi
        stack_args+=(--time="$STACK_TIME" --mem="$STACK_MEM" --cpus-per-task="$STACK_CPUS" -o "${dataset_log_dir}/%x_%j.out" -e "${dataset_log_dir}/%x_%j.err" --wrap="$stack_cmd")
        stack_jid=$(submit_sbatch "${stack_args[@]}")
        echo "[$ds] stack job: $stack_jid -> $state_h5ad"
        link_job_logs "$dataset_log_dir" "wm-xmod-stack-${ds}" "$stack_jid"
        manifest_add "$ds" "state" "stack" "$stack_jid" "none" "$state_h5ad" \
            "$(job_stdout "$dataset_log_dir" "wm-xmod-stack-${ds}" "$stack_jid")" "$(job_stderr "$dataset_log_dir" "wm-xmod-stack-${ds}" "$stack_jid")"
    fi

    esm_jid=""
    if [[ "$REUSE_ESM" == "1" && -s "$esm_h5ad" ]]; then
        echo "[$ds] reusing ESM support-action AnnData: $esm_h5ad"
        manifest_add "$ds" "support_action" "esm2_650M" "REUSED" "none" "$esm_h5ad" "" ""
    else
        esm_cmd="set -euo pipefail; cd ${PROJECT_DIR}; export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export PYTHONNOUSERSITE=1; export TMPDIR=\"${TMP_BASE}/esm-\${SLURM_JOB_ID:-manual}\"; mkdir -p \"\$TMPDIR\"; trap 'rm -rf \"\$TMPDIR\"' EXIT; pixi run -e ${ESM_PIXI_ENV} -- python -m world_model.scripts.embed_perturbations --dataset ${ds} --h5ad ${state_h5ad} --model esm2_650M --entity-type gene --output-h5ad ${esm_h5ad} --obsm-key ${SUPPORT_OBSM_KEY} --perturbation-key ${PERTURBATION_KEY} --device cuda --region full --pooling-strategy mean --organism human --id-type symbol${fail_args}"
        esm_args=(--job-name="wm-xmod-esm-${ds}" --partition="$PARTITION" --qos="$QOS" --gres=gpu:1 --export=ALL)
        if [[ -n "$ACTION_GPU_CONSTRAINT" ]]; then esm_args+=(--constraint="$ACTION_GPU_CONSTRAINT"); fi
        esm_args+=(--time="$ESM_TIME" --mem="$ESM_MEM" --cpus-per-task="$ESM_CPUS" -o "${dataset_log_dir}/%x_%j.out" -e "${dataset_log_dir}/%x_%j.err" --wrap="$esm_cmd")
        if [[ -n "$stack_jid" ]]; then esm_args=(--dependency=afterok:"$stack_jid" "${esm_args[@]}"); fi
        esm_jid=$(submit_sbatch "${esm_args[@]}")
        echo "[$ds] ESM support-action job: $esm_jid -> $esm_h5ad"
        link_job_logs "$dataset_log_dir" "wm-xmod-esm-${ds}" "$esm_jid"
        manifest_add "$ds" "support_action" "esm2_650M" "$esm_jid" "${stack_jid:+afterok:${stack_jid}}" "$esm_h5ad" \
            "$(job_stdout "$dataset_log_dir" "wm-xmod-esm-${ds}" "$esm_jid")" "$(job_stderr "$dataset_log_dir" "wm-xmod-esm-${ds}" "$esm_jid")"
    fi

    subcell_jid=""
    if [[ "$REUSE_SUBCELL" == "1" && -s "$ready_h5ad" ]]; then
        echo "[$ds] reusing final cross-modality AnnData: $ready_h5ad"
        manifest_add "$ds" "query_action" "subcell_mae_rybg" "REUSED" "none" "$ready_h5ad" "" ""
    else
        morph_extra=""
        if [[ -n "$MORPHOLOGY_PLATE_TYPE" ]]; then morph_extra="${morph_extra} --plate-type ${MORPHOLOGY_PLATE_TYPE}"; fi
        if [[ -n "$JUMP_PROFILES_DIR" ]]; then morph_extra="${morph_extra} --jump-profiles-dir ${JUMP_PROFILES_DIR}"; fi
        subcell_cmd="set -euo pipefail; cd ${PROJECT_DIR}; export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export PYTHONNOUSERSITE=1; export TMPDIR=\"${TMP_BASE}/subcell-\${SLURM_JOB_ID:-manual}\"; mkdir -p \"\$TMPDIR\"; trap 'rm -rf \"\$TMPDIR\"' EXIT; pixi run -e ${SUBCELL_PIXI_ENV} -- python -m world_model.scripts.embed_perturbations --dataset ${ds} --h5ad ${esm_h5ad} --model subcell_mae_rybg --entity-type perturbation --output-h5ad ${ready_h5ad} --obsm-key ${QUERY_OBSM_KEY} --perturbation-key ${PERTURBATION_KEY} --device cuda --pooling-strategy attention_pool --morphology-dataset ${MORPHOLOGY_DATASET} --morphology-source ${MORPHOLOGY_SOURCE} --morphology-local-dir ${MORPHOLOGY_LOCAL_DIR}/${ds} --max-images ${MORPHOLOGY_MAX_IMAGES} --aggregation ${MORPHOLOGY_AGGREGATION} --morphology-workers ${MORPHOLOGY_WORKERS}${morph_extra}${fail_args}"
        subcell_args=(--job-name="wm-xmod-subcell-${ds}" --partition="$PARTITION" --qos="$QOS" --gres=gpu:1 --export=ALL)
        if [[ -n "$ACTION_GPU_CONSTRAINT" ]]; then subcell_args+=(--constraint="$ACTION_GPU_CONSTRAINT"); fi
        subcell_args+=(--time="$SUBCELL_TIME" --mem="$SUBCELL_MEM" --cpus-per-task="$SUBCELL_CPUS" -o "${dataset_log_dir}/%x_%j.out" -e "${dataset_log_dir}/%x_%j.err" --wrap="$subcell_cmd")
        if [[ -n "$esm_jid" ]]; then subcell_args=(--dependency=afterok:"$esm_jid" "${subcell_args[@]}"); fi
        subcell_jid=$(submit_sbatch "${subcell_args[@]}")
        echo "[$ds] SubCell query-action job: $subcell_jid -> $ready_h5ad"
        link_job_logs "$dataset_log_dir" "wm-xmod-subcell-${ds}" "$subcell_jid"
        manifest_add "$ds" "query_action" "subcell_mae_rybg" "$subcell_jid" "${esm_jid:+afterok:${esm_jid}}" "$ready_h5ad" \
            "$(job_stdout "$dataset_log_dir" "wm-xmod-subcell-${ds}" "$subcell_jid")" "$(job_stderr "$dataset_log_dir" "wm-xmod-subcell-${ds}" "$subcell_jid")"
    fi

    if [[ "$RUN_TRAIN" == "1" ]]; then
        cfg="$(project_path "$(base_cfg_for "$ds")")"
        run_name="crossmod_incontext_${ds}_stack_esm2_650M_to_subcell_mae_rybg"
        out_dir="${dataset_run_root}/${run_name}"
        train_args=(--job-name="wm-xmod-train-${ds}" --partition="$PARTITION" --qos="$QOS" --gres="$TRAIN_GRES" --mem="$TRAIN_MEM" --cpus-per-task="$TRAIN_CPUS" -o "${dataset_log_dir}/%x_%j.out" -e "${dataset_log_dir}/%x_%j.err" --export=ALL,EMBPY_PIXI_ENV="${TRAIN_PIXI_ENV}")
        if [[ -n "$TRAIN_GPU_CONSTRAINT" ]]; then train_args+=(--constraint="$TRAIN_GPU_CONSTRAINT"); fi
        if [[ -n "$subcell_jid" ]]; then train_args=(--dependency=afterok:"$subcell_jid" "${train_args[@]}"); fi
        train_args+=("$TRAIN_LAUNCHER" "$cfg"
            "data.h5ad_path=${ready_h5ad}"
            "data.state_obsm_key=${STATE_OBSM_KEY}"
            "data.context_mode=incontext_set"
            "dynamics.kind=incontext_set"
            "state_backbone.kind=stack"
            "action_embedding.source=anndata_obsm"
            "action_embedding.obsm_key=${SUPPORT_OBSM_KEY}"
            "action_embedding.model_name=esm2_650M"
            "query_action_embedding.source=anndata_obsm"
            "query_action_embedding.obsm_key=${QUERY_OBSM_KEY}"
            "query_action_embedding.model_name=subcell_mae_rybg"
            "run_name=${run_name}"
            "output_dir=${out_dir}")
        train_jid=$(submit_sbatch "${train_args[@]}")
        echo "[$ds] training job: $train_jid -> $out_dir"
        link_job_logs "$dataset_log_dir" "wm-xmod-train-${ds}" "$train_jid"
        manifest_add "$ds" "train" "$ACTION_LABEL" "$train_jid" "${subcell_jid:+afterok:${subcell_jid}}" "$out_dir" \
            "$(job_stdout "$dataset_log_dir" "wm-xmod-train-${ds}" "$train_jid")" "$(job_stderr "$dataset_log_dir" "wm-xmod-train-${ds}" "$train_jid")"
    fi
    echo "[$ds] final training h5ad: $ready_h5ad"
    echo
done

if command -v squeue >/dev/null 2>&1 && [[ "$DRYRUN" != "1" ]]; then
    squeue --me --format="%.18i %.9P %.32j %.10T %.10M %.6D %R" || true
fi
