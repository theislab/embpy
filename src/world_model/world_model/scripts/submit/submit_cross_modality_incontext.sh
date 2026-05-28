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
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

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
RUN_ROOT="${RUN_ROOT:-runs/world_model/cross_modality_incontext}"
LOG_ROOT="${LOG_ROOT:-logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/cross_modality_incontext}"

mkdir -p "$LOG_DIR" "$STATE_H5AD_DIR" "$STATE_NPZ_DIR" "$CROSSMOD_H5AD_DIR" "$TMP_ROOT" "$RUN_ROOT"

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
        printf '+ sbatch' >&2
        local arg
        for arg in "$@"; do
            printf ' %q' "$arg" >&2
        done
        printf '\n\n' >&2
        echo "DRYRUN-$RANDOM"
    else
        sbatch --parsable "$@"
    fi
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
echo "  run root:       $RUN_ROOT"
echo "  logs:           $LOG_DIR"
echo "  dry run:        $DRYRUN"
echo

for ds in "${DATASETS[@]}"; do
    source_h5ad="$(h5ad_for "$ds")"
    state_npz="${STATE_NPZ_DIR}/${ds}_stack.npz"
    state_h5ad="${STATE_H5AD_DIR}/${ds}_stack.h5ad"
    esm_h5ad="${CROSSMOD_H5AD_DIR}/${ds}_stack_esm2_650M.h5ad"
    ready_h5ad="${CROSSMOD_H5AD_DIR}/${ds}_stack_esm2_650M_subcell_mae_rybg.h5ad"

    stack_jid=""
    if [[ "$REUSE_STATE" == "1" && -s "$state_h5ad" ]]; then
        echo "[$ds] reusing state AnnData: $state_h5ad"
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
            [[ -f "$STACK_CHECKPOINT" ]] || { echo "ERROR: STACK_CHECKPOINT not found: $STACK_CHECKPOINT" >&2; exit 1; }
            [[ -f "$STACK_GENELIST" ]] || { echo "ERROR: STACK_GENELIST not found: $STACK_GENELIST" >&2; exit 1; }
        fi
        stack_gene_col_arg=""
        if [[ -n "$STACK_GENE_NAME_COL" ]]; then
            stack_gene_col_arg=" --stack-gene-name-col ${STACK_GENE_NAME_COL}"
        fi
        stack_cmd="set -euo pipefail; cd ${PROJECT_DIR}; export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export PYTHONNOUSERSITE=1; export TMPDIR=\"${PROJECT_DIR}/${TMP_ROOT}/stack-\${SLURM_JOB_ID:-manual}\"; mkdir -p \"\$TMPDIR\"; trap 'rm -rf \"\$TMPDIR\"' EXIT; pixi run -e ${STATE_PIXI_ENV} -- python -m world_model.scripts.encode_cells --kind stack --adata ${source_h5ad} --output ${state_npz} --output-h5ad ${state_h5ad} --obsm-key ${STATE_OBSM_KEY} --device cuda --batch-size ${STACK_BATCH_SIZE} --cache-dir ${STACK_CACHE_DIR} --stack-checkpoint ${STACK_CHECKPOINT} --stack-genelist ${STACK_GENELIST}${stack_gene_col_arg}"
        stack_args=(--job-name="wm-xmod-stack-${ds}" --partition="$PARTITION" --qos="$QOS" --gres=gpu:1)
        if [[ -n "$STACK_GPU_CONSTRAINT" ]]; then stack_args+=(--constraint="$STACK_GPU_CONSTRAINT"); fi
        stack_args+=(--time="$STACK_TIME" --mem="$STACK_MEM" --cpus-per-task="$STACK_CPUS" -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err" --wrap="$stack_cmd")
        stack_jid=$(submit_sbatch "${stack_args[@]}")
        echo "[$ds] stack job: $stack_jid -> $state_h5ad"
    fi

    esm_jid=""
    if [[ "$REUSE_ESM" == "1" && -s "$esm_h5ad" ]]; then
        echo "[$ds] reusing ESM support-action AnnData: $esm_h5ad"
    else
        esm_cmd="set -euo pipefail; cd ${PROJECT_DIR}; export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export PYTHONNOUSERSITE=1; pixi run -e ${ESM_PIXI_ENV} -- python -m world_model.scripts.embed_perturbations --dataset ${ds} --h5ad ${state_h5ad} --model esm2_650M --entity-type gene --output-h5ad ${esm_h5ad} --obsm-key ${SUPPORT_OBSM_KEY} --perturbation-key ${PERTURBATION_KEY} --device cuda --region full --pooling-strategy mean --organism human --id-type symbol${fail_args}"
        esm_args=(--job-name="wm-xmod-esm-${ds}" --partition="$PARTITION" --qos="$QOS" --gres=gpu:1)
        if [[ -n "$ACTION_GPU_CONSTRAINT" ]]; then esm_args+=(--constraint="$ACTION_GPU_CONSTRAINT"); fi
        esm_args+=(--time="$ESM_TIME" --mem="$ESM_MEM" --cpus-per-task="$ESM_CPUS" -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err" --wrap="$esm_cmd")
        if [[ -n "$stack_jid" ]]; then esm_args=(--dependency=afterok:"$stack_jid" "${esm_args[@]}"); fi
        esm_jid=$(submit_sbatch "${esm_args[@]}")
        echo "[$ds] ESM support-action job: $esm_jid -> $esm_h5ad"
    fi

    subcell_jid=""
    if [[ "$REUSE_SUBCELL" == "1" && -s "$ready_h5ad" ]]; then
        echo "[$ds] reusing final cross-modality AnnData: $ready_h5ad"
    else
        morph_extra=""
        if [[ -n "$MORPHOLOGY_PLATE_TYPE" ]]; then morph_extra="${morph_extra} --plate-type ${MORPHOLOGY_PLATE_TYPE}"; fi
        if [[ -n "$JUMP_PROFILES_DIR" ]]; then morph_extra="${morph_extra} --jump-profiles-dir ${JUMP_PROFILES_DIR}"; fi
        subcell_cmd="set -euo pipefail; cd ${PROJECT_DIR}; export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export PYTHONNOUSERSITE=1; export TMPDIR=\"${PROJECT_DIR}/${TMP_ROOT}/subcell-\${SLURM_JOB_ID:-manual}\"; mkdir -p \"\$TMPDIR\"; trap 'rm -rf \"\$TMPDIR\"' EXIT; pixi run -e ${SUBCELL_PIXI_ENV} -- python -m world_model.scripts.embed_perturbations --dataset ${ds} --h5ad ${esm_h5ad} --model subcell_mae_rybg --entity-type perturbation --output-h5ad ${ready_h5ad} --obsm-key ${QUERY_OBSM_KEY} --perturbation-key ${PERTURBATION_KEY} --device cuda --pooling-strategy attention_pool --morphology-dataset ${MORPHOLOGY_DATASET} --morphology-source ${MORPHOLOGY_SOURCE} --morphology-local-dir ${MORPHOLOGY_LOCAL_DIR}/${ds} --max-images ${MORPHOLOGY_MAX_IMAGES} --aggregation ${MORPHOLOGY_AGGREGATION} --morphology-workers ${MORPHOLOGY_WORKERS}${morph_extra}${fail_args}"
        subcell_args=(--job-name="wm-xmod-subcell-${ds}" --partition="$PARTITION" --qos="$QOS" --gres=gpu:1)
        if [[ -n "$ACTION_GPU_CONSTRAINT" ]]; then subcell_args+=(--constraint="$ACTION_GPU_CONSTRAINT"); fi
        subcell_args+=(--time="$SUBCELL_TIME" --mem="$SUBCELL_MEM" --cpus-per-task="$SUBCELL_CPUS" -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err" --wrap="$subcell_cmd")
        if [[ -n "$esm_jid" ]]; then subcell_args=(--dependency=afterok:"$esm_jid" "${subcell_args[@]}"); fi
        subcell_jid=$(submit_sbatch "${subcell_args[@]}")
        echo "[$ds] SubCell query-action job: $subcell_jid -> $ready_h5ad"
    fi

    if [[ "$RUN_TRAIN" == "1" ]]; then
        cfg="$(base_cfg_for "$ds")"
        run_name="crossmod_incontext_${ds}_stack_esm2_650M_to_subcell_mae_rybg"
        out_dir="${RUN_ROOT}/${ds}_stack_esm2_650M_to_subcell_mae_rybg"
        train_args=(--job-name="wm-xmod-train-${ds}" --partition="$PARTITION" --qos="$QOS" --gres="$TRAIN_GRES" --mem="$TRAIN_MEM" --cpus-per-task="$TRAIN_CPUS" -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err" --export=ALL,EMBPY_PIXI_ENV="${TRAIN_PIXI_ENV}")
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
    fi
    echo "[$ds] final training h5ad: $ready_h5ad"
    echo
done

if command -v squeue >/dev/null 2>&1 && [[ "$DRYRUN" != "1" ]]; then
    squeue --me --format="%.18i %.9P %.32j %.10T %.10M %.6D %R" || true
fi
