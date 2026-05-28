#!/usr/bin/env bash
# =============================================================================
# submit_world_model_ready_adatas.sh
#
# Build world-model-ready AnnData files on SLURM.
#
# For each selected dataset this submits:
#   1. STACK cell/state embedding job:
#        input .h5ad -> .obsm["X_stack"]
#   2. one action / perturbation gene embedding NPZ job per selected
#      embedding, dependent on (1)
#   3. one merge job per dataset that writes all action embeddings into
#      the same AnnData as .obsm["X_pert_<embedding>"] plus embpy
#      perturbation payloads in .uns
#
# The final artifact is:
#   runs/_cache/world_model_ready_h5ad/<dataset>_stack_all_gene_embeddings.h5ad
#
# Required:
#   STACK_CHECKPOINT=/path/to/STACK/checkpoint.ckpt
#   STACK_GENELIST=/path/to/basecount_1000per_15000max.pkl
#
# Useful:
#   DATASET=nadig|replogle|both       default: both
#   EMBEDDINGS=all                    default: all usable catalog entries
#   EMB=<one embedding>               backwards-compatible single embedding
#   DRYRUN=1                          print sbatch commands without submitting
#   REUSE_EXISTING=1                  reuse existing state/action outputs
#   REUSE_STATE=1                     reuse existing state .h5ad/.npz only
#   REUSE_ACTION=1                    reuse existing action .npz files only
#   RUN_TRAIN=1                       optionally chain train_embedding.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONNOUSERSITE=1

SELF="src/world_model/world_model/scripts/submit/submit_world_model_ready_adatas.sh"
SLURM_DIR="src/world_model/world_model/scripts/slurm"
TRAIN_LAUNCHER="${SLURM_DIR}/train_embedding.sbatch"

DATASET="${DATASET:-both}"
EMB="${EMB:-}"
EMBEDDINGS="${EMBEDDINGS:-}"
STATE_KIND="${STATE_KIND:-stack}"
STATE_OBSM_KEY="${STATE_OBSM_KEY:-X_stack}"
ACTION_OBSM_KEY="${ACTION_OBSM_KEY:-}"
PERTURBATION_KEY="${PERTURBATION_KEY:-perturbation}"

STACK_CHECKPOINT="${STACK_CHECKPOINT:-}"
STACK_GENELIST="${STACK_GENELIST:-}"
STACK_GENE_NAME_COL="${STACK_GENE_NAME_COL:-}"
STACK_BATCH_SIZE="${STACK_BATCH_SIZE:-64}"
STACK_CACHE_DIR="${STACK_CACHE_DIR:-runs/_cache/state_backbone}"

PARTITION="${PARTITION:-gpu_p}"
QOS="${QOS:-gpu_normal}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p}"
CPU_QOS="${CPU_QOS:-cpu_normal}"
STACK_TIME="${STACK_TIME:-24:00:00}"
STACK_MEM="${STACK_MEM:-128G}"
STACK_CPUS="${STACK_CPUS:-8}"
STACK_GPU_CONSTRAINT="${STACK_GPU_CONSTRAINT:-}"
ACTION_TIME="${ACTION_TIME:-08:00:00}"
ACTION_MEM="${ACTION_MEM:-64G}"
ACTION_CPUS="${ACTION_CPUS:-8}"
ACTION_GPU_CONSTRAINT="${ACTION_GPU_CONSTRAINT:-a100_80gb|h100_80gb}"
PRECOMPUTED_ACTION_TIME="${PRECOMPUTED_ACTION_TIME:-02:00:00}"
PRECOMPUTED_ACTION_MEM="${PRECOMPUTED_ACTION_MEM:-64G}"
PRECOMPUTED_ACTION_CPUS="${PRECOMPUTED_ACTION_CPUS:-4}"
ASSEMBLE_TIME="${ASSEMBLE_TIME:-04:00:00}"
ASSEMBLE_MEM="${ASSEMBLE_MEM:-128G}"
ASSEMBLE_CPUS="${ASSEMBLE_CPUS:-4}"

MATRIX_PIXI_ENV="${MATRIX_PIXI_ENV:-gpu}"
STATE_PIXI_ENV="${STATE_PIXI_ENV:-arc-gpu}"
ASSEMBLE_PIXI_ENV="${ASSEMBLE_PIXI_ENV:-gpu}"
TRAIN_PIXI_ENV="${TRAIN_PIXI_ENV:-gpu}"
TRAIN_GPU_CONSTRAINT="${TRAIN_GPU_CONSTRAINT:-h100_80gb}"
TRAIN_GRES="${TRAIN_GRES:-gpu:h100:1}"
TRAIN_MEM="${TRAIN_MEM:-128G}"
TRAIN_CPUS="${TRAIN_CPUS:-8}"
DRYRUN="${DRYRUN:-0}"
RUN_TRAIN="${RUN_TRAIN:-0}"
FAIL_ON_UNRESOLVED="${FAIL_ON_UNRESOLVED:-0}"
STRICT_RESOLVER="${STRICT_RESOLVER:-0}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"
REUSE_STATE="${REUSE_STATE:-$REUSE_EXISTING}"
REUSE_ACTION="${REUSE_ACTION:-$REUSE_EXISTING}"

STATE_NPZ_DIR="${STATE_NPZ_DIR:-runs/_cache/state_embeddings}"
STATE_H5AD_DIR="${STATE_H5AD_DIR:-runs/_cache/state_h5ad}"
ACTION_NPZ_DIR="${ACTION_NPZ_DIR:-runs/_cache/action_embeddings}"
READY_H5AD_DIR="${READY_H5AD_DIR:-runs/_cache/world_model_ready_h5ad}"
TMP_ROOT="${TMP_ROOT:-runs/_tmp}"
LOG_ROOT="${LOG_ROOT:-logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/world_model_ready_adatas}"

mkdir -p "$LOG_DIR" "$STATE_NPZ_DIR" "$STATE_H5AD_DIR" "$ACTION_NPZ_DIR" "$READY_H5AD_DIR" "$TMP_ROOT"

matrix_value() {
    pixi run -e "$MATRIX_PIXI_ENV" -- python -m world_model.configs.run_matrix "$@"
}

base_cfg_for() {
    matrix_value base-config "$1"
}

h5ad_for() {
    matrix_value h5ad "$1"
}

CATALOG=()
while IFS= read -r row; do
    CATALOG+=("$row")
done < <(matrix_value catalog --target all)

lookup_embedding() {
    local row n k t d
    for row in "${CATALOG[@]}"; do
        IFS='|' read -r n k t d <<<"$row"
        if [[ "$n" == "$1" ]]; then
            echo "$k|$t|$d"
            return 0
        fi
    done
    return 1
}

resolve_embedding() {
    local emb="$1"
    local resolved
    if resolved="$(lookup_embedding "$emb")"; then
        echo "$resolved"
    else
        echo "bio|${emb}|custom BioEmbedder MODEL_REGISTRY key"
    fi
}

print_catalog() {
    local row n k t d
    echo "World-model-ready AnnData builder"
    echo
    echo "Required STACK inputs:"
    echo "  STACK_CHECKPOINT=/path/to/bc_large.ckpt"
    echo "  STACK_GENELIST=/path/to/basecount_1000per_15000max.pkl"
    echo
    printf "  %-24s %-11s %s\n" "EMB" "KIND" "DESCRIPTION"
    for row in "${CATALOG[@]}"; do
        IFS='|' read -r n k t d <<<"$row"
        printf "  %-24s %-11s %s\n" "$n" "$k" "$d"
    done
    echo
    echo "Example:"
    echo "  DATASET=both EMBEDDINGS=all \\"
    echo "    STACK_CHECKPOINT=/path/to/bc_large.ckpt \\"
    echo "    STACK_GENELIST=/path/to/basecount_1000per_15000max.pkl \\"
    echo "    bash $SELF"
    echo
    echo "Single embedding:"
    echo "  DATASET=both EMB=esm2_650M STACK_CHECKPOINT=... STACK_GENELIST=... bash $SELF"
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
    list|--list|-l|help|--help|-h) print_catalog; exit 0 ;;
esac

if [[ "$STATE_KIND" != "stack" ]]; then
    echo "ERROR: this submitter is for STACK state embeddings; got STATE_KIND='$STATE_KIND'." >&2
    exit 2
fi

if [[ -z "$EMBEDDINGS" ]]; then
    EMBEDDINGS="${EMB:-all}"
fi

if [[ "$EMBEDDINGS" == "all" || "$EMBEDDINGS" == "default" ]]; then
    expanded_embeddings="$(matrix_value default-embeddings --target slurm)"
else
    expanded_embeddings="$EMBEDDINGS"
fi

EMBEDDING_KEYS=()
for key in $expanded_embeddings; do
    EMBEDDING_KEYS+=("$key")
done

if [[ "${#EMBEDDING_KEYS[@]}" -eq 0 ]]; then
    echo "ERROR: no action embeddings selected." >&2
    echo >&2
    print_catalog >&2
    exit 2
fi

if [[ -n "$ACTION_OBSM_KEY" && "${#EMBEDDING_KEYS[@]}" -ne 1 ]]; then
    echo "ERROR: ACTION_OBSM_KEY can only be set when exactly one embedding is selected." >&2
    echo "       Selected: ${EMBEDDING_KEYS[*]}" >&2
    exit 2
fi
if [[ -n "$ACTION_OBSM_KEY" && "$ACTION_OBSM_KEY" != X_pert_* ]]; then
    echo "ERROR: ACTION_OBSM_KEY must start with 'X_pert_' so the assembly step can map it back to an embedding suffix." >&2
    exit 2
fi

if [[ -z "$STACK_CHECKPOINT" || -z "$STACK_GENELIST" ]]; then
    echo "ERROR: set both STACK_CHECKPOINT and STACK_GENELIST." >&2
    echo >&2
    echo "Example:" >&2
    echo "  EMB=$EMB STACK_CHECKPOINT=/path/to/bc_large.ckpt STACK_GENELIST=/path/to/basecount_1000per_15000max.pkl bash $SELF" >&2
    exit 2
fi

if [[ "$DRYRUN" != "1" ]]; then
    [[ -f "$STACK_CHECKPOINT" ]] || { echo "ERROR: STACK_CHECKPOINT not found: $STACK_CHECKPOINT" >&2; exit 1; }
    [[ -f "$STACK_GENELIST" ]] || { echo "ERROR: STACK_GENELIST not found: $STACK_GENELIST" >&2; exit 1; }
fi

case "$DATASET" in
    nadig|replogle) DATASETS=("$DATASET") ;;
    both)           DATASETS=(nadig replogle) ;;
    *) echo "ERROR: DATASET must be nadig|replogle|both, got '$DATASET'." >&2; exit 2 ;;
esac

for emb in "${EMBEDDING_KEYS[@]}"; do
    IFS='|' read -r kind target desc <<<"$(resolve_embedding "$emb")"
    if [[ "$kind" == "convert" ]]; then
        echo "ERROR: '$emb' needs one-time conversion before it can be attached to AnnData." >&2
        echo "       target: $target" >&2
        echo "       $desc" >&2
        exit 3
    fi
    if [[ "$kind" == "precomputed" && "$DRYRUN" != "1" && ! -f "$target" ]]; then
        echo "ERROR: precomputed embedding not found for '$emb': $target" >&2
        exit 1
    fi
done

echo "World-model-ready AnnData build"
echo "  project:          $PROJECT_DIR"
echo "  datasets:         ${DATASETS[*]}"
echo "  state embedding:  STACK -> obsm[$STATE_OBSM_KEY]"
echo "  action embeddings (${#EMBEDDING_KEYS[@]}): ${EMBEDDING_KEYS[*]}"
echo "  final h5ad dir:   $READY_H5AD_DIR"
echo "  logs:             $LOG_DIR"
echo "  reuse state:      $REUSE_STATE"
echo "  reuse action:     $REUSE_ACTION"
echo "  dry run:          $DRYRUN"
echo

for ds in "${DATASETS[@]}"; do
    source_h5ad="$(h5ad_for "$ds")"
    state_npz="${STATE_NPZ_DIR}/${ds}_${STATE_KIND}.npz"
    state_h5ad="${STATE_H5AD_DIR}/${ds}_${STATE_KIND}.h5ad"

    stack_gene_col_arg=""
    if [[ -n "$STACK_GENE_NAME_COL" ]]; then
        stack_gene_col_arg=" --stack-gene-name-col ${STACK_GENE_NAME_COL}"
    fi

    stack_jid=""
    if [[ "$REUSE_STATE" == "1" && -s "$state_h5ad" ]]; then
        echo "[$ds] reusing STACK state AnnData: $state_h5ad obsm[$STATE_OBSM_KEY]"
    else
        stack_cmd="set -euo pipefail; cd ${PROJECT_DIR}; \
            export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
            export PYTHONNOUSERSITE=1; \
            export TMPDIR=\"${PROJECT_DIR}/${TMP_ROOT}/stack-\${SLURM_JOB_ID:-manual}\"; \
            mkdir -p \"\$TMPDIR\"; \
            trap 'rm -rf \"\$TMPDIR\"' EXIT; \
            pixi run -e ${STATE_PIXI_ENV} -- python -m world_model.scripts.encode_cells \
                --kind stack \
                --adata ${source_h5ad} \
                --output ${state_npz} \
                --output-h5ad ${state_h5ad} \
                --obsm-key ${STATE_OBSM_KEY} \
                --device cuda \
                --batch-size ${STACK_BATCH_SIZE} \
                --cache-dir ${STACK_CACHE_DIR} \
                --stack-checkpoint ${STACK_CHECKPOINT} \
                --stack-genelist ${STACK_GENELIST}${stack_gene_col_arg}"

        echo "[$ds] submitting STACK state embedding job ..."
        stack_sbatch_args=(
            --job-name="wm-stack-${ds}" \
            --partition="$PARTITION" --qos="$QOS" \
            --gres=gpu:1
        )
        if [[ -n "$STACK_GPU_CONSTRAINT" ]]; then
            stack_sbatch_args+=(--constraint="$STACK_GPU_CONSTRAINT")
        fi
        stack_sbatch_args+=(
            --time="$STACK_TIME" --mem="$STACK_MEM" --cpus-per-task="$STACK_CPUS"
            -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err"
            --wrap="$stack_cmd"
        )
        stack_jid=$(submit_sbatch "${stack_sbatch_args[@]}")
        echo "[$ds]   STACK job: $stack_jid -> $state_h5ad obsm[$STATE_OBSM_KEY]"
    fi

    fail_args=""
    if [[ "$FAIL_ON_UNRESOLVED" == "1" ]]; then
        fail_args="${fail_args} --fail-on-unresolved"
    fi
    if [[ "$STRICT_RESOLVER" == "1" ]]; then
        fail_args="${fail_args} --strict-resolver"
    fi

    action_jids=()
    assemble_embedding_args=()
    for emb in "${EMBEDDING_KEYS[@]}"; do
        IFS='|' read -r kind target desc <<<"$(resolve_embedding "$emb")"
        action_npz="${ACTION_NPZ_DIR}/${ds}_${emb}.npz"
        assemble_name="$emb"
        if [[ -n "$ACTION_OBSM_KEY" ]]; then
            assemble_name="${ACTION_OBSM_KEY#X_pert_}"
        fi

        if [[ "$REUSE_ACTION" == "1" && -s "$action_npz" ]]; then
            echo "[$ds][$emb] reusing action NPZ: $action_npz"
            assemble_embedding_args+=("${assemble_name}=${action_npz}")
            continue
        fi

        if [[ "$kind" == "precomputed" ]]; then
            action_pixi_env="gpu"
            action_cmd="set -euo pipefail; cd ${PROJECT_DIR}; \
                export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
                export PYTHONNOUSERSITE=1; \
                pixi run -e ${action_pixi_env} -- python -m world_model.scripts.embed_perturbations \
                    --dataset ${ds} \
                    --h5ad ${state_h5ad} \
                    --table ${target} \
                    --output ${action_npz} \
                    --perturbation-key ${PERTURBATION_KEY}${fail_args}"
            action_sbatch_args=(
                --job-name="wm-act-${ds}-${emb}"
                --partition="$CPU_PARTITION" --qos="$CPU_QOS"
                --time="$PRECOMPUTED_ACTION_TIME" --mem="$PRECOMPUTED_ACTION_MEM" --cpus-per-task="$PRECOMPUTED_ACTION_CPUS"
                -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err"
                --wrap="$action_cmd"
            )
        else
            action_pixi_env="$(matrix_value pixi-env "$target")"
            action_cmd="set -euo pipefail; cd ${PROJECT_DIR}; \
                export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
                export PYTHONNOUSERSITE=1; \
                export TMPDIR=\"${PROJECT_DIR}/${TMP_ROOT}/action-\${SLURM_JOB_ID:-manual}\"; \
                mkdir -p \"\$TMPDIR\"; \
                trap 'rm -rf \"\$TMPDIR\"' EXIT; \
                pixi run -e ${action_pixi_env} -- python -m world_model.scripts.embed_perturbations \
                    --dataset ${ds} \
                    --h5ad ${state_h5ad} \
                    --model ${target} \
                    --output ${action_npz} \
                    --perturbation-key ${PERTURBATION_KEY} \
                    --device cuda \
                    --region full \
                    --pooling-strategy mean \
                    --organism human \
                    --id-type symbol${fail_args}"
            action_sbatch_args=(
                --job-name="wm-act-${ds}-${emb}"
                --partition="$PARTITION" --qos="$QOS"
                --gres=gpu:1
            )
            if [[ -n "$ACTION_GPU_CONSTRAINT" ]]; then
                action_sbatch_args+=(--constraint="$ACTION_GPU_CONSTRAINT")
            fi
            action_sbatch_args+=(
                --time="$ACTION_TIME" --mem="$ACTION_MEM" --cpus-per-task="$ACTION_CPUS"
                -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err"
                --wrap="$action_cmd"
            )
        fi
        if [[ -n "$stack_jid" ]]; then
            action_sbatch_args=(--dependency=afterok:"${stack_jid}" "${action_sbatch_args[@]}")
        fi

        echo "[$ds][$emb] submitting action embedding job (${kind}, pixi env: ${action_pixi_env}) ..."
        action_jid=$(submit_sbatch "${action_sbatch_args[@]}")
        echo "[$ds][$emb]   action NPZ job: $action_jid -> $action_npz"
        action_jids+=("$action_jid")
        assemble_embedding_args+=("${assemble_name}=${action_npz}")
    done

    ready_h5ad="${READY_H5AD_DIR}/${ds}_${STATE_KIND}_all_gene_embeddings.h5ad"
    assemble_args=""
    for item in "${assemble_embedding_args[@]}"; do
        assemble_args="${assemble_args} --embedding ${item}"
    done
    assemble_fail_arg=""
    if [[ "$FAIL_ON_UNRESOLVED" == "1" ]]; then
        assemble_fail_arg=" --fail-on-unresolved"
    fi
    assemble_dependency=""
    if [[ "${#action_jids[@]}" -gt 0 ]]; then
        assemble_dependency="$(IFS=:; echo "${action_jids[*]}")"
    elif [[ -n "$stack_jid" ]]; then
        assemble_dependency="$stack_jid"
    fi
    assemble_cmd="set -euo pipefail; cd ${PROJECT_DIR}; \
        export PATH=\"\$HOME/.pixi/bin:\$PATH\"; \
        export PYTHONNOUSERSITE=1; \
        pixi run -e ${ASSEMBLE_PIXI_ENV} -- python -m world_model.scripts.assemble_ready_adata \
            --input-h5ad ${state_h5ad} \
            --output-h5ad ${ready_h5ad} \
            --state-obsm-key ${STATE_OBSM_KEY} \
            --perturbation-key ${PERTURBATION_KEY}${assemble_args}${assemble_fail_arg}"

    echo "[$ds] submitting final AnnData assembly job ..."
    assemble_sbatch_args=(
        --job-name="wm-ready-${ds}-all"
        --partition="$CPU_PARTITION" --qos="$CPU_QOS"
        --time="$ASSEMBLE_TIME" --mem="$ASSEMBLE_MEM" --cpus-per-task="$ASSEMBLE_CPUS"
        -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err"
        --wrap="$assemble_cmd"
    )
    if [[ -n "$assemble_dependency" ]]; then
        assemble_sbatch_args=(--dependency=afterok:"${assemble_dependency}" "${assemble_sbatch_args[@]}")
    fi
    assemble_jid=$(submit_sbatch "${assemble_sbatch_args[@]}")
    echo "[$ds]   ready AnnData job: $assemble_jid -> $ready_h5ad"
    echo "[$ds]   state key: obsm[$STATE_OBSM_KEY]"
    echo "[$ds]   action keys: ${EMBEDDING_KEYS[*]/#/X_pert_}"

    if [[ "$RUN_TRAIN" == "1" ]]; then
        cfg="$(base_cfg_for "$ds")"
        for emb in "${EMBEDDING_KEYS[@]}"; do
            action_obsm_key="${ACTION_OBSM_KEY:-X_pert_${emb}}"
            run_name="single_${ds}_${STATE_KIND}_${emb}"
            out_dir="runs/world_model/${run_name}"
            echo "[$ds][$emb] chaining training after ready AnnData build ..."
            train_jid=$(submit_sbatch \
                --job-name="wm-${ds}-${emb}" \
                --partition="$PARTITION" --qos="$QOS" \
                --gres="$TRAIN_GRES" --constraint="$TRAIN_GPU_CONSTRAINT" \
                --mem="$TRAIN_MEM" --cpus-per-task="$TRAIN_CPUS" \
                -o "${LOG_DIR}/%x_%j.out" -e "${LOG_DIR}/%x_%j.err" \
                --dependency=afterok:"${assemble_jid}" \
                --export=ALL,EMBPY_PIXI_ENV="${TRAIN_PIXI_ENV}" \
                "$TRAIN_LAUNCHER" "$cfg" \
                "data.h5ad_path=${ready_h5ad}" \
                "data.state_obsm_key=${STATE_OBSM_KEY}" \
                "state_backbone.kind=stack" \
                "action_embedding.source=anndata_obsm" \
                "action_embedding.obsm_key=${action_obsm_key}" \
                "action_embedding.model_name=${emb}" \
                "run_name=${run_name}" \
                "output_dir=${out_dir}")
            echo "[$ds][$emb]   train job: $train_jid -> $out_dir"
        done
    fi
    echo
done

if command -v squeue >/dev/null 2>&1 && [[ "$DRYRUN" != "1" ]]; then
    squeue --me --format="%.18i %.9P %.26j %.8T %.10M %.6D %R" || true
fi
