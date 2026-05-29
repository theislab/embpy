#!/usr/bin/env bash
# =============================================================================
# run_embedding_sweep_mps.sh
#
# Local Apple Silicon launcher for within-dataset world-model embedding runs.
# It mirrors the cluster flow: attach action embeddings to an AnnData copy,
# then train from adata.obsm state/action matrices.
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

PIXI_ENV="${PIXI_ENV:-mps}"
USE_PIXI="${USE_PIXI:-1}"
PROFILE="${PROFILE:-quick}" # quick | full
RUN_ROOT="${RUN_ROOT:-runs/world_model/mps_action_sweep}"
ACTION_H5AD_ROOT="${ACTION_H5AD_ROOT:-${RUN_ROOT}/_cache/action_h5ad}"
ACTION_NPZ_ROOT="${ACTION_NPZ_ROOT:-${RUN_ROOT}/_cache/action_embeddings}"
STATE_OBSM_KEY="${STATE_OBSM_KEY:-X_state}"
SEED="${SEED:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRYRUN="${DRYRUN:-0}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

case "$PROFILE" in
    quick)
        EPOCHS="${EPOCHS:-5}"
        BATCH_SIZE="${BATCH_SIZE:-16}"
        N_SEQUENCES_PER_EPOCH="${N_SEQUENCES_PER_EPOCH:-1024}"
        SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"
        USE_CELL_EVAL="${USE_CELL_EVAL:-false}"
        ;;
    full)
        EPOCHS="${EPOCHS:-50}"
        BATCH_SIZE="${BATCH_SIZE:-64}"
        N_SEQUENCES_PER_EPOCH="${N_SEQUENCES_PER_EPOCH:-}"
        SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-true}"
        USE_CELL_EVAL="${USE_CELL_EVAL:-true}"
        ;;
    *)
        echo "ERROR: PROFILE must be 'quick' or 'full', got '$PROFILE'." >&2
        exit 2
        ;;
esac

matrix_value() {
    if [[ "$USE_PIXI" == "1" ]]; then
        pixi run -e "$PIXI_ENV" -- python -m world_model.configs.run_matrix "$@"
    else
        python -m world_model.configs.run_matrix "$@"
    fi
}

run_python() {
    if [[ "$USE_PIXI" == "1" ]]; then
        pixi run -e "$PIXI_ENV" -- "$@"
    else
        "$@"
    fi
}

print_run_python() {
    if [[ "$USE_PIXI" == "1" ]]; then
        printf '  %q' pixi run -e "$PIXI_ENV" -- "$@"
    else
        printf '  %q' "$@"
    fi
    echo
}

base_cfg_for() {
    matrix_value base-config "$1"
}

h5ad_for() {
    case "$1" in
        nadig) echo "${NADIG_H5AD:-$(matrix_value h5ad nadig)}" ;;
        replogle) echo "${REPLOGLE_H5AD:-$(matrix_value h5ad replogle)}" ;;
        *) return 1 ;;
    esac
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required file not found: $path" >&2
        exit 1
    fi
}

safe_key() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '_'
}

embedding_spec_for() {
    matrix_value embedding-spec "$1" --target mps
}

DATASETS="${DATASETS:-nadig replogle}"
EMBEDDINGS="${EMBEDDINGS:-$(matrix_value default-embeddings --target mps)}"

echo "world_model MPS within-dataset embedding sweep"
echo "  profile:      $PROFILE"
echo "  pixi env:     $PIXI_ENV (USE_PIXI=$USE_PIXI)"
echo "  datasets:     $DATASETS"
echo "  embeddings:   $EMBEDDINGS"
echo "  output root:  $RUN_ROOT"
echo "  state obsm:   $STATE_OBSM_KEY"
echo "  epochs:       $EPOCHS"
echo "  batch size:   $BATCH_SIZE"
echo "  n_seq/epoch:  ${N_SEQUENCES_PER_EPOCH:-full dataset default}"
echo

if [[ "$DRYRUN" == "1" ]]; then
    echo "dry-run: skipping PyTorch MPS availability check"
else
    run_python python - <<'PY'
import torch

available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"torch={torch.__version__} mps_available={available}")
if not available:
    raise SystemExit("PyTorch MPS is not available. Run `pixi run -e mps verify-mps` first.")
PY
fi

mkdir -p "$RUN_ROOT" "$ACTION_H5AD_ROOT" "$ACTION_NPZ_ROOT"

read -r -a DATASET_LIST <<<"$DATASETS"
read -r -a EMBEDDING_LIST <<<"$EMBEDDINGS"

prepare_action_h5ad() {
    local ds="$1"
    local h5ad="$2"
    local key="$3"
    local kind="$4"
    local target="$5"
    local out_h5ad="$6"
    local out_npz="$7"
    local obsm_key="$8"

    echo "AnnData action attach: ${out_h5ad} obsm[${obsm_key}]"
    if [[ "$DRYRUN" == "1" ]]; then
        if [[ "$kind" == "precomputed" ]]; then
            print_run_python python -m world_model.scripts.embed_perturbations \
                --dataset "$ds" --h5ad "$h5ad" --table "$target" \
                --output "$out_npz" --output-h5ad "$out_h5ad" \
                --obsm-key "$obsm_key"
        else
            print_run_python python -m world_model.scripts.embed_perturbations \
                --dataset "$ds" --h5ad "$h5ad" --model "$target" \
                --output "$out_npz" --output-h5ad "$out_h5ad" \
                --obsm-key "$obsm_key" \
                --region full --pooling-strategy mean \
                --organism human --id-type symbol --device mps
        fi
        return
    fi

    if [[ -f "$out_h5ad" ]]; then
        return
    fi
    if [[ "$kind" == "precomputed" ]]; then
        run_python python -m world_model.scripts.embed_perturbations \
            --dataset "$ds" --h5ad "$h5ad" --table "$target" \
            --output "$out_npz" --output-h5ad "$out_h5ad" \
            --obsm-key "$obsm_key"
    else
        run_python python -m world_model.scripts.embed_perturbations \
            --dataset "$ds" --h5ad "$h5ad" --model "$target" \
            --output "$out_npz" --output-h5ad "$out_h5ad" \
            --obsm-key "$obsm_key" \
            --region full --pooling-strategy mean \
            --organism human --id-type symbol --device mps
    fi
}

for ds in "${DATASET_LIST[@]}"; do
    if ! cfg="$(base_cfg_for "$ds")"; then
        echo "ERROR: unknown dataset '$ds'. Supported: nadig replogle." >&2
        exit 2
    fi
    h5ad="$(h5ad_for "$ds")"
    require_file "$h5ad"

    for model in "${EMBEDDING_LIST[@]}"; do
        key="$(safe_key "$model")"
        run_name="single_${ds}_${key}_mps"
        out_dir="${RUN_ROOT}/single/${ds}/${key}"
        spec="$(embedding_spec_for "$model")"
        kind="${spec%%|*}"
        target="${spec#*|}"
        obsm_key="X_pert_${key}"
        action_h5ad="${ACTION_H5AD_ROOT}/${ds}_${key}.h5ad"
        action_npz="${ACTION_NPZ_ROOT}/${ds}_${key}.npz"

        echo "=== dataset=${ds} action_embedding=${model} ==="
        if [[ "$kind" == "unsupported" || "$kind" == "convert" ]]; then
            echo "Skipping: $target"
            echo
            continue
        fi
        if [[ "$kind" == "precomputed" && ! -f "$target" ]]; then
            echo "Skipping missing precomputed table: $target"
            echo
            continue
        fi

        prepare_action_h5ad "$ds" "$h5ad" "$key" "$kind" "$target" "$action_h5ad" "$action_npz" "$obsm_key"

        overrides=(
            "seed=${SEED}"
            "run_name=${run_name}"
            "output_dir=${out_dir}"
            "data.h5ad_path=${action_h5ad}"
            "data.state_obsm_key=${STATE_OBSM_KEY}"
            "data.batch_size=${BATCH_SIZE}"
            "data.num_workers=0"
            "data.pin_memory=false"
            "split.train_fraction=0.8"
            "train.device=mps"
            "train.amp=false"
            "train.n_epochs=${EPOCHS}"
            "train.enable_tensorboard=false"
            "eval.use_cell_eval=${USE_CELL_EVAL}"
            "eval.save_predictions=${SAVE_PREDICTIONS}"
            "action_embedding.source=anndata_obsm"
            "action_embedding.obsm_key=${obsm_key}"
            "action_embedding.model_name=${model}"
        )
        if [[ -n "$N_SEQUENCES_PER_EPOCH" ]]; then
            overrides+=("data.n_sequences_per_epoch=${N_SEQUENCES_PER_EPOCH}")
        fi

        echo "output_dir=${out_dir}"
        if [[ "$DRYRUN" == "1" ]]; then
            print_run_python python -m world_model.scripts.train --config "$cfg" --log-level "$LOG_LEVEL" "${overrides[@]}"
        else
            run_python python -m world_model.scripts.train \
                --config "$cfg" \
                --log-level "$LOG_LEVEL" \
                "${overrides[@]}"
        fi
        echo
    done
done

echo "Sweep complete. Results are under: $RUN_ROOT"
