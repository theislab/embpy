#!/usr/bin/env bash
# =============================================================================
# run_embedding_sweep_mps.sh
#
# Local Apple Silicon / MacBook launcher for sweeping world_model action
# embeddings on Nadig and Replogle with PyTorch MPS.
#
# This is deliberately separate from the SLURM launchers: it runs jobs
# sequentially, uses the local pixi `mps` env, and applies Mac-friendly
# overrides (MPS device, no CUDA AMP, no dataloader pinned memory).
#
# Default profile is a quick local check. For a full-size run:
#
#   PROFILE=full bash src/world_model/world_model/scripts/local/run_embedding_sweep_mps.sh
#
# Common overrides:
#
#   DATASETS="nadig replogle" \
#   EMBEDDINGS="esm2_650M minilm_l6_v2" \
#   EPOCHS=3 \
#   BATCH_SIZE=8 \
#   bash src/world_model/world_model/scripts/local/run_embedding_sweep_mps.sh
#
# When called from `pixi run -e mps wm-sweep-mps`, USE_PIXI=0 is set by
# pixi.toml so the script does not recursively invoke pixi for each run.
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

PIXI_ENV="${PIXI_ENV:-mps}"
USE_PIXI="${USE_PIXI:-1}"
PROFILE="${PROFILE:-quick}" # quick | full
RUN_ROOT="${RUN_ROOT:-runs/world_model/mps_action_sweep}"
SEED="${SEED:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRYRUN="${DRYRUN:-0}"
SETUPS="${SETUPS:-single finetune zeroshot}" # single | finetune | zeroshot

# Keep MPS usable even if a PyTorch op has not been implemented on Metal yet.
# Unsupported ops fall back to CPU instead of crashing the whole sweep.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

case "$PROFILE" in
    quick)
        EPOCHS="${EPOCHS:-5}"
        FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-3}"
        BATCH_SIZE="${BATCH_SIZE:-16}"
        N_TOP_GENES="${N_TOP_GENES:-2000}"
        N_SEQUENCES_PER_EPOCH="${N_SEQUENCES_PER_EPOCH:-1024}"
        SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"
        USE_CELL_EVAL="${USE_CELL_EVAL:-false}"
        ;;
    full)
        EPOCHS="${EPOCHS:-50}"
        FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
        BATCH_SIZE="${BATCH_SIZE:-64}"
        N_TOP_GENES="${N_TOP_GENES:-5000}"
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

DATASETS="${DATASETS:-nadig replogle}"
EMBEDDINGS="${EMBEDDINGS:-$(matrix_value default-embeddings --target mps)}"

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

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required file not found: $path" >&2
        echo "       Set DATASETS to only installed datasets, or update H5AD paths in this script." >&2
        exit 1
    fi
}

safe_key() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '_'
}

embedding_spec_for() {
    matrix_value embedding-spec "$1" --target mps
}

has_setup() {
    case " $SETUPS " in
        *" $1 "*) return 0 ;;
    esac
    if [[ "$1" == "zeroshot" ]]; then
        case " $SETUPS " in
            *" cross "*) return 0 ;;
        esac
    fi
    return 1
}

echo "world_model MPS action-embedding sweep"
echo "  profile:      $PROFILE"
echo "  pixi env:     $PIXI_ENV (USE_PIXI=$USE_PIXI)"
echo "  datasets:     $DATASETS"
echo "  embeddings:   $EMBEDDINGS"
echo "  setups:       $SETUPS"
echo "  output root:  $RUN_ROOT"
echo "  epochs:       $EPOCHS"
echo "  ft epochs:    $FINETUNE_EPOCHS"
echo "  batch size:   $BATCH_SIZE"
echo "  n_top_genes:  $N_TOP_GENES"
echo "  n_seq/epoch:  ${N_SEQUENCES_PER_EPOCH:-full dataset default}"
echo "  cell_eval:    $USE_CELL_EVAL"
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

mkdir -p "$RUN_ROOT"

read -r -a EMBEDDING_LIST <<<"$EMBEDDINGS"

if has_setup single; then
    read -r -a DATASET_LIST <<<"$DATASETS"
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

            if [[ "$kind" == "unsupported" ]]; then
                echo "=== setup=single dataset=${ds} action_embedding=${model} ==="
                echo "Skipping: $target"
                echo
                continue
            fi
            if [[ "$kind" == "convert" ]]; then
                echo "=== setup=single dataset=${ds} action_embedding=${model} ==="
                echo "Skipping conversion-only source on local MPS: $target"
                echo
                continue
            fi

            overrides=(
                "seed=${SEED}"
                "run_name=${run_name}"
                "output_dir=${out_dir}"
                "data.h5ad_path=${h5ad}"
                "data.batch_size=${BATCH_SIZE}"
                "data.n_top_genes=${N_TOP_GENES}"
                "data.num_workers=0"
                "data.pin_memory=false"
                "split.train_fraction=0.8"
                "train.device=mps"
                "train.amp=false"
                "train.n_epochs=${EPOCHS}"
                "train.enable_tensorboard=false"
                "eval.use_cell_eval=${USE_CELL_EVAL}"
                "eval.save_predictions=${SAVE_PREDICTIONS}"
            )
            if [[ -n "$N_SEQUENCES_PER_EPOCH" ]]; then
                overrides+=("data.n_sequences_per_epoch=${N_SEQUENCES_PER_EPOCH}")
            fi

            if [[ "$kind" == "precomputed" ]]; then
                if [[ ! -f "$target" ]]; then
                    echo "=== setup=single dataset=${ds} action_embedding=${model} ==="
                    echo "Skipping missing precomputed table: $target"
                    echo
                    continue
                fi
                store_path="${target%.*}.emstore"
                if [[ ! -d "$store_path" ]]; then
                    echo "Migrating precomputed action embedding to .emstore:"
                    echo "  source: $target"
                    echo "  dest:   $store_path"
                    if [[ "$DRYRUN" != "1" ]]; then
                        run_python python -m embpy.store.migrate \
                            "$target" "$store_path" \
                            --model "$model" --entity-type gene --id-scheme symbol
                    fi
                fi
                overrides+=(
                    "action_embedding.source=store"
                    "action_embedding.store_path=${store_path}"
                    "action_embedding.store_key=gene:${model}"
                )
            else
                overrides+=(
                    "action_embedding.source=bio_embedder"
                    "action_embedding.model_name=${target}"
                    "action_embedding.organism=human"
                    "action_embedding.id_type=symbol"
                    "action_embedding.region=full"
                    "action_embedding.pooling_strategy=mean"
                    "action_embedding.device=mps"
                )
            fi

            echo "=== setup=single dataset=${ds} action_embedding=${model} ==="
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
fi

if has_setup zeroshot || has_setup finetune; then
    for model in "${EMBEDDING_LIST[@]}"; do
        key="$(safe_key "$model")"
        spec="$(embedding_spec_for "$model")"
        kind="${spec%%|*}"
        target="${spec#*|}"

        if [[ "$kind" == "unsupported" ]]; then
            echo "=== setup=cross action_embedding=${model} ==="
            echo "Skipping: $target"
            echo
            continue
        fi
        if [[ "$kind" == "convert" ]]; then
            echo "=== setup=cross action_embedding=${model} ==="
            echo "Skipping conversion-only source on local MPS: $target"
            echo
            continue
        fi

        store_path=""
        if [[ "$kind" == "precomputed" ]]; then
            if [[ ! -f "$target" ]]; then
                echo "=== setup=cross action_embedding=${model} ==="
                echo "Skipping missing precomputed table: $target"
                echo
                continue
            fi
            store_path="${target%.*}.emstore"
            if [[ ! -d "$store_path" ]]; then
                echo "Migrating precomputed action embedding to .emstore:"
                echo "  source: $target"
                echo "  dest:   $store_path"
                if [[ "$DRYRUN" != "1" ]]; then
                    run_python python -m embpy.store.migrate \
                        "$target" "$store_path" \
                        --model "$model" --entity-type gene --id-scheme symbol
                fi
            fi
        fi

        for pair in "nadig replogle" "replogle nadig"; do
            set -- $pair
            source_ds="$1"
            target_ds="$2"
            source_cfg="$(base_cfg_for "$source_ds")"
            target_cfg="$(base_cfg_for "$target_ds")"
            source_h5ad="$(h5ad_for "$source_ds")"
            target_h5ad="$(h5ad_for "$target_ds")"
            require_file "$source_h5ad"
            require_file "$target_h5ad"

            overrides=(
                "seed=${SEED}"
                "data.batch_size=${BATCH_SIZE}"
                "data.n_top_genes=${N_TOP_GENES}"
                "data.num_workers=0"
                "data.pin_memory=false"
                "train.device=mps"
                "train.amp=false"
                "train.n_epochs=${EPOCHS}"
                "train.enable_tensorboard=false"
                "eval.use_cell_eval=${USE_CELL_EVAL}"
                "eval.save_predictions=${SAVE_PREDICTIONS}"
            )
            if [[ -n "$N_SEQUENCES_PER_EPOCH" ]]; then
                overrides+=("data.n_sequences_per_epoch=${N_SEQUENCES_PER_EPOCH}")
            fi
            if [[ "$kind" == "precomputed" ]]; then
                overrides+=(
                    "action_embedding.source=store"
                    "action_embedding.store_path=${store_path}"
                    "action_embedding.store_key=gene:${model}"
                )
            else
                overrides+=(
                    "action_embedding.source=bio_embedder"
                    "action_embedding.model_name=${target}"
                    "action_embedding.organism=human"
                    "action_embedding.id_type=symbol"
                    "action_embedding.region=full"
                    "action_embedding.pooling_strategy=mean"
                    "action_embedding.device=mps"
                )
            fi

            if has_setup finetune; then
                run_name="finetune_${source_ds}_to_${target_ds}_${key}_mps"
                out_dir="${RUN_ROOT}/finetune/${source_ds}_to_${target_ds}/${key}"
                echo "=== setup=finetune source=${source_ds} target=${target_ds} action_embedding=${model} ==="
                echo "output_dir=${out_dir}"
                if [[ "$DRYRUN" == "1" ]]; then
                    print_run_python python -m world_model.scripts.cross_dataset_train_eval \
                        --mode fine-tune \
                        --finetune-epochs "$FINETUNE_EPOCHS" \
                        --source-config "$source_cfg" \
                        --target-config "$target_cfg" \
                        --source-h5ad "$source_h5ad" \
                        --target-h5ad "$target_h5ad" \
                        --source-name "$source_ds" \
                        --target-name "$target_ds" \
                        --output-dir "$out_dir" \
                        --run-name "$run_name" \
                        --log-level "$LOG_LEVEL" \
                        "${overrides[@]}"
                else
                    run_python python -m world_model.scripts.cross_dataset_train_eval \
                        --mode fine-tune \
                        --finetune-epochs "$FINETUNE_EPOCHS" \
                        --source-config "$source_cfg" \
                        --target-config "$target_cfg" \
                        --source-h5ad "$source_h5ad" \
                        --target-h5ad "$target_h5ad" \
                        --source-name "$source_ds" \
                        --target-name "$target_ds" \
                        --output-dir "$out_dir" \
                        --run-name "$run_name" \
                        --log-level "$LOG_LEVEL" \
                        "${overrides[@]}"
                fi
                echo
            fi

            if has_setup zeroshot; then
                run_name="zeroshot_${source_ds}_to_${target_ds}_${key}_mps"
                out_dir="${RUN_ROOT}/zeroshot/${source_ds}_to_${target_ds}/${key}"
                echo "=== setup=zeroshot source=${source_ds} target=${target_ds} action_embedding=${model} ==="
                echo "output_dir=${out_dir}"
                if [[ "$DRYRUN" == "1" ]]; then
                    print_run_python python -m world_model.scripts.cross_dataset_train_eval \
                        --mode zero-shot \
                        --source-config "$source_cfg" \
                        --target-config "$target_cfg" \
                        --source-h5ad "$source_h5ad" \
                        --target-h5ad "$target_h5ad" \
                        --source-name "$source_ds" \
                        --target-name "$target_ds" \
                        --output-dir "$out_dir" \
                        --run-name "$run_name" \
                        --log-level "$LOG_LEVEL" \
                        "${overrides[@]}"
                else
                    run_python python -m world_model.scripts.cross_dataset_train_eval \
                        --mode zero-shot \
                        --source-config "$source_cfg" \
                        --target-config "$target_cfg" \
                        --source-h5ad "$source_h5ad" \
                        --target-h5ad "$target_h5ad" \
                        --source-name "$source_ds" \
                        --target-name "$target_ds" \
                        --output-dir "$out_dir" \
                        --run-name "$run_name" \
                        --log-level "$LOG_LEVEL" \
                        "${overrides[@]}"
                fi
                echo
            fi
        done
    done
fi

echo "Sweep complete. Results are under: $RUN_ROOT"
