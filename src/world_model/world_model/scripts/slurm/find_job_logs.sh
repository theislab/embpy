#!/usr/bin/env bash
# Locate Slurm accounting, dependencies, manifests, stdout, and stderr for job IDs.
#
# Usage:
#   bash src/world_model/world_model/scripts/slurm/find_job_logs.sh 37116120 37013726
#
# Environment:
#   PROJECT_DIR=/path/to/embpy   Repository root. Defaults to the current tree.
#   TAIL_LINES=120               Lines to show from each discovered log.
#   ARTIFACT_LINES=40            Lines to show from each discovered train.log.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustre/groups/ml01/workspace/goncalo.pinto/embpy}"
TAIL_LINES="${TAIL_LINES:-120}"
ARTIFACT_LINES="${ARTIFACT_LINES:-40}"

if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 <job_id> [job_id ...]" >&2
    exit 2
fi

cd "$PROJECT_DIR"

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

print_sacct() {
    local ids="$1"
    if ! have_cmd sacct; then
        echo "sacct not found on PATH"
        return 0
    fi
    sacct -j "$ids" -X \
        --format=JobIDRaw,JobName%45,State,ExitCode,Reason%45,Elapsed,NodeList%20 \
        2>/dev/null || true
}

job_dependency_ids() {
    local job="$1"
    if ! have_cmd scontrol; then
        return 0
    fi
    local dep
    dep="$(
        scontrol show job -o "$job" 2>/dev/null \
            | awk '{
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^Dependency=/) {
                        sub(/^Dependency=/, "", $i)
                        print $i
                    }
                }
            }'
    )"
    [[ -n "$dep" && "$dep" != "(null)" ]] || return 0
    printf "%s\n" "$dep" | grep -oE '[0-9]+' | awk '!seen[$0]++' | paste -sd, -
}

print_scontrol() {
    local job="$1"
    if ! have_cmd scontrol; then
        echo "scontrol not found on PATH"
        return 0
    fi
    scontrol show job -o "$job" 2>/dev/null \
        | tr ' ' '\n' \
        | egrep 'JobId=|JobName=|JobState=|Reason=|Dependency=|WorkDir=|Command=|StdOut=|StdErr=' \
        || true
}

find_manifest_hits() {
    local job="$1"
    local roots=()
    [[ -d logs/World_Model ]] && roots+=(logs/World_Model)
    [[ -d logs ]] && roots+=(logs)
    [[ "${#roots[@]}" -gt 0 ]] || return 0

    find "${roots[@]}" -name manifest.tsv -type f -print 2>/dev/null \
        | awk '!seen[$0]++' \
        | while IFS= read -r manifest; do
            grep -H -F "$job" "$manifest" || true
        done
}

find_logs_for_job() {
    local job="$1"
    local roots=()
    [[ -d logs/World_Model ]] && roots+=(logs/World_Model)
    [[ -d logs ]] && roots+=(logs)
    [[ "${#roots[@]}" -gt 0 ]] || return 0

    find "${roots[@]}" \( -type f -o -type l \) \( \
        -path "*/jobs/${job}/stdout" -o \
        -path "*/jobs/${job}/stderr" -o \
        -name "*_${job}.out" -o \
        -name "*_${job}.err" -o \
        -name "slurm-${job}.out" -o \
        -name "slurm-${job}.err" \
    \) -print 2>/dev/null | awk '!seen[$0]++'
}

find_run_dirs_for_job() {
    local job="$1"
    local roots=()
    [[ -d runs/World_Model ]] && roots+=(runs/World_Model)
    [[ -d runs/world_model ]] && roots+=(runs/world_model)
    [[ -d runs ]] && roots+=(runs)
    [[ "${#roots[@]}" -gt 0 ]] || return 0

    {
        find "${roots[@]}" -type d -name "*job${job}*" -print 2>/dev/null || true
        find "${roots[@]}" -name run_info.json -type f -print 2>/dev/null \
            | while IFS= read -r info; do
                if grep -F "\"job_id\": \"${job}\"" "$info" >/dev/null 2>&1 \
                    || grep -F "\"jobid\": \"${job}\"" "$info" >/dev/null 2>&1 \
                    || grep -F "job${job}" "$info" >/dev/null 2>&1; then
                    dirname "$info"
                fi
            done
    } | awk '!seen[$0]++'
}

print_run_artifacts() {
    local run_dir="$1"
    echo
    echo "----- artifacts in ${run_dir} -----"
    for rel in \
        run_info.json \
        report.md \
        comparison.csv \
        world_model_metrics.csv \
        baselines.csv \
        plots/loss_curves.png \
        plots/comparison.png \
        plots/scatter_world_model.png \
        plots/perpert_r2.png \
        plots/deg_overlap.png \
        train.log \
        cli_overrides.txt; do
        if [[ -e "${run_dir}/${rel}" ]]; then
            echo "${run_dir}/${rel}"
        fi
    done
    if [[ -d "${run_dir}/plots" ]]; then
        find "${run_dir}/plots" -maxdepth 1 -type f \( -name "*.png" -o -name "*.svg" -o -name "*.csv" \) -print 2>/dev/null \
            | sort | awk '!seen[$0]++'
    fi
    if [[ -f "${run_dir}/train.log" ]]; then
        echo
        echo "----- tail -n ${ARTIFACT_LINES} ${run_dir}/train.log -----"
        tail -n "$ARTIFACT_LINES" "${run_dir}/train.log" || true
    fi
}

tail_log() {
    local path="$1"
    [[ -e "$path" || -L "$path" ]] || return 0
    echo
    echo "----- tail -n ${TAIL_LINES} ${path} -----"
    tail -n "$TAIL_LINES" "$path" || true
}

for job in "$@"; do
    echo "================================================================================"
    echo "JOB ${job}"
    echo "================================================================================"

    echo
    echo "== sacct =="
    print_sacct "$job"

    echo
    echo "== scontrol =="
    print_scontrol "$job"

    deps="$(job_dependency_ids "$job")"
    if [[ -n "${deps:-}" ]]; then
        echo
        echo "== dependency sacct (${deps}) =="
        print_sacct "$deps"
    fi

    echo
    echo "== manifest hits =="
    find_manifest_hits "$job"

    echo
    echo "== log paths =="
    mapfile -t logs_for_job < <(find_logs_for_job "$job")
    if [[ "${#logs_for_job[@]}" -eq 0 ]]; then
        echo "No stdout/stderr files found under logs/ or logs/World_Model for job ${job}."
    else
        printf "%s\n" "${logs_for_job[@]}"

        for path in "${logs_for_job[@]}"; do
            case "$path" in
                *.err|*/stderr) tail_log "$path" ;;
            esac
        done
        for path in "${logs_for_job[@]}"; do
            case "$path" in
                *.out|*/stdout) tail_log "$path" ;;
            esac
        done
    fi

    echo
    echo "== run output directories =="
    mapfile -t run_dirs_for_job < <(find_run_dirs_for_job "$job")
    if [[ "${#run_dirs_for_job[@]}" -eq 0 ]]; then
        echo "No run directories found under runs/World_Model, runs/world_model, or runs for job ${job}."
        continue
    fi
    printf "%s\n" "${run_dirs_for_job[@]}"
    for run_dir in "${run_dirs_for_job[@]}"; do
        print_run_artifacts "$run_dir"
    done
done
