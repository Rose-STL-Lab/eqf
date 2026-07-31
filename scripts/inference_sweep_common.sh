#!/usr/bin/env bash

# Shared CLI and launch loop for the compact public inference scripts.
# The calling script defines JOB_NAMES/JOB_SHORTCODES/JOB_ALGORITHMS/JOB_TAGS,
# SHARED_OVERRIDES, and build_job_overrides(index, num_steps).

set -euo pipefail

print_inference_usage() {
  cat <<USAGE
Usage: bash scripts/${SCRIPT_NAME} [options]

${SCRIPT_DESCRIPTION}

Selection:
  --run-indices CSV       1-based indices and ranges, e.g. 1,3-5
  --list                  Print the method table and exit
                          With no selector, all methods run.

Options:
  --dry-run               Print commands without executing them
  --seed S[,S2,...]       Deterministic inference seed(s) (default ${SEEDS_CSV})
  --ckpt-map NAME         Hydra checkpoint map (default ${CKPT_MAP})
  --batch-size N          Validation batch size (default ${BATCH_SIZE})
  --devices N|auto        Trainer devices (default ${DEVICES})
  --num-sampling-steps CSV
                          Sampling-step value(s) (default ${NUM_SAMPLING_STEPS_CSV})
  --num-videos N          Number of validation videos (default ${NUM_VIDEOS})
  --wandb-mode MODE       W&B mode (default ${WANDB_MODE})
  --tag TAG               Append a W&B tag; repeatable
  --raw-dir-base PATH     Per-run raw output root (default: ${RAW_DIR_BASE:-disabled})
  --extra ARG ...         Append all remaining Hydra overrides
  -h, --help              Show this help
USAGE
}

parse_positive_int_csv() {
  local label="$1"
  local csv="$2"
  local out_name="$3"
  local -a raw=()
  local value seen="" count=0

  IFS=',' read -r -a raw <<< "$csv"
  for value in "${raw[@]}"; do
    value="${value//[[:space:]]/}"
    [[ -z "$value" ]] && continue
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
      echo "Invalid ${label} entry '${value}' (expected positive integer)" >&2
      exit 2
    fi
    if [[ ",${seen}," == *",${value},"* ]]; then
      echo "Duplicate ${label} entry '${value}'" >&2
      exit 2
    fi
    seen="${seen:+${seen},}${value}"
    eval "${out_name}+=(\"${value}\")"
    count=$((count + 1))
  done

  if [[ "$count" -eq 0 ]]; then
    echo "No values supplied for ${label}" >&2
    exit 2
  fi
}

select_run_indices() {
  local csv="$1"
  local -a entries=()
  local entry start stop idx seen=""

  IFS=',' read -r -a entries <<< "$csv"
  for entry in "${entries[@]}"; do
    entry="${entry//[[:space:]]/}"
    [[ -z "$entry" ]] && continue
    if [[ "$entry" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      stop="${BASH_REMATCH[2]}"
      if [[ "$start" -gt "$stop" ]]; then
        echo "Invalid descending run range '${entry}'" >&2
        exit 2
      fi
    elif [[ "$entry" =~ ^[0-9]+$ ]]; then
      start="$entry"
      stop="$entry"
    else
      echo "Invalid run index '${entry}'" >&2
      exit 2
    fi

    for ((idx=start; idx<=stop; idx++)); do
      if [[ "$idx" -lt 1 ]] || [[ "$idx" -gt "${#JOB_NAMES[@]}" ]]; then
        echo "Run index ${idx} is outside 1..${#JOB_NAMES[@]}" >&2
        exit 2
      fi
      if [[ ",${seen}," == *",${idx},"* ]]; then
        echo "Duplicate run index ${idx}" >&2
        exit 2
      fi
      seen="${seen:+${seen},}${idx}"
      SELECTED_INDICES+=("$idx")
    done
  done

  if [[ "${#SELECTED_INDICES[@]}" -eq 0 ]]; then
    echo "No run indices selected" >&2
    exit 2
  fi
}

print_shell_command() {
  local arg
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_inference_sweep() {
  local dry_run=0
  local mode="all"
  local indices_csv=""
  local -a extra_tags=()
  local -a extra_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        dry_run=1
        shift
        ;;
      --seed)
        SEEDS_CSV="${2:-}"
        shift 2
        ;;
      --ckpt-map)
        CKPT_MAP="${2:-}"
        shift 2
        ;;
      --batch-size)
        BATCH_SIZE="${2:-}"
        shift 2
        ;;
      --devices)
        DEVICES="${2:-}"
        shift 2
        ;;
      --num-sampling-steps)
        NUM_SAMPLING_STEPS_CSV="${2:-}"
        shift 2
        ;;
      --num-videos)
        NUM_VIDEOS="${2:-}"
        shift 2
        ;;
      --wandb-mode)
        WANDB_MODE="${2:-}"
        shift 2
        ;;
      --tag)
        extra_tags+=("${2:-}")
        shift 2
        ;;
      --raw-dir-base)
        RAW_DIR_BASE="${2:-}"
        shift 2
        ;;
      --run-indices)
        if [[ "$mode" != "all" ]]; then
          echo "Only one selector may be supplied" >&2
          exit 2
        fi
        mode="indices"
        indices_csv="${2:-}"
        shift 2
        ;;
      --list)
        if [[ "$mode" != "all" ]]; then
          echo "Only one selector may be supplied" >&2
          exit 2
        fi
        mode="list"
        shift
        ;;
      --extra)
        shift
        while [[ $# -gt 0 ]]; do
          extra_args+=("$1")
          shift
        done
        ;;
      -h|--help)
        print_inference_usage
        return 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        print_inference_usage >&2
        exit 2
        ;;
    esac
  done

  if [[ "${#JOB_NAMES[@]}" -eq 0 ]] ||
     [[ "${#JOB_NAMES[@]}" -ne "${#JOB_SHORTCODES[@]}" ]] ||
     [[ "${#JOB_NAMES[@]}" -ne "${#JOB_ALGORITHMS[@]}" ]] ||
     [[ "${#JOB_NAMES[@]}" -ne "${#JOB_TAGS[@]}" ]]; then
    echo "Internal error: inference job arrays are inconsistent" >&2
    exit 3
  fi

  if [[ "$mode" == "list" ]]; then
    printf '%-4s  %-32s  %-42s  %s\n' "idx" "method" "shortcode" "default steps"
    local list_idx
    for ((list_idx=0; list_idx<${#JOB_NAMES[@]}; list_idx++)); do
      printf '%-4d  %-32s  %-42s  %s\n' \
        "$((list_idx + 1))" "${JOB_NAMES[$list_idx]}" \
        "${JOB_SHORTCODES[$list_idx]}" "${NUM_SAMPLING_STEPS_CSV}"
    done
    return 0
  fi

  if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -lt 1 ]]; then
    echo "Invalid --batch-size '${BATCH_SIZE}'" >&2
    exit 2
  fi
  if [[ "$DEVICES" != "auto" ]] &&
     { ! [[ "$DEVICES" =~ ^[0-9]+$ ]] || [[ "$DEVICES" -lt 1 ]]; }; then
    echo "Invalid --devices '${DEVICES}'" >&2
    exit 2
  fi
  if ! [[ "$NUM_VIDEOS" =~ ^[0-9]+$ ]] || [[ "$NUM_VIDEOS" -lt 1 ]]; then
    echo "Invalid --num-videos '${NUM_VIDEOS}'" >&2
    exit 2
  fi
  if [[ -z "$CKPT_MAP" ]] || [[ -z "$WANDB_MODE" ]]; then
    echo "Checkpoint map and W&B mode must be non-empty" >&2
    exit 2
  fi

  local tag
  for tag in "${extra_tags[@]}"; do
    if ! [[ "$tag" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
      echo "Invalid --tag '${tag}'" >&2
      exit 2
    fi
  done

  local -a seeds=()
  local -a sampling_steps=()
  parse_positive_int_csv "--seed" "$SEEDS_CSV" seeds
  parse_positive_int_csv "--num-sampling-steps" "$NUM_SAMPLING_STEPS_CSV" sampling_steps

  SELECTED_INDICES=()
  if [[ "$mode" == "indices" ]]; then
    select_run_indices "$indices_csv"
  else
    local all_idx
    for ((all_idx=1; all_idx<=${#JOB_NAMES[@]}; all_idx++)); do
      SELECTED_INDICES+=("$all_idx")
    done
  fi

  local total=$(( ${#seeds[@]} * ${#SELECTED_INDICES[@]} * ${#sampling_steps[@]} ))
  echo "[inference] script=${SCRIPT_NAME} methods=(${SELECTED_INDICES[*]}) steps=(${sampling_steps[*]}) seeds=(${seeds[*]}) videos=${NUM_VIDEOS} ckpt_map=${CKPT_MAP} total=${total} dry_run=${dry_run}"

  local seed one_based zero_based nsteps run_name tags_json extra_tag
  local -a cmd=()
  local launched=0
  for seed in "${seeds[@]}"; do
    for one_based in "${SELECTED_INDICES[@]}"; do
      zero_based=$((one_based - 1))
      for nsteps in "${sampling_steps[@]}"; do
        printf -v run_name '%s_%02d_%s_ns%s' \
          "$RUN_PREFIX" "$one_based" "${JOB_NAMES[$zero_based]}" "$nsteps"
        tags_json="[\"${BASE_TAG}\",\"${JOB_TAGS[$zero_based]}\",\"nsteps_${nsteps}\""
        for extra_tag in "${extra_tags[@]}"; do
          tags_json+=",\"${extra_tag}\""
        done
        tags_json+="]"

        JOB_OVERRIDES=()
        build_job_overrides "$one_based" "$nsteps"

        cmd=(
          python -m main
          "shortcode=${JOB_SHORTCODES[$zero_based]}"
          "algorithm=${JOB_ALGORITHMS[$zero_based]}"
          "+name=${run_name}"
        )
        cmd+=("${SHARED_OVERRIDES[@]}")
        cmd+=("ckpt_map=${CKPT_MAP}")
        cmd+=("wandb.mode=${WANDB_MODE}")
        cmd+=("experiment.devices=${DEVICES}")
        cmd+=("experiment.validation.batch_size=${BATCH_SIZE}")
        cmd+=("dataset.num_validation_clips=${NUM_VIDEOS}")
        if [[ "${SYNC_LOG_MAX_VIDEOS:-0}" -eq 1 ]]; then
          cmd+=("algorithm.logging.max_num_videos=${NUM_VIDEOS}")
        fi
        cmd+=("algorithm.logging.deterministic=${seed}")
        cmd+=("${JOB_OVERRIDES[@]}")
        cmd+=("algorithm.denoising.num_sampling_steps=${nsteps}")
        if [[ -n "${RAW_DIR_BASE:-}" ]]; then
          cmd+=("algorithm.logging.raw_dir=${RAW_DIR_BASE}/${run_name}")
        fi
        cmd+=("+tags=${tags_json}")
        cmd+=("${extra_args[@]}")

        launched=$((launched + 1))
        echo "[run ${launched}/${total}] method=${one_based} seed=${seed} steps=${nsteps} ${JOB_NAMES[$zero_based]}"
        if [[ "$dry_run" -eq 1 ]]; then
          print_shell_command "${cmd[@]}"
        else
          "${cmd[@]}"
        fi
      done
    done
  done
}
