#!/usr/bin/env bash
# Download released EqF checkpoints and their official Wan dependencies.
#
# Examples:
#   bash scripts/download_checkpoints.sh all
#   bash scripts/download_checkpoints.sh minecraft
#   bash scripts/download_checkpoints.sh re10k droid
#   bash scripts/download_checkpoints.sh --dry-run --root /path/to/checkpoints all

set -euo pipefail

# Huggingface repo:
EQF_REPO="${EQF_REPO:-hlillemark/eqf}"
ROOT="${ROOT:-./downloaded_checkpoints/shared}"
DRY_RUN=0
SELECT_MINECRAFT=0
SELECT_RE10K=0
SELECT_DROID=0

usage() {
    cat <<'EOF'
Usage: bash scripts/download_checkpoints.sh [options] <dataset> [dataset ...]

Datasets:
  all          Download Minecraft, Re10K, and DROID checkpoints.
  minecraft    Download released Minecraft checkpoints and ImageVAE.
  re10k        Download Re10K checkpoints, null prompt, and Wan2.1 dependencies.
  droid        Download DROID checkpoints, null prompt, and Wan2.2 dependencies.

Options:
  --root PATH  Destination root (default: ./downloaded_checkpoints/shared).
  --dry-run    Print the Hugging Face commands without downloading.
  -h, --help   Show this help.

The resulting tree is compatible with `ckpt_map=default`. When using a
custom --root, set EQF_CHECKPOINT_ROOT to the same path during inference.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        all)
            SELECT_MINECRAFT=1
            SELECT_RE10K=1
            SELECT_DROID=1
            shift
            ;;
        minecraft)
            SELECT_MINECRAFT=1
            shift
            ;;
        re10k)
            SELECT_RE10K=1
            shift
            ;;
        droid)
            SELECT_DROID=1
            shift
            ;;
        --root)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --root requires a path." >&2
                exit 2
            }
            ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( ! SELECT_MINECRAFT && ! SELECT_RE10K && ! SELECT_DROID )); then
    echo "ERROR: Select at least one of: all, minecraft, re10k, droid." >&2
    usage >&2
    exit 2
fi

if (( ! DRY_RUN )); then
    command -v hf >/dev/null 2>&1 || {
        echo "ERROR: Hugging Face CLI command 'hf' was not found." >&2
        echo "Activate the EqF environment or install huggingface_hub." >&2
        exit 1
    }
fi

run_command() {
    if (( DRY_RUN )); then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

download_files() {
    local repo="$1"
    local destination="$2"
    shift 2

    echo
    echo "Downloading from ${repo} to ${destination}"
    run_command hf download "$repo" "$@" --local-dir "$destination"
}

declare -a EQF_FILES=()
declare -A EQF_FILE_SEEN=()

add_eqf_file() {
    local path="$1"
    if [[ -z "${EQF_FILE_SEEN[$path]:-}" ]]; then
        EQF_FILES+=("$path")
        EQF_FILE_SEEN["$path"]=1
    fi
}

if (( SELECT_MINECRAFT )); then
    add_eqf_file "minecraft/dfot.ckpt"
    add_eqf_file "minecraft/flowf.ckpt"
    add_eqf_file "minecraft/eqm.ckpt"
    add_eqf_file "minecraft/eqf_noiselevelpred.ckpt"
    add_eqf_file "minecraft/vae.ckpt"
fi

if (( SELECT_RE10K )); then
    add_eqf_file "re10k/flowf.ckpt"
    add_eqf_file "re10k/eqf_noiselevelpred.ckpt"
    add_eqf_file "wan/null_caption_t5.pth"
fi

if (( SELECT_DROID )); then
    add_eqf_file "droid/flowf.ckpt"
    add_eqf_file "droid/eqf_noiselevelpred.ckpt"
    add_eqf_file "wan/null_caption_t5.pth"
fi

download_files "$EQF_REPO" "$ROOT" "${EQF_FILES[@]}"

if (( SELECT_RE10K )); then
    WAN21_FILES=(
        "config.json"
        "diffusion_pytorch_model.safetensors"
        "Wan2.1_VAE.pth"
        "models_t5_umt5-xxl-enc-bf16.pth"
        "google/umt5-xxl/special_tokens_map.json"
        "google/umt5-xxl/spiece.model"
        "google/umt5-xxl/tokenizer.json"
        "google/umt5-xxl/tokenizer_config.json"
    )
    download_files \
        "Wan-AI/Wan2.1-T2V-1.3B" \
        "$ROOT/wan/Wan2.1-T2V-1.3B" \
        "${WAN21_FILES[@]}"
fi

if (( SELECT_DROID )); then
    WAN22_FILES=(
        "config.json"
        "configuration.json"
        "diffusion_pytorch_model.safetensors.index.json"
        "diffusion_pytorch_model-00001-of-00003.safetensors"
        "diffusion_pytorch_model-00002-of-00003.safetensors"
        "diffusion_pytorch_model-00003-of-00003.safetensors"
        "Wan2.2_VAE.pth"
        "models_t5_umt5-xxl-enc-bf16.pth"
        "google/umt5-xxl/special_tokens_map.json"
        "google/umt5-xxl/spiece.model"
        "google/umt5-xxl/tokenizer.json"
        "google/umt5-xxl/tokenizer_config.json"
    )
    download_files \
        "Wan-AI/Wan2.2-TI2V-5B" \
        "$ROOT/wan/Wan2.2-TI2V-5B" \
        "${WAN22_FILES[@]}"
fi

echo
if (( DRY_RUN )); then
    echo "Dry run complete; no files were downloaded."
else
    echo "Checkpoint download complete. Use ckpt_map=default."
fi
if [[ "$ROOT" != "./downloaded_checkpoints/shared" ]]; then
    printf 'Custom root selected; run inference with:\n  export EQF_CHECKPOINT_ROOT=%q\n' "$ROOT"
fi
