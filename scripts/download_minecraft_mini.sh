#!/usr/bin/env bash
# Download the Minecraft mini quickstart dataset and its two required checkpoints.

set -euo pipefail

REPO="${EQF_REPO:-hlillemark/eqf}"
DEST="data/minecraft/mini"
CHECKPOINT_ROOT="${EQF_CHECKPOINT_ROOT:-./downloaded_checkpoints/shared}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/download_minecraft_mini.sh [options]

Options:
  --dest PATH             Mini dataset root (default: data/minecraft/mini)
  --checkpoint-root PATH  Checkpoint root (default: ./downloaded_checkpoints/shared)
  --dry-run               Print download commands without running them
  -h, --help              Show this help

Downloads 256 paired MP4/NPZ validation clips plus the Minecraft ImageVAE and
EqF checkpoints. The resulting paths work with `ckpt_map=default`.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            [[ $# -ge 2 ]] || { echo "ERROR: --dest requires a path." >&2; exit 2; }
            DEST="$2"
            shift 2
            ;;
        --checkpoint-root)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --checkpoint-root requires a path." >&2
                exit 2
            }
            CHECKPOINT_ROOT="$2"
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

run_command() {
    if (( DRY_RUN )); then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if (( ! DRY_RUN )); then
    command -v hf >/dev/null 2>&1 || {
        echo "ERROR: Hugging Face CLI command 'hf' was not found." >&2
        echo "Activate the EqF environment or install huggingface_hub." >&2
        exit 1
    }
fi

staging_dir=""
cleanup() {
    if [[ -n "$staging_dir" && -d "$staging_dir" ]]; then
        rm -rf "$staging_dir"
    fi
}
trap cleanup EXIT

if (( DRY_RUN )); then
    staging_dir="<temporary-directory>"
else
    staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/minecraft-mini.XXXXXX")"
fi

echo "Downloading Minecraft mini data from ${REPO}"
run_command hf download \
    "$REPO" \
    --include "minecraft_mini/**" \
    --local-dir "$staging_dir"

echo "Downloading the ImageVAE and EqF checkpoints"
run_command hf download \
    "$REPO" \
    "minecraft/vae.ckpt" \
    "minecraft/eqf_noiselevelpred.ckpt" \
    --local-dir "$CHECKPOINT_ROOT"

if (( DRY_RUN )); then
    echo "Dry run complete; no files were downloaded."
    exit 0
fi

source_dir="$staging_dir/minecraft_mini/validation/0"
[[ -d "$source_dir" ]] || {
    echo "ERROR: Download did not contain minecraft_mini/validation/0." >&2
    exit 1
}

shopt -s nullglob
mp4_files=("$source_dir"/*.mp4)
npz_files=("$source_dir"/*.npz)
if (( ${#mp4_files[@]} != 256 || ${#npz_files[@]} != 256 )); then
    echo "ERROR: Expected 256 MP4 and 256 NPZ files, found" \
        "${#mp4_files[@]} MP4 and ${#npz_files[@]} NPZ." >&2
    exit 1
fi

for (( index = 0; index < 256; index++ )); do
    stem="$(printf '%06d' "$index")"
    [[ -s "$source_dir/$stem.mp4" && -s "$source_dir/$stem.npz" ]] || {
        echo "ERROR: Missing or empty pair for Minecraft clip $stem." >&2
        exit 1
    }
done

mkdir -p "$DEST/validation"
rm -rf "$DEST/validation/0"
cp -a "$source_dir" "$DEST/validation/0"
rm -f "$DEST/metadata/validation.pt"

echo
echo "Minecraft mini is ready under $DEST."
echo "Checkpoints are ready under $CHECKPOINT_ROOT."
if [[ "$CHECKPOINT_ROOT" != "./downloaded_checkpoints/shared" ]]; then
    printf 'Use the custom checkpoint root with:\n  export EQF_CHECKPOINT_ROOT=%q\n' \
        "$CHECKPOINT_ROOT"
fi
