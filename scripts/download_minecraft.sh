#!/usr/bin/env bash
# Download and extract the Minecraft MARL dataset from Internet Archive.
# Usage: bash scripts/download_minecraft.sh [--dest data/minecraft] [--workers 11]

set -euo pipefail

DEST="data/minecraft"
WORKERS=11
KEEP_ARCHIVES=0

usage() {
  cat <<'EOF'
Usage: bash scripts/download_minecraft.sh [options]

Options:
  --dest PATH       Output directory (default: data/minecraft)
  --workers N       Concurrent part downloads (default: 11)
  --keep-archives   Keep downloaded archive parts after extraction
  -h, --help        Show this help message

Downloads are resumable. Re-run the same command after an interruption.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      [[ $# -ge 2 ]] || { echo "error: --dest requires a path" >&2; exit 2; }
      DEST="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "error: --workers requires a number" >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;
    --keep-archives)
      KEEP_ARCHIVES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --workers must be a positive integer" >&2
  exit 2
}

command -v wget >/dev/null 2>&1 || {
  echo "error: wget is required" >&2
  exit 1
}
command -v tar >/dev/null 2>&1 || {
  echo "error: tar is required" >&2
  exit 1
}

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

if [[ -d "$DEST/training" && -d "$DEST/validation" ]]; then
  log "Minecraft is already prepared under $DEST; nothing to do."
  exit 0
fi
if [[ -e "$DEST/training" || -e "$DEST/validation" ]]; then
  echo "error: partial prepared dataset found under $DEST" >&2
  echo "Move or remove the existing training/validation directory, then retry." >&2
  exit 1
fi

PARTS_DIR="$DEST/.download"
EXTRACT_DIR="$DEST/.extract"
SUFFIXES=(aa ab ac ad ae af ag ah ai aj ak)

mkdir -p "$PARTS_DIR"

download_part() {
  local suffix="$1"
  local identifier="minecraft_marsh_dataset_${suffix}"
  local filename="minecraft.tar.part${suffix}"
  local url="https://archive.org/download/${identifier}/${filename}"
  local output="${PARTS_DIR}/${filename}"
  local expected_size=21474836480
  if [[ "$suffix" == "ak" ]]; then
    expected_size=10434549760
  fi

  if [[ -f "$output" ]]; then
    local actual_size
    actual_size="$(stat -c '%s' "$output")"
    if [[ "$actual_size" == "$expected_size" ]]; then
      log "${filename} is already complete; skipping"
      return 0
    fi
    if (( actual_size > expected_size )); then
      echo "error: ${output} is larger than expected (${actual_size} > ${expected_size})" >&2
      return 1
    fi
  fi

  log "Downloading ${filename}"
  wget -c "$url" -O "$output" \
    --retry-connrefused --waitretry=5 --tries=20 \
    --read-timeout=60 --timeout=60

  local actual_size
  actual_size="$(stat -c '%s' "$output")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "error: ${output} has size ${actual_size}; expected ${expected_size}" >&2
    return 1
  fi
}

log "Downloading ${#SUFFIXES[@]} parts with up to $WORKERS workers"
active=0
failed=0
for suffix in "${SUFFIXES[@]}"; do
  download_part "$suffix" &
  ((active += 1))
  if (( active >= WORKERS )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done
if (( failed != 0 )); then
  echo "error: one or more downloads failed; re-run to resume" >&2
  exit 1
fi

archive_parts=()
for suffix in "${SUFFIXES[@]}"; do
  part="$PARTS_DIR/minecraft.tar.part${suffix}"
  [[ -s "$part" ]] || {
    echo "error: missing or empty downloaded part: $part" >&2
    exit 1
  }
  archive_parts+=("$part")
done

# Older versions of this script created a redundant combined 210 GiB tar.
if [[ -f "$PARTS_DIR/minecraft.tar" || -f "$PARTS_DIR/minecraft.tar.tmp" ]]; then
  log "Removing obsolete combined archive"
  rm -f "$PARTS_DIR/minecraft.tar" "$PARTS_DIR/minecraft.tar.tmp"
fi

rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
log "Streaming archive parts directly into tar"
if command -v pv >/dev/null 2>&1; then
  archive_size=0
  for part in "${archive_parts[@]}"; do
    archive_size=$((archive_size + $(stat -c '%s' "$part")))
  done
  pv --size "$archive_size" "${archive_parts[@]}" | tar -xf - -C "$EXTRACT_DIR"
else
  cat "${archive_parts[@]}" | tar -xf - -C "$EXTRACT_DIR"
fi

TRAIN_DIR="$EXTRACT_DIR/minecraft/train"
TEST_DIR="$EXTRACT_DIR/minecraft/test"
[[ -d "$TRAIN_DIR" && -d "$TEST_DIR" ]] || {
  echo "error: archive did not contain minecraft/train and minecraft/test" >&2
  exit 1
}

mkdir -p "$DEST"
mv "$TRAIN_DIR" "$DEST/training"
mv "$TEST_DIR" "$DEST/validation"
rm -rf "$EXTRACT_DIR"

if (( KEEP_ARCHIVES == 0 )); then
  rm -rf "$PARTS_DIR"
else
  log "Keeping downloaded archives under $PARTS_DIR"
fi

log "Minecraft is ready under $DEST"
