#!/usr/bin/env bash
# Download DROID raw data and LVP caption metadata, then validate the result.
# Usage: bash scripts/download_droid.sh [--dest data/droid] [--check-only]

set -euo pipefail

DEST="data/droid"
CHECK_ONLY=0
PROCESS_COUNT="${PROCESS_COUNT:-8}"
THREAD_COUNT="${THREAD_COUNT:-16}"
MAX_COMPONENTS="${MAX_COMPONENTS:-4}"

usage() {
  cat <<'EOF'
Usage: bash scripts/download_droid.sh [options]

Options:
  --dest PATH       Output directory (default: data/droid)
  --check-only      Validate an existing download without downloading
  -h, --help        Show this help message

Environment overrides:
  PROCESS_COUNT     gsutil parallel processes (default: 8)
  THREAD_COUNT      gsutil threads per process (default: 16)
  MAX_COMPONENTS    sliced-download components (default: 4)

The raw download excludes SVO files and stereo MP4 files. Re-running the
command is safe: gsutil rsync and Hugging Face downloads are resumable.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      [[ $# -ge 2 ]] || { echo "error: --dest requires a path" >&2; exit 2; }
      DEST="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
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

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

command -v python >/dev/null 2>&1 || {
  echo "error: python is required" >&2
  exit 1
}

mkdir -p "$DEST"

if (( CHECK_ONLY == 0 )); then
  command -v gsutil >/dev/null 2>&1 || {
    echo "error: gsutil is required to download raw DROID data" >&2
    exit 1
  }
  command -v hf >/dev/null 2>&1 || {
    echo "error: the Hugging Face `hf` CLI is required for LVP metadata" >&2
    exit 1
  }

  log "Downloading raw DROID data to $DEST"
  gsutil -m \
    -o "GSUtil:parallel_process_count=${PROCESS_COUNT}" \
    -o "GSUtil:parallel_thread_count=${THREAD_COUNT}" \
    -o "GSUtil:sliced_object_download_threshold=150M" \
    -o "GSUtil:sliced_object_download_max_components=${MAX_COMPONENTS}" \
    rsync -r \
    -x ".*SVO.*|.*stereo.*\\.mp4$" \
    "gs://gresearch/robotics/droid_raw" \
    "$DEST"

  metadata_dir="$(mktemp -d)"
  trap 'rm -rf "$metadata_dir"' EXIT
  log "Downloading LVP cleaned metadata"
  hf download KempnerInstituteAI/LVP \
    --include "data/droid/cleaned_metadata.csv" \
    --local-dir "$metadata_dir"
  cp "$metadata_dir/data/droid/cleaned_metadata.csv" \
    "$DEST/cleaned_metadata.csv"
fi

log "Validating DROID download and LVP metadata"
DROID_DEST="$DEST" python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["DROID_DEST"])
version_root = root / "1.0.1"
metadata_path = root / "cleaned_metadata.csv"

if not version_root.is_dir():
    raise SystemExit(f"error: missing raw DROID directory: {version_root}")
if not metadata_path.is_file():
    raise SystemExit(f"error: missing LVP metadata: {metadata_path}")

required_columns = {
    "video_path",
    "fps",
    "n_frames",
    "width",
    "height",
    "gemini_caption",
}


def normalize_video_path(value: str) -> Path:
    normalized = value.replace("\\", "/").strip()
    for prefix in ("droid_raw/", "data/droid/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return Path(normalized)


def unsanitize_episode_timestamp(path: Path) -> Path:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if "_" not in part or ":" in part:
            continue
        pieces = part.rsplit("_", 3)
        if len(pieces) != 4:
            continue
        prefix, minute, second, year = pieces
        if (
            len(minute) == 2
            and minute.isdigit()
            and len(second) == 2
            and second.isdigit()
            and year.isdigit()
        ):
            parts[index] = f"{prefix}:{minute}:{second}_{year}"
    return Path(*parts)


missing = []
row_count = 0
with metadata_path.open(newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    columns = set(reader.fieldnames or [])
    absent_columns = sorted(required_columns - columns)
    if absent_columns:
        raise SystemExit(
            "error: cleaned_metadata.csv is missing columns: "
            + ", ".join(absent_columns)
        )
    for row in reader:
        row_count += 1
        relative = normalize_video_path(row["video_path"])
        candidates = (
            root / relative,
            root / unsanitize_episode_timestamp(relative),
        )
        if not any(candidate.is_file() for candidate in candidates):
            if len(missing) < 20:
                missing.append(row["video_path"])

if row_count == 0:
    raise SystemExit("error: cleaned_metadata.csv contains no rows")
if missing:
    print(f"error: at least {len(missing)} referenced videos are missing:")
    print(*missing, sep="\n")
    raise SystemExit(1)

print(f"DROID validation passed: {row_count} metadata rows")
print(f"Raw data: {version_root}")
print(f"Metadata: {metadata_path}")
PY

log "DROID data is ready under $DEST"
