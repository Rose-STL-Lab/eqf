#!/usr/bin/env bash
# Download and extract RE10K datasets from Hugging Face.
# Usage: bash scripts/download_re10k.sh
# Optional: set DEST=/path/to/re10k, WORKERS=N, and CLEAN=1.

set -euo pipefail

DEST="${DEST:-data/re10k}"
BASE="https://huggingface.co/kiwhansong/DFoT/resolve/main/datasets"

# Optional original camera-metadata archives. EqF uses the prepared videos and
# poses from RealEstate10K_Full, so these are not downloaded by default.
SINGLE_FILES=(
  # "RealEstate10K_Tiny.tar.gz"
  # "RealEstate10K_Mini.tar.gz"
  # "RealEstate10K.tar.gz"
)

# Split parts for the "Full" archive
FULL_PARTS=(
  "RealEstate10K_Full.tar.gz.part-aa"
  "RealEstate10K_Full.tar.gz.part-ab"
  "RealEstate10K_Full.tar.gz.part-ac"
)

DOWNLOAD_FILES=("${SINGLE_FILES[@]}" "${FULL_PARTS[@]}")
WORKERS="${WORKERS:-${#DOWNLOAD_FILES[@]}}"

[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: WORKERS must be a positive integer" >&2
  exit 2
}

mkdir -p "$DEST"

log() { printf "\n[%s] %s\n" "$(date '+%F %T')" "$*" >&2; }

download_one() {
  local fname="$1"
  local url="${BASE}/${fname}?download=true"
  local out="${DEST}/${fname}"

  log "Downloading ${fname} ..."
  # -c resume; retries for flaky links; robust timeouts
  wget -c "$url" \
    -O "$out" \
    --retry-connrefused --waitretry=5 --tries=20 \
    --read-timeout=60 --timeout=60
}

extract_tar() {
  local tarpath="$1"
  local outdir="$2"
  mkdir -p "$outdir"
  log "Extracting $(basename "$tarpath") -> $outdir"

  if command -v bsdtar >/dev/null 2>&1; then
    # Avoid xattrs/ACLs/flags on filesystems that don't support them
    # Also filter the noisy "Cannot restore extended attributes" warnings if they still appear.
    bsdtar --no-xattrs --no-acls --no-fflags -xpf "$tarpath" -C "$outdir" \
      2> >(grep -v "Cannot restore extended attributes" >&2)
  else
    # GNU tar: explicitly disable xattrs/ACLs to avoid warnings
    tar --no-xattrs --no-acls -xzf "$tarpath" -C "$outdir"
  fi
}



# 1) Download all archives and split parts concurrently
log "Downloading ${#DOWNLOAD_FILES[@]} files with up to $WORKERS workers"
active=0
failed=0
for f in "${DOWNLOAD_FILES[@]}"; do
  download_one "$f" &
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

# 2) Recombine the "Full" parts -> RealEstate10K_Full.tar.gz
FULL_TAR="${DEST}/RealEstate10K_Full.tar.gz"
if [[ ! -f "$FULL_TAR" ]]; then
  log "Recombining full archive parts -> $(basename "$FULL_TAR")"
  cat "${DEST}/RealEstate10K_Full.tar.gz.part-"* > "$FULL_TAR"
else
  log "Combined full archive already exists: $(basename "$FULL_TAR") (skipping recombine)"
fi

# 3) Extract all archives into their own folder under $DEST
#    Folder names are derived from the archive base name (without .tar.gz)
for f in "${SINGLE_FILES[@]}"; do
  base="${f%.tar.gz}"
  extract_tar "${DEST}/${f}" "${DEST}/${base}"
done

extract_tar "$FULL_TAR" "${DEST}/RealEstate10K_Full"

# 4) Optional cleanup of .tar.gz files if CLEAN=1
if [[ "${CLEAN:-0}" == "1" ]]; then
  log "CLEAN=1 set; removing .tar.gz files after successful extraction."
  rm -f "${DEST}/"*.tar.gz
  # Keep the split parts by default; uncomment to remove:
  # rm -f "${DEST}/RealEstate10K_Full.tar.gz.part-"*
fi

log "All done! Files are in: $DEST"
