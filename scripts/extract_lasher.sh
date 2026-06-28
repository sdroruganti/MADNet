#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTS_DIR="$REPO_ROOT/data_sources"
OUT_DIR="${1:-$REPO_ROOT/data_sources}"

mkdir -p "$OUT_DIR"

echo "Repo root: $REPO_ROOT"
echo "Parts dir: $PARTS_DIR"
echo "Output dir: $OUT_DIR"
echo

expected_parts=(
  "$PARTS_DIR/lasher.tar.gz.part.aa"
  "$PARTS_DIR/lasher.tar.gz.part.ab"
  "$PARTS_DIR/lasher.tar.gz.part.ac"
  "$PARTS_DIR/lasher.tar.gz.part.ad"
  "$PARTS_DIR/lasher.tar.gz.part.ae"
)

for part in "${expected_parts[@]}"; do
  if [[ ! -f "$part" ]]; then
    echo "Missing file: $part"
    exit 1
  fi
done

echo "Found all LasHeR parts:"
ls -lh "${expected_parts[@]}"
echo

echo "Extracting LasHeR..."
echo "This streams the split tar.gz directly into tar without creating a 224GB intermediate file."
echo

cat "${expected_parts[@]}" | tar -xzvf - -C "$OUT_DIR"

echo
echo "Done. Extracted dataset to:"
echo "$OUT_DIR"

