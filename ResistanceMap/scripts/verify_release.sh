#!/usr/bin/env bash
# Verify v3-artifacts release end-to-end.
#
# 1. Concatenate multi-part archives.
# 2. Verify SHA256 of the CONCATENATED blob (not the parts).
# 3. Decompress + list archive contents.
#
# Usage:  bash scripts/verify_release.sh <release-dir>
set -euo pipefail

REL_DIR="${1:-release}"
cd "$REL_DIR"

if ! ls data_raw.tar.zst.part-* >/dev/null 2>&1; then
    echo "ERROR: no data_raw.tar.zst.part-* files in $REL_DIR" >&2
    exit 1
fi

echo "[1/4] Concatenating parts..."
cat data_raw.tar.zst.part-* > data_raw.tar.zst

echo "[2/4] Verifying SHA256SUMS (concatenated + checkpoints)..."
sha256sum -c SHA256SUMS

echo "[3/4] Decompressing..."
zstd -d --force data_raw.tar.zst -o data_raw.tar

echo "[4/4] Archive top-level contents:"
tar -tf data_raw.tar | head -20

echo
echo "OK. Manifest:"
cat manifest.json | python3 -m json.tool | head -40
