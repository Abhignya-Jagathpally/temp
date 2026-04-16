#!/usr/bin/env bash
# One-command reproducer for the headline numbers.
#
# CI runs this nightly on a public subsample. If wall-time or headline metrics
# drift outside the documented CI band, the build fails.
set -euo pipefail

RELEASE_DIR="${RELEASE_DIR:-release}"
OUT_DIR="${OUT_DIR:-reports/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

echo "=== Verifying release artifacts ==="
bash scripts/verify_release.sh "$RELEASE_DIR"

echo "=== Running ResistanceMap reproducer ==="
python -m resistancemap.reproduce \
    --checkpoints "$RELEASE_DIR/checkpoints" \
    --data        "$RELEASE_DIR/data_raw" \
    --split       patient \
    --folds       5 \
    --seeds       0 1 2 3 4 \
    --baselines   global_mean per_drug_mean ridge_esm2 \
    --targets     resistance_binary log_ic50 \
    --out         "$OUT_DIR/headline_table.md"

echo "=== Checking against documented CI band ==="
python scripts/assert_reproduction_band.py \
    --report "$OUT_DIR/headline_table.md" \
    --band   configs/reproduction_band.yaml

echo
echo "Headline table: $OUT_DIR/headline_table.md"
