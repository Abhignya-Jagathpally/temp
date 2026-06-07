#!/bin/bash
# =============================================================================
# ResistanceMap v20 — Run Everything
# =============================================================================
# This script executes the full v20 pipeline on open-access data.
# Run from the ResistanceMap/ root directory.
#
# Prerequisites:
#   - Python 3.10+
#   - GPU recommended for scVI (CPU fallback works but slower)
#   - ~5GB disk space for downloads
#   - Internet connection for GDC API + GEO downloads
#
# Usage:
#   cd ResistanceMap
#   chmod +x scripts/run_all.sh
#   ./scripts/run_all.sh
# =============================================================================

set -e  # Exit on first error

echo "================================================================"
echo "ResistanceMap v20 — Full Pipeline Execution"
echo "================================================================"
echo ""

# --- 0. Environment Setup ---
echo "[0/6] Installing dependencies..."
pip install -q scikit-survival lifelines scvi-tools scanpy anndata requests pyarrow gdown 2>/dev/null || true
pip install -q torch torchdiffeq torch-geometric 2>/dev/null || true
echo "      Done."

# --- 1. Run Baselines on GDC Open Data ---
echo ""
echo "[1/6] Running censoring-aware baselines on GDC MMRF open-tier..."
python scripts/run_first_results.py
echo "      Baseline results saved to results/v20_first_baselines/"

# --- 2. Download GEO scRNA-seq ---
echo ""
echo "[2/6] Downloading GEO scRNA-seq datasets..."
python scripts/download_geo_scrna.py
echo "      scRNA-seq data saved to data/open_access/geo_scrna/"

# --- 3. Run scVI Integration ---
echo ""
echo "[3/6] Training scVI on merged MM scRNA-seq..."
python scripts/run_scvi_integration.py
echo "      scVI model + latent saved to checkpoints/scvi/"

# --- 4. Wire PK-SSM + Run Forward Pass ---
echo ""
echo "[4/6] Running PK-SSM forward pass on scVI latent..."
python scripts/run_pkssm_forward.py
echo "      PK-SSM results saved to results/v20_pkssm/"

# --- 5. Produce Final Results Table ---
echo ""
echo "[5/6] Generating combined results report..."
python scripts/generate_report.py
echo "      Report saved to results/v20_combined_report.md"

# --- 6. Summary ---
echo ""
echo "[6/6] Done."
echo "================================================================"
echo "Results:"
echo "  - Baselines:    results/v20_first_baselines/v20_first_baselines.md"
echo "  - scVI model:   checkpoints/scvi/model.pt"
echo "  - PK-SSM:       results/v20_pkssm/"
echo "  - Full report:  results/v20_combined_report.md"
echo "================================================================"