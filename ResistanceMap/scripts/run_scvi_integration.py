#!/usr/bin/env python3
"""
ResistanceMap v20 — scVI Integration
=====================================
Trains scVI on the downloaded MM scRNA-seq data, producing:
  1. Batch-corrected latent representations (z ∈ R^30)
  2. Chromatin reader/writer gene expression for ChromatinODE
  3. Biomarker proxy gene expression for PK observation model

Requires: scvi-tools, scanpy, anndata
Input: data/open_access/geo_scrna/panImmune.h5ad (or GSE161801/GSE189460)
Output: checkpoints/scvi/model.pt, data/processed/scvi_latent.npy
"""

import os
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/open_access/geo_scrna")
PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints/scvi")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Chromatin reader/writer genes (same 20 as v19's ChromatinODE)
CHROMATIN_GENES = [
    'EZH2', 'KDM6A', 'KDM6B', 'KMT2A', 'KMT2D',
    'DNMT1', 'DNMT3A', 'DNMT3B', 'TET1', 'TET2',
    'HDAC1', 'HDAC2', 'KAT2A', 'KAT2B', 'EP300',
    'BRD4', 'SMARCA4', 'ARID1A', 'SUZ12', 'EED',
]

# Biomarker proxy genes (for PK observation model)
BIOMARKER_PROXY_GENES = [
    'IGHG1', 'IGHG2', 'IGHG3', 'IGHG4',  # IgG heavy chains → M-protein
    'IGHA1', 'IGHA2',                        # IgA heavy chains
    'IGHM',                                   # IgM heavy chain
    'IGKC',                                   # Kappa constant → FLC-κ
    'IGLC1', 'IGLC2', 'IGLC3', 'IGLC7',     # Lambda constants → FLC-λ
    'B2M',                                    # Beta-2 microglobulin
    'ALB',                                    # Albumin
    'LDHA', 'LDHB',                           # LDH
    'TNFRSF17',                               # BCMA → sBCMA proxy
    'CD38',                                   # CD38 surface
    'SDC1',                                   # CD138 (syndecan-1) → tumor burden
]

# MM-relevant cell type markers for scANVI
MM_MARKERS = {
    'Plasma_cells': ['SDC1', 'TNFRSF17', 'XBP1', 'IRF4', 'PRDM1'],
    'T_cells': ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B'],
    'NK_cells': ['NCAM1', 'FCGR3A', 'NKG7', 'GNLY'],
    'Monocytes': ['CD14', 'FCGR3A', 'CST3'],
    'B_cells': ['CD19', 'MS4A1', 'CD79A'],
}


def load_atlas() -> ad.AnnData:
    """Load the Zenodo panImmune atlas (primary) or fall back to GSE data."""
    atlas_path = DATA_DIR / "panImmune.h5ad"

    if atlas_path.exists():
        logger.info(f"Loading Zenodo MM atlas: {atlas_path}")
        adata = sc.read_h5ad(atlas_path)
        logger.info(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata

    # Fallback: try to build from GSE files
    logger.warning("panImmune.h5ad not found. Attempting to build from GSE data...")

    gse_dirs = list(DATA_DIR.glob("GSE*"))
    if not gse_dirs:
        raise FileNotFoundError(
            f"No scRNA-seq data found in {DATA_DIR}. "
            "Run download_geo_scrna.py first."
        )

    # Load whatever h5/mtx files we find
    adatas = []
    for gse_dir in gse_dirs:
        for h5_file in gse_dir.rglob("*.h5"):
            try:
                adata = sc.read_10x_h5(h5_file)
                adata.obs["dataset"] = gse_dir.name
                adata.var_names_make_unique()
                adatas.append(adata)
                logger.info(f"  Loaded {h5_file.name}: {adata.n_obs} cells")
            except Exception as e:
                logger.warning(f"  Failed to load {h5_file}: {e}")

    if not adatas:
        raise FileNotFoundError("No loadable h5 files found in GEO directories.")

    adata = ad.concat(adatas, label="dataset", join="outer")
    logger.info(f"  Merged: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    """Standard scRNA-seq preprocessing for scVI."""
    logger.info("Preprocessing...")

    # Store raw counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Filter
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)

    # Highly variable genes (keep all chromatin + biomarker genes)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=3000, flavor="seurat_v3",
        layer="counts", batch_key="dataset" if "dataset" in adata.obs else None,
    )

    # Force-include our genes of interest
    genes_of_interest = set(CHROMATIN_GENES + BIOMARKER_PROXY_GENES)
    for gene in genes_of_interest:
        if gene in adata.var_names:
            adata.var.loc[gene, "highly_variable"] = True

    logger.info(f"  After filtering: {adata.n_obs} cells, "
                f"{adata.var['highly_variable'].sum()} HVGs (incl. forced)")

    return adata


def train_scvi(adata: ad.AnnData) -> "scvi.model.SCVI":
    """Train scVI for batch-corrected latent space."""
    import scvi

    logger.info("Setting up scVI...")

    # Subset to HVGs
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # Setup
    batch_key = "dataset" if "dataset" in adata_hvg.obs else None
    scvi.model.SCVI.setup_anndata(
        adata_hvg, layer="counts", batch_key=batch_key
    )

    # Train
    model = scvi.model.SCVI(
        adata_hvg,
        n_latent=30,
        n_layers=2,
        n_hidden=256,
        gene_likelihood="zinb",
    )

    logger.info("Training scVI (this may take 10-30 min on GPU, longer on CPU)...")
    model.train(
        max_epochs=400,
        early_stopping=True,
        early_stopping_patience=20,
        train_size=0.9,
        batch_size=256,
    )

    logger.info(f"  Training complete. Final ELBO: {model.history['elbo_train'].iloc[-1]:.2f}")

    # Save
    model_path = str(CHECKPOINT_DIR / "scvi_model")
    model.save(model_path, overwrite=True)
    logger.info(f"  Model saved to: {model_path}")

    return model


def extract_representations(model, adata: ad.AnnData):
    """Extract latent z, chromatin expression, and biomarker proxies."""
    import scvi

    logger.info("Extracting representations...")

    # Subset to HVGs (same as training)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # 1. Latent representations
    z = model.get_latent_representation(adata_hvg)
    np.save(PROCESSED_DIR / "scvi_latent_z.npy", z)
    logger.info(f"  Latent z: {z.shape} saved to scvi_latent_z.npy")

    # 2. Chromatin reader/writer expression (denoised)
    available_chromatin = [g for g in CHROMATIN_GENES if g in adata_hvg.var_names]
    if available_chromatin:
        chromatin_expr = model.get_normalized_expression(
            adata_hvg, gene_list=available_chromatin, n_samples=25, return_mean=True
        )
        np.save(PROCESSED_DIR / "chromatin_expression.npy", chromatin_expr.values)
        chromatin_expr.to_csv(PROCESSED_DIR / "chromatin_expression.csv")
        logger.info(f"  Chromatin genes: {len(available_chromatin)}/{len(CHROMATIN_GENES)} "
                    f"available, shape {chromatin_expr.shape}")
    else:
        logger.warning("  No chromatin genes found in HVG set!")

    # 3. Biomarker proxy expression (denoised)
    available_proxies = [g for g in BIOMARKER_PROXY_GENES if g in adata_hvg.var_names]
    if available_proxies:
        proxy_expr = model.get_normalized_expression(
            adata_hvg, gene_list=available_proxies, n_samples=25, return_mean=True
        )
        np.save(PROCESSED_DIR / "biomarker_proxy_expression.npy", proxy_expr.values)
        proxy_expr.to_csv(PROCESSED_DIR / "biomarker_proxy_expression.csv")
        logger.info(f"  Biomarker proxies: {len(available_proxies)}/{len(BIOMARKER_PROXY_GENES)} "
                    f"available, shape {proxy_expr.shape}")
    else:
        logger.warning("  No biomarker proxy genes found!")

    # 4. UMAP for visualization
    adata_hvg.obsm["X_scvi"] = z
    sc.pp.neighbors(adata_hvg, use_rep="X_scvi")
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.5)

    # Save processed AnnData
    adata_hvg.write_h5ad(PROCESSED_DIR / "adata_scvi_processed.h5ad")
    logger.info(f"  Processed AnnData saved with UMAP + Leiden clustering")

    # 5. Save cell metadata
    adata_hvg.obs.to_csv(PROCESSED_DIR / "cell_metadata.csv")

    return z


def main():
    logger.info("=" * 60)
    logger.info("ResistanceMap v20 — scVI Integration Pipeline")
    logger.info("=" * 60)

    # Load data
    adata = load_atlas()

    # Preprocess
    adata = preprocess(adata)

    # Train scVI
    model = train_scvi(adata)

    # Extract all representations
    z = extract_representations(model, adata)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DONE. Outputs in data/processed/:")
    logger.info(f"  - scvi_latent_z.npy: {z.shape}")
    logger.info(f"  - chromatin_expression.csv")
    logger.info(f"  - biomarker_proxy_expression.csv")
    logger.info(f"  - adata_scvi_processed.h5ad")
    logger.info(f"  - cell_metadata.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()