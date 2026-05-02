"""
resistancemap/data/preprocessing.py
====================================
Multi-omics preprocessing pipeline and nested cross-validation splitter
for ResistanceMap v6 ICML/ICLR submission.

Handles:
  - RNA-seq: log1p, HVG selection, batch correction (ComBat/Harmony)
  - Proteomics: median normalization, KNN imputation, ID mapping
  - Epigenomics: ATAC-seq / methylation processing
  - PPI network: STRING v12 loading and adjacency construction
  - Drug response: IC50 binarization
  - Quality control and feature selection
  - Nested cross-validation with temporal awareness

Requirements:
    pip install numpy pandas scipy scikit-learn scanpy anndata
    Optional: pip install harmonypy combat
"""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse, stats

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold

try:
    import anndata as ad
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNA-seq Preprocessing
# ---------------------------------------------------------------------------

class RNASeqPreprocessor:
    """Preprocess RNA-seq count data for downstream modeling."""

    def __init__(self, n_top_genes=2000, min_counts=10, min_cells_frac=0.05,
                 normalization="cpm", batch_correction="none",
                 hvg_flavor="seurat_v3", scale=True):
        self.n_top_genes = n_top_genes
        self.min_counts = min_counts
        self.min_cells_frac = min_cells_frac
        self.normalization = normalization
        self.batch_correction = batch_correction
        self.hvg_flavor = hvg_flavor
        self.scale = scale
        self._hvg_mask = None
        self._gene_names = None
        self._scaler = None
        self._is_fitted = False

    def fit_transform(self, counts, gene_names=None, batch_labels=None):
        if isinstance(counts, pd.DataFrame):
            gene_names = np.array(counts.columns) if gene_names is None else gene_names
            counts = counts.values
        counts = counts.astype(np.float64)
        n_samples, n_genes = counts.shape
        if gene_names is None:
            gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
        self._gene_names = gene_names.copy()
        logger.info(f"RNA-seq preprocessing: {n_samples} samples, {n_genes} genes")
        min_cells = max(int(n_samples * self.min_cells_frac), 3)
        gene_detected = (counts > 0).sum(axis=0)
        gene_total = counts.sum(axis=0)
        keep_mask = (gene_detected >= min_cells) & (gene_total >= self.min_counts)
        counts = counts[:, keep_mask]
        gene_names = gene_names[keep_mask]
        logger.info(f"  After filtering: {counts.shape[1]} genes ({n_genes - counts.shape[1]} removed)")
        if self.normalization == "cpm":
            lib_sizes = counts.sum(axis=1, keepdims=True)
            lib_sizes = np.maximum(lib_sizes, 1)
            counts = counts / lib_sizes * 1e6
        elif self.normalization == "median_of_ratios":
            log_counts = np.log(counts + 1)
            geo_means = np.exp(log_counts.mean(axis=0))
            geo_means[geo_means == 0] = 1
            ratios = counts / geo_means
            size_factors = np.median(ratios, axis=1, keepdims=True)
            size_factors = np.maximum(size_factors, 1e-6)
            counts = counts / size_factors
        elif self.normalization == "scran":
            lib_sizes = counts.sum(axis=1, keepdims=True)
            median_lib = np.median(lib_sizes)
            size_factors = lib_sizes / median_lib
            counts = counts / np.maximum(size_factors, 1e-6)
        X = np.log1p(counts)
        if HAS_SCANPY and self.hvg_flavor in ("seurat_v3", "seurat"):
            adata = ad.AnnData(X=X)
            adata.var_names = pd.Index(gene_names)
            if batch_labels is not None:
                adata.obs["batch"] = batch_labels
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=min(self.n_top_genes, X.shape[1]),
                    flavor="seurat_v3" if self.hvg_flavor == "seurat_v3" else "seurat",
                    batch_key="batch" if batch_labels is not None else None,
                )
            hvg_mask = adata.var["highly_variable"].values
        else:
            gene_var = X.var(axis=0)
            n_select = min(self.n_top_genes, len(gene_var))
            threshold = np.sort(gene_var)[::-1][n_select - 1]
            hvg_mask = gene_var >= threshold
        self._hvg_mask = hvg_mask
        X = X[:, hvg_mask]
        selected_genes = gene_names[hvg_mask]
        logger.info(f"  Selected {X.shape[1]} highly variable genes")
        if batch_labels is not None and self.batch_correction != "none":
            X = self._batch_correct(X, batch_labels, selected_genes)
        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._is_fitted = True
        self._selected_genes = selected_genes
        return X, selected_genes

    def transform(self, counts):
        if not self._is_fitted:
            raise RuntimeError("Must call fit_transform first")
        counts = counts.astype(np.float64)
        if self.normalization == "cpm":
            lib_sizes = counts.sum(axis=1, keepdims=True)
            counts = counts / np.maximum(lib_sizes, 1) * 1e6
        X = np.log1p(counts)
        X = X[:, self._hvg_mask]
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    def _batch_correct(self, X, batch_labels, gene_names):
        if self.batch_correction == "combat":
            try:
                from combat.pycombat import pycombat
                df = pd.DataFrame(X.T, index=gene_names)
                corrected = pycombat(df, batch_labels)
                return corrected.values.T
            except ImportError:
                logger.warning("pycombat not installed, trying scanpy ComBat")
            if HAS_SCANPY:
                adata = ad.AnnData(X=X)
                adata.obs["batch"] = batch_labels
                sc.pp.combat(adata, key="batch")
                return adata.X
            logger.warning("No ComBat implementation available, skipping")
            return X
        elif self.batch_correction == "harmony":
            try:
                import harmonypy
                ho = harmonypy.run_harmony(X, pd.DataFrame({"batch": batch_labels}), "batch")
                return ho.Z_corr.T
            except ImportError:
                logger.warning("harmonypy not installed, skipping batch correction")
            return X
        return X


# ---------------------------------------------------------------------------
# Proteomics Preprocessing
# ---------------------------------------------------------------------------

class ProteomicsPreprocessor:
    """Preprocess proteomics data (mass-spec or reverse-phase protein array)."""

    def __init__(self, n_neighbors_impute=10, max_missing_frac=0.5,
                 n_top_proteins=500, log_transform=True):
        self.n_neighbors_impute = n_neighbors_impute
        self.max_missing_frac = max_missing_frac
        self.n_top_proteins = n_top_proteins
        self.log_transform = log_transform
        self._scaler = None
        self._selected_mask = None
        self._is_fitted = False

    def fit_transform(self, data, protein_names=None, protein_gene_map=None):
        if isinstance(data, pd.DataFrame):
            protein_names = np.array(data.columns) if protein_names is None else protein_names
            data = data.values.astype(np.float64)
        else:
            data = data.astype(np.float64)
        if protein_names is None:
            protein_names = np.array([f"prot_{i}" for i in range(data.shape[1])])
        logger.info(f"Proteomics preprocessing: {data.shape}")
        missing_frac = np.isnan(data).mean(axis=0)
        keep = missing_frac <= self.max_missing_frac
        data = data[:, keep]
        protein_names = protein_names[keep]
        logger.info(f"  After missing filter: {data.shape[1]} proteins")
        if self.log_transform:
            data = np.log2(np.maximum(data, 1e-10))
        sample_medians = np.nanmedian(data, axis=1, keepdims=True)
        global_median = np.nanmedian(data)
        data = data - sample_medians + global_median
        if np.any(np.isnan(data)):
            imputer = KNNImputer(n_neighbors=min(self.n_neighbors_impute, data.shape[0] - 1))
            data = imputer.fit_transform(data)
        if protein_gene_map:
            mapped_names = np.array([protein_gene_map.get(p, p) for p in protein_names])
            unique_genes = np.unique(mapped_names)
            aggregated = np.zeros((data.shape[0], len(unique_genes)))
            for i, gene in enumerate(unique_genes):
                mask = mapped_names == gene
                aggregated[:, i] = data[:, mask].mean(axis=1)
            data = aggregated
            protein_names = unique_genes
        variances = data.var(axis=0)
        n_select = min(self.n_top_proteins, len(variances))
        top_idx = np.argsort(variances)[::-1][:n_select]
        self._selected_mask = np.zeros(len(variances), dtype=bool)
        self._selected_mask[top_idx] = True
        data = data[:, top_idx]
        selected_names = protein_names[top_idx]
        self._scaler = RobustScaler()
        data = self._scaler.fit_transform(data)
        self._is_fitted = True
        return data, selected_names

    def transform(self, data):
        if not self._is_fitted:
            raise RuntimeError("Must call fit_transform first")
        if self.log_transform:
            data = np.log2(np.maximum(data.astype(np.float64), 1e-10))
        sample_medians = np.nanmedian(data, axis=1, keepdims=True)
        global_median = np.nanmedian(data)
        data = data - sample_medians + global_median
        if np.any(np.isnan(data)):
            imputer = KNNImputer(n_neighbors=min(self.n_neighbors_impute, data.shape[0] - 1))
            data = imputer.fit_transform(data)
        data = data[:, self._selected_mask]
        return self._scaler.transform(data)


# ---------------------------------------------------------------------------
# Epigenomics Preprocessing
# ---------------------------------------------------------------------------

class EpigenomicsPreprocessor:
    """Preprocess ATAC-seq peaks or DNA methylation beta values."""

    def __init__(self, data_type="atac", n_top_features=5000,
                 use_lsi=False, n_components=50):
        self.data_type = data_type
        self.n_top_features = n_top_features
        self.use_lsi = use_lsi
        self.n_components = n_components
        self._selected_mask = None
        self._lsi_model = None
        self._scaler = None
        self._is_fitted = False

    def fit_transform(self, data, feature_names=None):
        if isinstance(data, pd.DataFrame):
            feature_names = np.array(data.columns) if feature_names is None else feature_names
            data = data.values.astype(np.float64)
        else:
            data = data.astype(np.float64)
        if feature_names is None:
            feature_names = np.array([f"feat_{i}" for i in range(data.shape[1])])
        logger.info(f"Epigenomics preprocessing ({self.data_type}): {data.shape}")
        if self.data_type == "atac":
            data = self._preprocess_atac(data)
        elif self.data_type == "methylation":
            data = self._preprocess_methylation(data)
        variances = np.nanvar(data, axis=0)
        n_select = min(self.n_top_features, len(variances))
        top_idx = np.argsort(variances)[::-1][:n_select]
        self._selected_mask = np.zeros(len(variances), dtype=bool)
        self._selected_mask[top_idx] = True
        data = data[:, top_idx]
        selected_names = feature_names[top_idx]
        if self.use_lsi and data.shape[1] > self.n_components:
            from sklearn.decomposition import TruncatedSVD
            self._lsi_model = TruncatedSVD(n_components=self.n_components, random_state=42)
            data = self._lsi_model.fit_transform(data)
            selected_names = np.array([f"LSI_{i}" for i in range(self.n_components)])
        self._scaler = StandardScaler()
        data = self._scaler.fit_transform(data)
        self._is_fitted = True
        return data, selected_names

    def _preprocess_atac(self, data):
        tf = data / (data.sum(axis=1, keepdims=True) + 1e-10)
        n_cells = data.shape[0]
        n_cells_with_peak = (data > 0).sum(axis=0) + 1
        idf = np.log1p(n_cells / n_cells_with_peak)
        return tf * idf

    def _preprocess_methylation(self, data):
        data = np.clip(data, 0, 1)
        var = np.nanvar(data, axis=0)
        keep = var > 1e-6
        data = data[:, keep]
        epsilon = 1e-6
        data = np.log2((data + epsilon) / (1 - data + epsilon))
        return data

    def transform(self, data):
        if not self._is_fitted:
            raise RuntimeError("Must call fit_transform first")
        data = data.astype(np.float64)
        if self.data_type == "atac":
            data = self._preprocess_atac(data)
        elif self.data_type == "methylation":
            data = self._preprocess_methylation(data)
        data = data[:, self._selected_mask]
        if self._lsi_model is not None:
            data = self._lsi_model.transform(data)
        return self._scaler.transform(data)


# ---------------------------------------------------------------------------
# PPI Network
# ---------------------------------------------------------------------------

class PPINetworkLoader:
    """Load and process protein-protein interaction network from STRING v12."""

    def __init__(self, confidence_threshold=700, species=9606):
        self.confidence_threshold = confidence_threshold
        self.species = species
        self._adjacency = None
        self._gene_to_idx = None

    def load(self, string_file, gene_names, alias_file=None):
        protein_to_gene = {}
        if alias_file and os.path.exists(alias_file):
            aliases = pd.read_csv(alias_file, sep="\t", header=None,
                                  names=["string_id", "alias", "source"], low_memory=False)
            for _, row in aliases.iterrows():
                if "BioMart_HUGO" in str(row.get("source", "")):
                    protein_to_gene[row["string_id"]] = row["alias"]
        logger.info(f"Loading STRING interactions from {string_file}")
        interactions = pd.read_csv(string_file, sep=" ", low_memory=False,
                                   dtype={"protein1": str, "protein2": str, "combined_score": int})
        interactions = interactions[interactions["combined_score"] >= self.confidence_threshold]
        logger.info(f"  {len(interactions)} interactions above confidence {self.confidence_threshold}")
        gene_set = set(gene_names)
        self._gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n = len(gene_names)
        rows, cols, vals = [], [], []
        for _, row in interactions.iterrows():
            g1 = protein_to_gene.get(row["protein1"], row["protein1"].split(".")[-1])
            g2 = protein_to_gene.get(row["protein2"], row["protein2"].split(".")[-1])
            if g1 in gene_set and g2 in gene_set:
                i, j = self._gene_to_idx[g1], self._gene_to_idx[g2]
                score = row["combined_score"] / 1000.0
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([score, score])
        self._adjacency = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        n_edges = len(vals) // 2
        density = n_edges / (n * (n - 1) / 2) * 100
        logger.info(f"  PPI adjacency: {n} nodes, {n_edges} edges, density={density:.2f}%")
        return self._adjacency

    def load_from_precomputed(self, path):
        data = sparse.load_npz(path)
        self._adjacency = data
        logger.info(f"Loaded precomputed PPI: {data.shape}")
        return data

    @property
    def adjacency(self):
        return self._adjacency


# ---------------------------------------------------------------------------
# Drug Response Processing
# ---------------------------------------------------------------------------

class DrugResponseProcessor:
    """Process drug sensitivity data for binary classification."""

    def __init__(self, threshold_method="median", min_samples_per_drug=20):
        self.threshold_method = threshold_method
        self.min_samples_per_drug = min_samples_per_drug
        self._thresholds = {}

    def fit_transform(self, drug_response_df):
        required_cols = {"sample_id", "drug_name"}
        value_col = None
        for col in ["ic50", "IC50", "auc", "AUC", "ln_ic50", "LN_IC50"]:
            if col in drug_response_df.columns:
                value_col = col
                break
        if value_col is None:
            raise ValueError(f"No IC50/AUC column found. Columns: {list(drug_response_df.columns)}")
        results = []
        for drug, group in drug_response_df.groupby("drug_name"):
            if len(group) < self.min_samples_per_drug:
                continue
            values = group[value_col].dropna()
            if len(values) < self.min_samples_per_drug:
                continue
            if self.threshold_method == "median":
                threshold = values.median()
            elif self.threshold_method == "waterfall":
                sorted_vals = np.sort(values)
                diffs = np.diff(sorted_vals)
                max_gap_idx = np.argmax(diffs)
                threshold = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx + 1]) / 2
            elif self.threshold_method == "bimodal":
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(values.values.reshape(-1, 1))
                means = gmm.means_.flatten()
                threshold = np.mean(means)
            else:
                threshold = values.median()
            self._thresholds[drug] = float(threshold)
            is_ic50 = "ic50" in value_col.lower()
            if is_ic50:
                binary = (group[value_col] <= threshold).astype(int)
            else:
                binary = (group[value_col] >= threshold).astype(int)
            drug_result = group[["sample_id", "drug_name"]].copy()
            drug_result["label"] = binary.values
            drug_result["value"] = group[value_col].values
            drug_result["threshold"] = threshold
            results.append(drug_result)
        if not results:
            raise ValueError("No drugs with sufficient samples")
        binary_df = pd.concat(results, ignore_index=True)
        n_drugs = binary_df["drug_name"].nunique()
        logger.info(f"Binarized {n_drugs} drugs, {len(binary_df)} drug-sample pairs")
        return binary_df, self._thresholds


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Pathway-guided or data-driven feature selection."""

    CANCER_PATHWAY_GENES = {
        "hsa05200_pathways_in_cancer": [
            "TP53", "KRAS", "BRAF", "PIK3CA", "PTEN", "APC", "RB1", "MYC",
            "CDKN2A", "EGFR", "ERBB2", "JAK2", "STAT3", "NOTCH1", "WNT1",
            "CTNNB1", "NRAS", "RAF1", "MAP2K1", "MAPK1", "AKT1", "MTOR",
        ],
        "hsa04151_pi3k_akt": [
            "PIK3CA", "PIK3R1", "AKT1", "AKT2", "MTOR", "PTEN", "TSC1",
            "TSC2", "GSK3B", "FOXO3", "BAD", "BCL2", "MDM2",
        ],
        "hsa04010_mapk": [
            "KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2",
            "MAPK1", "MAPK3", "MAPK8", "FOS", "JUN", "MYC",
        ],
        "hsa04110_cell_cycle": [
            "RB1", "TP53", "CDKN1A", "CDKN2A", "CCND1", "CCNE1", "CDK4",
            "CDK6", "CDK2", "E2F1", "CDC25A", "CHEK1", "CHEK2",
        ],
        "hsa04210_apoptosis": [
            "BCL2", "BAX", "BAK1", "BID", "BCL2L1", "CASP3", "CASP8",
            "CASP9", "CYCS", "APAF1", "XIAP", "BIRC5", "MCL1",
        ],
        "hsa04064_nf_kappa_b": [
            "NFKB1", "RELA", "IKBKB", "CHUK", "TNF", "TNFRSF1A",
            "TRAF2", "TRAF6", "BIRC2", "BIRC3", "BCL2", "MYD88",
        ],
        "myeloma_specific": [
            "FGFR3", "MMSET", "WHSC1", "CCND1", "MAF", "MAFB", "IRF4",
            "XBP1", "BLIMP1", "PRDM1", "DKK1", "MYC", "TP53", "RB1",
            "CDKN2C", "FAM46C", "TRAF3", "CYLD", "NRAS", "KRAS", "BRAF",
        ],
    }

    def __init__(self, mode="combined", n_features=2000):
        self.mode = mode
        self.n_features = n_features
        self._selected_mask = None

    def select(self, X, gene_names, y=None):
        if self.mode == "pathway":
            return self._pathway_select(X, gene_names)
        elif self.mode == "variance":
            return self._variance_select(X, gene_names)
        elif self.mode == "mutual_info":
            return self._mi_select(X, gene_names, y)
        elif self.mode == "combined":
            return self._combined_select(X, gene_names, y)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _pathway_select(self, X, gene_names):
        pathway_genes = set()
        for genes in self.CANCER_PATHWAY_GENES.values():
            pathway_genes.update(genes)
        mask = np.isin(gene_names, list(pathway_genes))
        if mask.sum() < 10:
            logger.warning(f"Only {mask.sum()} pathway genes found, falling back to variance")
            return self._variance_select(X, gene_names)
        self._selected_mask = mask
        return X[:, mask], gene_names[mask]

    def _variance_select(self, X, gene_names):
        var = X.var(axis=0)
        n_sel = min(self.n_features, len(var))
        idx = np.argsort(var)[::-1][:n_sel]
        mask = np.zeros(len(var), dtype=bool)
        mask[idx] = True
        self._selected_mask = mask
        return X[:, idx], gene_names[idx]

    def _mi_select(self, X, gene_names, y):
        if y is None:
            return self._variance_select(X, gene_names)
        from sklearn.feature_selection import mutual_info_classif
        valid = ~np.isnan(y)
        mi = mutual_info_classif(X[valid], y[valid].astype(int), random_state=42)
        n_sel = min(self.n_features, len(mi))
        idx = np.argsort(mi)[::-1][:n_sel]
        mask = np.zeros(len(mi), dtype=bool)
        mask[idx] = True
        self._selected_mask = mask
        return X[:, idx], gene_names[idx]

    def _combined_select(self, X, gene_names, y):
        pathway_genes = set()
        for genes in self.CANCER_PATHWAY_GENES.values():
            pathway_genes.update(genes)
        pathway_mask = np.isin(gene_names, list(pathway_genes))
        var = X.var(axis=0)
        n_var = max(self.n_features - int(pathway_mask.sum()), self.n_features // 2)
        top_var_idx = np.argsort(var)[::-1][:n_var]
        var_mask = np.zeros(len(var), dtype=bool)
        var_mask[top_var_idx] = True
        combined = pathway_mask | var_mask
        self._selected_mask = combined
        return X[:, combined], gene_names[combined]


# ---------------------------------------------------------------------------
# Quality Control
# ---------------------------------------------------------------------------

class MultiOmicsQC:
    """Quality control for multi-omics integration."""

    def __init__(self, max_missing_modality_frac=0.5, outlier_n_mads=5.0):
        self.max_missing_modality_frac = max_missing_modality_frac
        self.outlier_n_mads = outlier_n_mads

    def filter_samples(self, modalities, sample_ids):
        n_samples = len(sample_ids)
        n_modalities = len(modalities)
        available = np.zeros(n_samples)
        for name, data in modalities.items():
            if data is not None:
                if data.ndim == 2:
                    valid = ~np.all(np.isnan(data), axis=1)
                else:
                    valid = ~np.isnan(data)
                available += valid.astype(float)
        missing_frac = 1 - (available / n_modalities)
        keep = missing_frac <= self.max_missing_modality_frac
        outlier_flags = np.zeros(n_samples, dtype=bool)
        for name, data in modalities.items():
            if data is not None and data.ndim == 2:
                row_sums = np.nansum(np.abs(data), axis=1)
                median_val = np.nanmedian(row_sums)
                mad = np.nanmedian(np.abs(row_sums - median_val))
                if mad > 0:
                    z = np.abs(row_sums - median_val) / (mad * 1.4826)
                    outlier_flags |= z > self.outlier_n_mads
        n_removed = (~keep).sum()
        n_outliers = outlier_flags.sum()
        logger.info(f"QC: removed {n_removed} samples (missing), flagged {n_outliers} outliers")
        filtered = {}
        for name, data in modalities.items():
            if data is not None:
                filtered[name] = data[keep]
            else:
                filtered[name] = None
        return filtered, sample_ids[keep], outlier_flags[keep]


# ---------------------------------------------------------------------------
# Multi-Omics Preprocessor (Orchestrator)
# ---------------------------------------------------------------------------

class MultiOmicsPreprocessor:
    """End-to-end multi-omics preprocessing pipeline."""

    def __init__(self, n_top_genes=2000, n_top_proteins=500, n_top_peaks=5000,
                 ppi_confidence=700, batch_correction="combat",
                 feature_selection="combined", qc_max_missing=0.5):
        self.rnaseq_proc = RNASeqPreprocessor(n_top_genes=n_top_genes, batch_correction=batch_correction)
        self.proteomics_proc = ProteomicsPreprocessor(n_top_proteins=n_top_proteins)
        self.epigenomics_proc = EpigenomicsPreprocessor(n_top_features=n_top_peaks)
        self.ppi_loader = PPINetworkLoader(confidence_threshold=ppi_confidence)
        self.drug_proc = DrugResponseProcessor()
        self.feature_selector = FeatureSelector(mode=feature_selection)
        self.qc = MultiOmicsQC(max_missing_modality_frac=qc_max_missing)

    def preprocess(self, expression=None, gene_names=None, proteomics=None,
                   protein_names=None, epigenomics=None, epi_names=None,
                   sample_ids=None, batch_labels=None, response_labels=None):
        results = {}
        if expression is not None:
            X_expr, sel_genes = self.rnaseq_proc.fit_transform(expression, gene_names, batch_labels)
            results["expression"] = X_expr
            results["gene_names"] = sel_genes
        else:
            sel_genes = gene_names
        if proteomics is not None:
            X_prot, sel_prots = self.proteomics_proc.fit_transform(proteomics, protein_names)
            results["proteomics"] = X_prot
            results["protein_names"] = sel_prots
        if epigenomics is not None:
            X_epi, sel_epi = self.epigenomics_proc.fit_transform(epigenomics, epi_names)
            results["epigenomics"] = X_epi
            results["epi_names"] = sel_epi
        if "expression" in results and response_labels is not None:
            X_sel, sel_genes = self.feature_selector.select(
                results["expression"], results["gene_names"], response_labels)
            results["expression_selected"] = X_sel
            results["selected_gene_names"] = sel_genes
        if sample_ids is not None:
            modalities = {k: results.get(k) for k in ["expression", "proteomics", "epigenomics"]}
            filtered, kept_ids, outliers = self.qc.filter_samples(modalities, sample_ids)
            results["kept_sample_ids"] = kept_ids
            results["outlier_flags"] = outliers
            for k, v in filtered.items():
                if v is not None:
                    results[k] = v
        return results


# ---------------------------------------------------------------------------
# Nested Cross-Validation Splitter
# ---------------------------------------------------------------------------

class DataSplitter:
    """Nested cross-validation with temporal awareness."""

    def __init__(self, n_outer_folds=5, n_inner_folds=5, seed=42, temporal_split=False):
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.seed = seed
        self.temporal_split = temporal_split

    def get_outer_splits(self, patient_ids, labels):
        valid_mask = ~np.isnan(labels)
        valid_idx = np.where(valid_mask)[0]
        valid_labels = labels[valid_mask]
        skf = StratifiedKFold(n_splits=self.n_outer_folds, shuffle=True, random_state=self.seed)
        splits = []
        for train_rel, test_rel in skf.split(valid_idx, valid_labels):
            train_idx = valid_idx[train_rel]
            test_idx = valid_idx[test_rel]
            train_pids = set(patient_ids[train_idx])
            test_pids = set(patient_ids[test_idx])
            assert len(train_pids & test_pids) == 0, "Patient leak detected!"
            splits.append((train_idx, test_idx))
        return splits

    def get_inner_splits(self, train_idx, labels):
        train_labels = labels[train_idx]
        skf = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.seed + 1)
        inner_splits = []
        for inner_train_rel, inner_val_rel in skf.split(train_idx, train_labels):
            inner_train_idx = train_idx[inner_train_rel]
            inner_val_idx = train_idx[inner_val_rel]
            inner_splits.append((inner_train_idx, inner_val_idx))
        return inner_splits

    def get_temporal_splits(self, patient_ids, labels, diagnosis_dates):
        valid_mask = ~np.isnan(labels) & ~pd.isna(diagnosis_dates)
        valid_idx = np.where(valid_mask)[0]
        dates = pd.to_datetime(diagnosis_dates[valid_mask])
        sorted_order = dates.argsort()
        sorted_idx = valid_idx[sorted_order]
        n = len(sorted_idx)
        splits = []
        for fold in range(self.n_outer_folds):
            test_start = int(n * (fold + 1) / (self.n_outer_folds + 1))
            test_end = int(n * (fold + 2) / (self.n_outer_folds + 1))
            test_end = min(test_end, n)
            train_idx = sorted_idx[:test_start]
            test_idx = sorted_idx[test_start:test_end]
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        return splits

    def get_nested_cv(self, patient_ids, labels, diagnosis_dates=None):
        if self.temporal_split and diagnosis_dates is not None:
            outer_splits = self.get_temporal_splits(patient_ids, labels, diagnosis_dates)
        else:
            outer_splits = self.get_outer_splits(patient_ids, labels)
        nested = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            inner = self.get_inner_splits(train_idx, labels)
            nested.append({
                "fold": fold_idx,
                "outer_train_idx": train_idx,
                "outer_test_idx": test_idx,
                "inner_splits": inner,
            })
        logger.info(f"Nested CV: {len(nested)} outer folds x {self.n_inner_folds} inner folds")
        return nested


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_rnaseq_preprocessor():
    np.random.seed(42)
    n_samples, n_genes = 50, 5000
    counts = np.random.negative_binomial(5, 0.3, size=(n_samples, n_genes)).astype(float)
    gene_names = np.array([f"GENE_{i}" for i in range(n_genes)])
    proc = RNASeqPreprocessor(n_top_genes=1000, batch_correction="none")
    X, sel = proc.fit_transform(counts, gene_names)
    assert X.shape == (n_samples, 1000), f"Expected (50, 1000), got {X.shape}"
    assert len(sel) == 1000
    assert np.abs(X.mean()) < 0.1
    print("  [PASS] test_rnaseq_preprocessor")


def test_proteomics_preprocessor():
    np.random.seed(42)
    data = np.random.lognormal(10, 2, size=(30, 200))
    data[np.random.rand(*data.shape) < 0.1] = np.nan
    proc = ProteomicsPreprocessor(n_top_proteins=100)
    X, names = proc.fit_transform(data)
    assert X.shape == (30, 100)
    assert not np.any(np.isnan(X))
    print("  [PASS] test_proteomics_preprocessor")


def test_drug_response_processor():
    df = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(100)] * 2,
        "drug_name": ["DrugA"] * 100 + ["DrugB"] * 100,
        "ic50": np.concatenate([np.random.lognormal(0, 1, 100), np.random.lognormal(1, 0.5, 100)]),
    })
    proc = DrugResponseProcessor()
    binary_df, thresholds = proc.fit_transform(df)
    assert "DrugA" in thresholds
    assert "DrugB" in thresholds
    assert set(binary_df["label"].unique()) == {0, 1}
    print("  [PASS] test_drug_response_processor")


def test_feature_selector():
    np.random.seed(42)
    n = 100
    gene_names = np.array(["TP53", "KRAS", "BRAF", "PTEN", "FAKE1", "FAKE2", "MYC"] +
                          [f"GENE_{i}" for i in range(993)])
    X = np.random.randn(n, len(gene_names))
    y = np.random.choice([0, 1], size=n)
    sel = FeatureSelector(mode="pathway", n_features=500)
    X_sel, sel_genes = sel.select(X, gene_names, y)
    assert "TP53" in sel_genes
    assert "KRAS" in sel_genes
    print("  [PASS] test_feature_selector")


def test_data_splitter():
    np.random.seed(42)
    n = 200
    pids = np.array([f"P{i}" for i in range(n)])
    labels = np.random.choice([0.0, 1.0], size=n, p=[0.3, 0.7])
    splitter = DataSplitter(n_outer_folds=5, n_inner_folds=5)
    nested = splitter.get_nested_cv(pids, labels)
    assert len(nested) == 5
    for fold in nested:
        assert len(fold["inner_splits"]) == 5
        train_set = set(fold["outer_train_idx"])
        test_set = set(fold["outer_test_idx"])
        assert len(train_set & test_set) == 0
    print("  [PASS] test_data_splitter")


def test_qc():
    np.random.seed(42)
    n = 50
    expr = np.random.randn(n, 100)
    expr[0, :] = np.nan
    expr[1, :] = 1000
    prot = np.random.randn(n, 50)
    ids = np.array([f"S{i}" for i in range(n)])
    qc = MultiOmicsQC()
    filtered, kept, outliers = qc.filter_samples({"expression": expr, "proteomics": prot}, ids)
    assert len(kept) < n
    print("  [PASS] test_qc")


def run_tests():
    print("Running preprocessing tests...")
    test_rnaseq_preprocessor()
    test_proteomics_preprocessor()
    test_drug_response_processor()
    test_feature_selector()
    test_data_splitter()
    test_qc()
    print("All preprocessing tests passed!")


if __name__ == "__main__":
    run_tests()
