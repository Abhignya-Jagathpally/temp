"""
resistancemap/data/single_cell_preprocessing.py
===============================================
Canonical scanpy preprocessing pipeline for MORT-FM single-cell inputs.

Pipeline (per ``configs/default.yaml`` scrna QC defaults + scanpy 1.10 idioms):

1. Detect raw counts. If ``adata.X`` looks normalised (non-integer), abort
   loudly — we cannot retroactively recover counts from log-norm space and
   the NB likelihood used by :class:`RNAEncoder` requires real counts.
2. Annotate mitochondrial fraction (``pct_counts_mt``).
3. Cell filter: ``min_genes=200``, ``max_genes=8000``, ``pct_counts_mt < 20``.
4. Gene filter: ``min_cells=3``.
5. Save raw counts to ``adata.layers["counts"]`` *before* normalising.
6. Normalize to TP10K + log1p; result lives in ``adata.X``.
7. Highly-variable-gene selection (top ``n_top_genes`` by Seurat v3 dispersion).
8. Enrich obs with MORT-FM-required columns:
   - ``patient_id`` derived from the GEO sample slug
   - ``pseudotime_disease_stage`` — INTEGER ordinal (0..K). *Explicitly named
     ``pseudotime_*`` so the endpoint validator + trajectory claim gate
     refuse to treat it as calendar time.*
   - ``disease_subtype`` — string label preserved as-is for downstream
     fairness audits.

Honest behaviour
----------------
* If the input matrix is not raw counts, raises :class:`ValueError`. We never
  reconstruct counts from log-norm.
* All filtering parameters are recorded in ``adata.uns["mortfm_qc"]`` so the
  exact provenance of the processed file is auditable.
* ``pseudotime_disease_stage`` is documented in ``adata.uns["pseudotime_note"]``
  as NOT calendar time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Canonical disease-stage ordinal for MM progression cohorts.
# These integers are PSEUDOTIME — not calendar months.
DEFAULT_DISEASE_STAGE_MAP: Dict[str, int] = {
    # GSE271107 conventions
    "healthy": 0,
    "HD": 0,
    "MGUS": 1,
    "SMM": 2,
    "MM": 3,
    # GSE124310 conventions (`condition` column)
    "NBM": 0,                   # normal bone marrow control
    "SMMh": 2,                  # SMM high-risk
    # AML / Beat-AML fallback (used by other ingestion paths)
    "AML": 3,
    "control": 0,
}


@dataclass
class QCReport:
    n_cells_in: int = 0
    n_cells_out: int = 0
    n_cells_filtered_low_genes: int = 0
    n_cells_filtered_high_genes: int = 0
    n_cells_filtered_high_mt: int = 0
    n_genes_in: int = 0
    n_genes_out: int = 0
    n_hvg_selected: int = 0
    min_genes: int = 0
    max_genes: int = 0
    max_pct_mt: float = 0.0
    n_top_genes: int = 0
    detected_counts_layer: str = ""
    pct_mt_p50: float = 0.0
    pct_mt_p95: float = 0.0
    samples_seen: List[str] = field(default_factory=list)


def _looks_like_counts(X) -> bool:
    """Heuristic: integer-valued (within 1e-3 tolerance) on a small sample."""
    if hasattr(X, "toarray"):
        sample = X[:50].toarray() if X.shape[0] >= 50 else X.toarray()
    else:
        sample = X[:50] if X.shape[0] >= 50 else X
    sample = np.asarray(sample)
    if sample.size == 0:
        return False
    return bool(np.allclose(sample, np.round(sample), atol=1e-3)) and sample.min() >= 0


def preprocess_scrna(
    adata,
    *,
    min_genes: int = 200,
    max_genes: int = 8000,
    max_pct_mt: float = 20.0,
    min_cells_per_gene: int = 3,
    n_top_genes: int = 2000,
    disease_stage_map: Optional[Dict[str, int]] = None,
    sample_obs_col: str = "sample",
    disease_stage_obs_col: str = "disease_stage",
    fallback_disease_col: Optional[str] = "condition",
):
    """In-place QC + normalisation + HVG selection. Returns the AnnData.

    The input AnnData must contain RAW counts in ``adata.X``. The processed
    AnnData has:
        * ``adata.X`` — log-normalised (TP10K + log1p)
        * ``adata.layers["counts"]`` — original raw counts (preserved)
        * ``adata.var["highly_variable"]`` — bool flag
        * ``adata.obs["patient_id"]`` — derived from sample slug
        * ``adata.obs["pseudotime_disease_stage"]`` — integer 0..K
        * ``adata.uns["mortfm_qc"]`` — :class:`QCReport` as dict
        * ``adata.uns["pseudotime_note"]`` — caveat about NOT being calendar time
    """
    import scanpy as sc

    if disease_stage_map is None:
        disease_stage_map = DEFAULT_DISEASE_STAGE_MAP

    report = QCReport(
        n_cells_in=int(adata.n_obs),
        n_genes_in=int(adata.n_vars),
        min_genes=min_genes,
        max_genes=max_genes,
        max_pct_mt=max_pct_mt,
        n_top_genes=n_top_genes,
    )
    if sample_obs_col in adata.obs.columns:
        report.samples_seen = sorted(adata.obs[sample_obs_col].astype(str).unique().tolist())[:64]

    # ---- 1. Sanity-check we have counts ----
    if not _looks_like_counts(adata.X):
        raise ValueError(
            "preprocess_scrna requires raw counts in adata.X — the input matrix "
            "looks log-normalised (non-integer min/max). Cannot reconstruct counts "
            "from log-norm; abort. If you have counts in a layer, pass adata with "
            "X=adata.layers['counts'] first."
        )
    report.detected_counts_layer = "adata.X"

    # ---- 2. MT annotation ----
    # GEO datasets vary on gene-name convention (HGNC symbols vs Ensembl).
    import pandas as pd
    if "feature_name" in adata.var.columns:
        names_for_mt = pd.Series(adata.var["feature_name"].astype(str).values, index=adata.var.index)
    else:
        names_for_mt = pd.Series(adata.var_names.astype(str).values, index=adata.var.index)
    adata.var["mt"] = names_for_mt.str.upper().str.startswith("MT-").values
    if not adata.var["mt"].any():
        if "chromosome" in adata.var.columns:
            chrom = pd.Series(adata.var["chromosome"].astype(str).values, index=adata.var.index)
            adata.var["mt"] = (chrom.str.upper() == "MT").values
        else:
            adata.var["mt"] = False
        if not bool(np.atleast_1d(adata.var["mt"]).any()):
            logger.warning(
                "No mitochondrial genes detected (no MT- prefix and no chromosome=MT); "
                "pct_counts_mt filter will be effectively disabled.",
            )
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    report.pct_mt_p50 = float(np.nanmedian(adata.obs["pct_counts_mt"].values))
    report.pct_mt_p95 = float(np.nanpercentile(adata.obs["pct_counts_mt"].values, 95))

    # ---- 3. Cell filter ----
    n0 = int(adata.n_obs)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    n1 = int(adata.n_obs)
    report.n_cells_filtered_low_genes = n0 - n1

    # Max-genes filter (manual — scanpy lacks a max_genes option).
    n_genes_per_cell = adata.obs.get("n_genes_by_counts", adata.X.getnnz(axis=1)
                                       if hasattr(adata.X, "getnnz") else (adata.X > 0).sum(axis=1))
    keep_high = np.asarray(n_genes_per_cell).reshape(-1) <= max_genes
    n_filtered_high = int((~keep_high).sum())
    adata._inplace_subset_obs(keep_high)
    report.n_cells_filtered_high_genes = n_filtered_high

    # MT filter
    keep_mt = adata.obs["pct_counts_mt"].values <= max_pct_mt
    n_filtered_mt = int((~keep_mt).sum())
    adata._inplace_subset_obs(keep_mt)
    report.n_cells_filtered_high_mt = n_filtered_mt

    # ---- 4. Gene filter ----
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    report.n_genes_out = int(adata.n_vars)

    # ---- 5. Save raw counts layer ----
    adata.layers["counts"] = adata.X.copy()

    # ---- 6. Normalize ----
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---- 7. HVG selection ----
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                     flavor="seurat_v3", layer="counts")
    except Exception:
        # Fallback: log-norm flavor.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    report.n_hvg_selected = int(adata.var["highly_variable"].sum())
    report.n_cells_out = int(adata.n_obs)

    # ---- 8. MORT-FM obs enrichment ----
    if sample_obs_col not in adata.obs.columns:
        adata.obs[sample_obs_col] = "unknown_sample"
    samples = adata.obs[sample_obs_col].astype(str)
    adata.obs["patient_id"] = _derive_patient_id(samples)

    if disease_stage_obs_col in adata.obs.columns:
        stage_strings = adata.obs[disease_stage_obs_col].astype(str)
    elif fallback_disease_col and fallback_disease_col in adata.obs.columns:
        stage_strings = adata.obs[fallback_disease_col].astype(str)
    else:
        stage_strings = _derive_disease_stage_from_sample(samples)
    pseudotime = stage_strings.map(lambda s: _stage_to_pseudotime(s, disease_stage_map))
    n_unmapped = int(pseudotime.isna().sum())
    if n_unmapped:
        logger.warning("%d cells have unmapped disease_stage values; setting pseudotime=-1.",
                       n_unmapped)
        pseudotime = pseudotime.fillna(-1)
    adata.obs["pseudotime_disease_stage"] = pseudotime.astype(int)
    adata.obs["disease_subtype"] = stage_strings

    # ---- 9. uns metadata ----
    adata.uns["mortfm_qc"] = asdict(report)
    adata.uns["pseudotime_note"] = (
        "pseudotime_disease_stage is an INTEGER ORDINAL of disease stage "
        "(HD/healthy=0, MGUS=1, SMM=2, MM=3). It is NOT calendar time and "
        "MUST NOT be passed to MORT-FM's trajectory head as a real timepoint."
    )

    # Make obs_names unique (downstream MORTBatch invariants require it).
    adata.obs_names_make_unique()

    logger.info(
        "Preprocessed: %d->%d cells, %d->%d genes, %d HVGs, pct_mt median=%.2f",
        report.n_cells_in, report.n_cells_out, report.n_genes_in, report.n_genes_out,
        report.n_hvg_selected, report.pct_mt_p50,
    )
    return adata


def _derive_patient_id(samples):
    """Strip GEO accession prefix and the '_donorN' / '_replicate' tails."""
    import re
    def _clean(s: str) -> str:
        # Drop GSM accession prefix like "GSM3528753_"
        s = re.sub(r"^GSM\d+_", "", str(s))
        # Drop trailing ".138N45P", ".138N", "_rep1", etc.
        s = re.sub(r"[._](138N45P|138N|rep\d+|R\d+)$", "", s)
        return s
    return samples.map(_clean)


def _derive_disease_stage_from_sample(samples):
    """Last-resort: extract disease stage from the sample-name prefix."""
    import re
    def _stage(s: str) -> str:
        s = re.sub(r"^GSM\d+_", "", str(s))
        token = s.split("-")[0].split("_")[0].split(".")[0]
        return token
    return samples.map(_stage)


def _stage_to_pseudotime(stage: str, m: Dict[str, int]):
    if not stage or stage == "nan":
        return np.nan
    if stage in m:
        return float(m[stage])
    # Case-insensitive token match.
    s_upper = stage.upper()
    for key, val in m.items():
        if key.upper() == s_upper:
            return float(val)
    # Token-prefix match (e.g. "SMMh" -> SMMh, "MM-1" -> MM)
    for key, val in m.items():
        if stage.startswith(key) or stage.upper().startswith(key.upper()):
            return float(val)
    return np.nan
