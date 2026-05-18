"""Tests for resistancemap.data.single_cell_preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

from resistancemap.data.single_cell_preprocessing import (
    DEFAULT_DISEASE_STAGE_MAP,
    QCReport,
    _derive_patient_id,
    _stage_to_pseudotime,
    preprocess_scrna,
)


def _toy_anndata(n_cells: int = 200, n_genes: int = 300):
    """Build a tiny structural-only AnnData of Poisson counts with sample / disease_stage metadata."""
    rng = np.random.default_rng(0)
    X = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
    # Inject some MT- prefixed gene names to test mitochondrial annotation
    var_names = [f"MT-CO{i}" if i < 5 else f"GENE_{i}" for i in range(n_genes)]
    samples = np.array(["HD1"] * 50 + ["MGUS_1"] * 50 + ["SMM_1"] * 50 + ["MM_1"] * 50)
    disease = np.array(["healthy"] * 50 + ["MGUS"] * 50 + ["SMM"] * 50 + ["MM"] * 50)
    obs = pd.DataFrame({"sample": samples, "disease_stage": disease},
                       index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=var_names)
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_refuses_non_count_input():
    a = _toy_anndata()
    a.X = (a.X + 0.5).astype(np.float32)  # break the integer invariant
    with pytest.raises(ValueError, match="raw counts"):
        preprocess_scrna(a)


def test_stage_to_pseudotime_mapping():
    m = DEFAULT_DISEASE_STAGE_MAP
    assert _stage_to_pseudotime("MM", m) == 3.0
    assert _stage_to_pseudotime("MGUS", m) == 1.0
    assert _stage_to_pseudotime("healthy", m) == 0.0
    assert _stage_to_pseudotime("SMM", m) == 2.0
    # Prefix match
    assert _stage_to_pseudotime("MM_2", m) == 3.0
    # Unknown
    assert np.isnan(_stage_to_pseudotime("paneth_cell", m))


def test_derive_patient_id_strips_gsm_and_tail():
    samples = pd.Series(["GSM3528753_MGUS-1.138N", "GSM8369863_HD1", "MM_2"])
    cleaned = _derive_patient_id(samples)
    assert cleaned.iloc[0] == "MGUS-1"
    assert cleaned.iloc[1] == "HD1"
    assert cleaned.iloc[2] == "MM_2"


def test_pipeline_adds_canonical_obs_and_uns():
    # With Poisson counts at mean 0.5, most cells have < 200 genes -> they get filtered out.
    # Use a much richer toy with mean 5 so >200 genes per cell is realistic at n_genes=300.
    rng = np.random.default_rng(1)
    n_cells, n_genes = 200, 300
    X = rng.poisson(5.0, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"MT-CO{i}" if i < 5 else f"GENE_{i}" for i in range(n_genes)]
    obs = pd.DataFrame(
        {"sample": ["HD1"] * 50 + ["MGUS_1"] * 50 + ["SMM_1"] * 50 + ["MM_1"] * 50,
         "disease_stage": ["healthy"] * 50 + ["MGUS"] * 50 + ["SMM"] * 50 + ["MM"] * 50},
        index=[f"c{i}" for i in range(n_cells)],
    )
    a = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    out = preprocess_scrna(a, min_genes=10, max_genes=1000,
                           max_pct_mt=99.0, n_top_genes=50)
    assert "counts" in out.layers
    assert "patient_id" in out.obs.columns
    assert "pseudotime_disease_stage" in out.obs.columns
    assert "disease_subtype" in out.obs.columns
    # All 4 pseudotime ordinals should appear.
    assert set(out.obs["pseudotime_disease_stage"].unique().tolist()) == {0, 1, 2, 3}
    assert "mortfm_qc" in out.uns
    assert "pseudotime_note" in out.uns
    # Note must explicitly say NOT calendar time
    assert "NOT calendar time" in out.uns["pseudotime_note"]


def test_pipeline_preserves_counts_layer():
    rng = np.random.default_rng(2)
    n_cells, n_genes = 100, 200
    X = rng.poisson(5.0, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"GENE_{i}" for i in range(n_genes)]
    obs = pd.DataFrame(
        {"sample": ["HD1"] * 100, "disease_stage": ["healthy"] * 100},
        index=[f"c{i}" for i in range(n_cells)],
    )
    a = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    out = preprocess_scrna(a, min_genes=10, max_genes=1000, max_pct_mt=99.0, n_top_genes=20)
    # X is log-normalised (max < 100 after TP10K+log1p)
    assert float(out.X.max()) < 50.0
    # counts layer still has integer-like values
    counts = out.layers["counts"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    assert np.allclose(counts, np.round(counts), atol=1e-3)
