"""v20 Phase 1-6 tests: GEO loaders, survival baselines, scVI guards,
config fields, PK decoder, mechanism classifier, and the validation harness.

These are unit tests: small deterministic fixtures exercise code paths/shapes.
No metric is computed on fabricated *pipeline* data — the fabrication rule is
about the pipeline/loaders, which here are tested precisely for their
no-fabrication behaviour (raise on missing/empty inputs).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(rel_path: str):
    """Import a numeric-prefixed script file (e.g. 11_validate_open_access.py)."""
    p = REPO / rel_path
    spec = importlib.util.spec_from_file_location(p.stem.replace("11_", "s11_"), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _harrell_cindex(t, e, r):
    t, e, r = map(np.asarray, (t, e, r))
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        for j in range(len(t)):
            if t[j] > t[i]:
                den += 1
                num += 1.0 if r[i] > r[j] else (0.5 if r[i] == r[j] else 0.0)
    return num / den if den else float("nan")


# ───────────────────────── Phase 4: config ─────────────────────────
def test_config_has_v20_fields():
    from resistancemap.config import DataConfig
    c = DataConfig()
    assert hasattr(c, "gdc_open_dir")
    assert hasattr(c, "geo_scrna_dir")
    assert hasattr(c, "geo_bulk_dir")
    assert c.scvi_n_latent == 30
    assert c.survival_endpoint == "imwg_pfs"  # Bug #2: not the tt2l proxy


# ───────────────────── Phase 1.3: geo_bulk_loader ───────────────────
def test_extract_survival_labels_censoring_and_drop():
    from resistancemap.data.geo_bulk_loader import extract_survival_labels
    import pandas as pd
    df = pd.DataFrame({
        "sid": ["a", "b", "c", "d"],
        "t": [100.0, 0.0, 50.0, np.nan],     # b dropped (t<=0), d dropped (nan)
        "e": ["dead", "alive", 1, 0],
    })
    out = extract_survival_labels(df, time_col="t", event_col="e", id_col="sid")
    assert out["structured"].dtype.names == ("event", "time")
    assert len(out["time"]) == 2                 # only a and c survive the filter
    assert out["event"].tolist() == [True, True]  # 'dead' and 1 -> True


def test_geo_bulk_loaders_raise_on_missing_dir(tmp_path):
    from resistancemap.data.geo_bulk_loader import load_gse24080, load_gse136337
    with pytest.raises(FileNotFoundError):
        load_gse24080(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_gse136337(tmp_path)


# ───────────────────── Phase 1.2: geo_scrna_loader ──────────────────
def test_geo_supplementary_url():
    from resistancemap.data.geo_scrna_loader import geo_supplementary_url, SUPPORTED_ACCESSIONS
    url = geo_supplementary_url("GSE161801")
    assert "GSE161nnn/GSE161801" in url and url.endswith("GSE161801_RAW.tar")
    assert "GSE189460" in SUPPORTED_ACCESSIONS


def test_geo_scrna_parsers_raise_on_missing(tmp_path):
    from resistancemap.data import geo_scrna_loader as g
    with pytest.raises(FileNotFoundError):
        g.parse_10x_h5(tmp_path / "nope.h5")
    with pytest.raises(FileNotFoundError):
        g.load_h5ad(tmp_path / "nope.h5ad")


def test_geo_scrna_merge_inner_join():
    sc = pytest.importorskip("scanpy")  # noqa: F841
    ad = pytest.importorskip("anndata")
    from resistancemap.data import geo_scrna_loader as g
    rng = np.random.RandomState(0)
    a = ad.AnnData(rng.poisson(1.0, (5, 4)).astype("float32"))
    a.var_names = ["g1", "g2", "g3", "g4"]
    a.obs["dataset"] = "GSE1"
    b = ad.AnnData(rng.poisson(1.0, (6, 4)).astype("float32"))
    b.var_names = ["g2", "g3", "g4", "g5"]
    b.obs["dataset"] = "GSE2"
    merged = g.merge_datasets([a, b])
    assert merged.n_obs == 11
    assert set(merged.var_names) == {"g2", "g3", "g4"}  # inner join, no zero-pad
    with pytest.raises(ValueError):
        g.merge_datasets([])


# ───────────────────── Phase 2: survival baselines ──────────────────
def _make_static_survival(n=120, p=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta = rng.randn(p)
    risk = X @ beta
    # higher risk -> shorter time
    t = np.exp(-0.5 * risk + 0.2 * rng.randn(n)) * 100 + 1
    e = (rng.rand(n) < 0.7).astype(float)  # ~70% events
    return X, t, e


def test_gradient_boosted_survival_learns_signal():
    from resistancemap.baselines.survival_baselines import GradientBoostedSurvivalBaseline
    X, t, e = _make_static_survival()
    Xtr, ttr, etr = X[:90], t[:90], e[:90]
    Xte, tte, ete = X[90:], t[90:], e[90:]
    m = GradientBoostedSurvivalBaseline(seed=0).fit(Xtr, ttr, etr)
    r = m.predict_risk(Xte)
    assert r.shape == (Xte.shape[0],) and np.isfinite(r).all()
    assert _harrell_cindex(tte, ete, r) > 0.6


def test_elasticnet_cox_runs_and_shapes():
    from resistancemap.baselines.survival_baselines import ElasticNetCoxBaseline
    X, t, e = _make_static_survival()
    m = ElasticNetCoxBaseline(seed=0).fit(X[:90], t[:90], e[:90])
    r = m.predict_risk(X[90:])
    assert r.shape == (30,) and np.isfinite(r).all()


def test_pangea_landmark_cox_trajectory_and_raise():
    from resistancemap.baselines.survival_baselines import (
        PangeaLandmarkCox, RequiresLongitudinalTrajectories,
    )
    rng = np.random.RandomState(1)
    N, T, F = 60, 5, 3
    traj = rng.randn(N, T, F).astype("float32")
    tmask = np.ones((N, T), dtype=bool)
    risk = traj[:, 2, 0]  # value at landmark drives risk
    t = np.exp(-0.5 * risk + 0.1 * rng.randn(N)) * 100 + 1
    e = (rng.rand(N) < 0.7)
    m = PangeaLandmarkCox(landmark_idx=2, seed=1).fit(traj, tmask, t, e)
    r = m.predict_risk(traj, tmask)
    assert r.shape == (N,) and np.isfinite(r).all()
    with pytest.raises(RequiresLongitudinalTrajectories):
        m.fit(rng.randn(N, F), tmask, t, e)  # 2D -> reject


def test_ferle_lstm_survival_shapes_and_raise():
    from resistancemap.baselines.survival_baselines import (
        FerleLSTMSurvival, RequiresLongitudinalTrajectories,
    )
    rng = np.random.RandomState(2)
    N, T, F = 30, 4, 3
    traj = rng.randn(N, T, F).astype("float32")
    tmask = np.ones((N, T), dtype=bool)
    t = (rng.rand(N) * 100 + 1).astype("float64")
    e = (rng.rand(N) < 0.6)
    m = FerleLSTMSurvival(hidden=8, epochs=3, seed=2).fit(traj, tmask, t, e)
    r = m.predict_risk(traj, tmask)
    assert r.shape == (N,) and np.isfinite(r).all()
    with pytest.raises(RequiresLongitudinalTrajectories):
        m.fit(rng.randn(N, F), tmask, t, e)


def test_baseline_runner_registers_new_names():
    mod = _load_script("scripts/mortfm/10_run_baselines.py")
    for name in ("elasticnet_cox", "gradient_boosted_survival",
                 "pangea_landmark_cox", "ferle_lstm_crbm"):
        assert name in mod.PATIENT_LONGITUDINAL_BASELINES
    assert mod.LONGITUDINAL_TRAJECTORY_BASELINES == {"pangea_landmark_cox", "ferle_lstm_crbm"}


def test_baseline_runner_static_cox_fit_predict():
    mod = _load_script("scripts/mortfm/10_run_baselines.py")
    X, t, e = _make_static_survival()
    bl = mod._PatientLongitudinalBaseline("elasticnet_cox", seed=0)
    bl.fit(X[:90], t[:90], e[:90], clinical_train=None)
    r = bl.predict_risk(X[90:], clinical_test=None)
    assert r.shape == (30,) and np.isfinite(r).all()


def test_baseline_runner_trajectory_baseline_raises_not_fabricates():
    mod = _load_script("scripts/mortfm/10_run_baselines.py")
    from resistancemap.baselines.survival_baselines import RequiresLongitudinalTrajectories
    X, t, e = _make_static_survival()
    bl = mod._PatientLongitudinalBaseline("pangea_landmark_cox", seed=0)
    with pytest.raises(RequiresLongitudinalTrajectories):
        bl.fit(X[:90], t[:90], e[:90], clinical_train=None)


# ───────────────────── Phase 3: scVI guards ─────────────────────────
def test_scvi_encoder_constants_and_import_guard():
    from resistancemap.models.scvi_encoder import (
        CHROMATIN_READER_WRITER_GENES, BIOMARKER_PROXY_GENES, SCVIEncoder,
    )
    assert len(CHROMATIN_READER_WRITER_GENES) == 20
    assert "B2M" in BIOMARKER_PROXY_GENES
    if importlib.util.find_spec("scvi") is None:
        with pytest.raises(ImportError, match="scvi-tools"):
            SCVIEncoder()


def test_contrastive_resistance_import_guard():
    from resistancemap.models.contrastive_resistance import ResistanceProgramIsolator
    if importlib.util.find_spec("scvi") is None:
        with pytest.raises(ImportError, match="scvi-tools"):
            ResistanceProgramIsolator()


# ───────────────────── Phase 5: interpretability ────────────────────
def test_pk_observation_decoder():
    import torch
    from resistancemap.models.pk_observation_decoder import (
        PKObservationDecoder, DEFAULT_BIOMARKERS,
    )
    dec = PKObservationDecoder(d_latent=16)
    z = torch.randn(8, 16)
    level = dec(z)
    assert level.shape == (8, len(DEFAULT_BIOMARKERS))
    assert (level >= 0).all()  # softplus -> nonneg
    decayed = dec.predict_after(z, torch.full((8,), 30.0))
    assert (decayed <= level + 1e-5).all()  # PK decay never increases level
    target = torch.rand(8, len(DEFAULT_BIOMARKERS))
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = dec.compute_loss(z, target, mask)
    assert torch.isfinite(loss)


def test_mechanism_classifier():
    import torch
    from resistancemap.models.mechanism_classifier import (
        MechanismClassifier, RESISTANCE_MECHANISMS,
    )
    clf = MechanismClassifier(d_latent=16)
    z = torch.randn(5, 16)
    logits = clf(z)
    assert logits.shape == (5, len(RESISTANCE_MECHANISMS))
    proba = clf.predict_proba(z)
    assert torch.allclose(proba.sum(-1), torch.ones(5), atol=1e-5)
    labels = clf.predict(z)
    assert len(labels) == 5 and all(l in RESISTANCE_MECHANISMS for l in labels)


# ───────────────────── Phase 6: validation harness ──────────────────
def test_validation_harness_dry_run_and_skips(tmp_path):
    mod = _load_script("scripts/mortfm/11_validate_open_access.py")
    assert mod.main(["--dry-run"]) == 0
    # No dirs -> every path reports skipped, no fabrication.
    assert mod.validate_gdc(None, None, 10, 0)["status"] == "skipped"
    assert mod.validate_gse136337(None, None, 10, 0)["status"] == "skipped"
    assert mod.validate_gse24080(None, None, 10, 0)["status"] == "skipped"
