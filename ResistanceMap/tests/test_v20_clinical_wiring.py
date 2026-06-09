"""Regression tests for the v20 Bug #1 *wiring* fix.

The original Bug #1 patch widened the clinical vocabulary
(``FULL_CLINICAL_FEATURE_NAMES`` = 21) but the model path was pinned to
width 5 at three independent chokepoints, so the extra features were
silently clipped and never reached the SDE drift network — the fix was
cosmetic. These tests assert the chokepoints are now config-driven and a
21-wide clinical vector actually flows end-to-end into
``GraphEnergyResistanceSDE``.

All fixtures are structural (real schemas / tiny tensors, no fabricated
results) and run without any MMRF/CoMMpass data on disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from resistancemap.mortfm.canonical_trainer import CanonicalMORTFMTrainer
from resistancemap.mortfm.longitudinal import clinical_features as cf
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTBatch, MORTFMConfig


def _cfg(d_clinical: int = 5) -> MORTFMConfig:
    return MORTFMConfig(
        rna_input_dim=64, d_token=32, d_latent=64,
        n_time_grid=6, integration_time=1.0,
        use_rna=True, use_drug=False,
        survival_n_bins=6,
        d_clinical=d_clinical,
    )


def _minimal_batch(d_latent: int = 64, B: int = 3) -> MORTBatch:
    return MORTBatch(
        rna=torch.zeros(B, d_latent, dtype=torch.float32),
        drug=torch.zeros(B, 4),
        modality_mask={
            "rna": torch.ones(B, dtype=torch.bool),
            "drug": torch.ones(B, dtype=torch.bool),
        },
        patient_ids=[f"p{i}" for i in range(B)],
        time=torch.zeros(B),
    )


def _sde_drift_in_features(model: MORTFM) -> int:
    return model.lens_sde.drift.net[0].in_features


# ---------------------------------------------------------------------------
# Config defaults — back-compat (existing checkpoints stay loadable).
# ---------------------------------------------------------------------------
def test_config_default_widths_unchanged():
    cfg = MORTFMConfig()
    assert cfg.d_clinical == 5
    assert cfg.d_drug == 8


def test_default_model_drift_width_matches_legacy():
    model = _cfg(d_clinical=5)
    m = MORTFM(model)
    # drift in_features = d_latent + d_graph(16) + d_drug(8) + d_clinical(5)
    assert m.d_clinical == 5 and m.d_drug == 8
    assert _sde_drift_in_features(m) == 64 + 16 + 8 + 5


# ---------------------------------------------------------------------------
# The real fix — 21-wide clinical reaches the SDE unclipped.
# ---------------------------------------------------------------------------
def test_d_clinical_21_changes_sde_drift_width():
    m = MORTFM(_cfg(d_clinical=21))
    # B19 construction assertion must still pass at the new width.
    assert m.d_clinical == 21
    assert _sde_drift_in_features(m) == 64 + 16 + 8 + 21


def test_forward_consumes_full_21_clinical_vector():
    m = MORTFM(_cfg(d_clinical=21))
    m.eval()
    batch = _minimal_batch(B=3)
    clinical = torch.randn(3, 21)  # full lab-first panel width
    with torch.no_grad():
        out = m(batch, clinical=clinical)
    # If the old hardcoded clip-to-5 were still present, the SDE (built for
    # width 21) would receive a width-5 tensor and raise a shape error.
    assert torch.isfinite(out["z_traj"]).all()
    assert out["z_traj"].shape == (3, 6, 64)


def test_forward_zero_fill_tracks_d_clinical():
    m = MORTFM(_cfg(d_clinical=21))
    m.eval()
    batch = _minimal_batch(B=2)
    with torch.no_grad():
        out = m(batch, clinical=None)  # zero-filled to (B, 21), not (B, 5)
    assert torch.isfinite(out["z_traj"]).all()


def test_model_exposes_width_attrs_for_trainer():
    # The trainer's _clean_clinical / _drug_padded read these off the model.
    m = MORTFM(_cfg(d_clinical=21))
    assert int(getattr(m, "d_clinical")) == 21
    assert int(getattr(m, "d_drug")) == 8


# ---------------------------------------------------------------------------
# Trainer width-fitting helper (clip / pad / NaN-zero).
# ---------------------------------------------------------------------------
def test_fit_width_clips_and_pads_and_zeros_nan():
    fw = CanonicalMORTFMTrainer._fit_width
    assert fw(None, 5) is None
    # clip
    assert fw(torch.zeros(2, 30), 21).shape == (2, 21)
    # pad
    padded = fw(torch.ones(2, 3), 21)
    assert padded.shape == (2, 21)
    assert (padded[:, :3] == 1.0).all() and (padded[:, 3:] == 0.0).all()
    # NaN -> 0
    t = torch.tensor([[1.0, float("nan"), 3.0]])
    assert torch.isfinite(fw(t, 3)).all()


# ---------------------------------------------------------------------------
# Producer — flat (N, 21) encoder, honest masking, no fabrication.
# ---------------------------------------------------------------------------
def _write_outcomes(tmp_path):
    p = tmp_path / "outcomes.tsv"
    pd.DataFrame(
        {
            "submitter_id": ["P1", "P2"],
            "iss_stage": ["II", "III"],
            "age_at_diagnosis_days": [20000.0, 25000.0],
            "gender": ["male", "female"],
            "bort_1L": [1, 0],
            "n_treatments": [2, 5],
        }
    ).to_csv(p, sep="\t", index=False)
    return str(p)


def test_encode_for_lens_full_without_labs_is_21_wide_and_masks_labs(tmp_path):
    out = _write_outcomes(tmp_path)
    enc = cf.encode_for_lens_full(out, patient_ids=["P1", "P2"], visit_csv=None)
    assert enc.features.shape == (2, 21)
    assert enc.feature_names == list(cf.FULL_CLINICAL_FEATURE_NAMES)
    n_lab = len(cf.LONGITUDINAL_LAB_FEATURES)  # 16
    # No visit table -> every lab dim masked False (never fabricated).
    assert not enc.mask[:, :n_lab].any()
    assert (enc.features[:, :n_lab] == 0.0).all()
    # Baseline covariates still carry real signal/mask.
    assert enc.mask[:, n_lab:].all()
    # iss_stage_ordinal is the first baseline feature (II -> 1).
    assert enc.features[0, n_lab] == 1.0


def test_encode_for_lens_full_with_labs_populates_baseline_visit(tmp_path):
    out = _write_outcomes(tmp_path)
    visit = tmp_path / "visits.csv"
    pd.DataFrame(
        {
            "PUBLIC_ID": ["P1", "P1", "P2"],
            "VISITDY": [0, 60, 0],
            "D_LAB_serum_m_protein": [3.0, 1.0, np.nan],
            "D_LAB_cbc_hemoglobin": [10.0, 11.0, 9.0],
        }
    ).to_csv(visit, index=False)

    enc = cf.encode_for_lens_full(
        out, patient_ids=["P1", "P2"], visit_csv=str(visit),
        max_timepoints=4, interval_days=60,
    )
    assert enc.features.shape == (2, 21)
    fi = enc.feature_names.index
    # P1 baseline (earliest) visit m_protein == 3.0, observed.
    assert enc.features[0, fi("serum_m_protein")] == 3.0
    assert enc.mask[0, fi("serum_m_protein")]
    # P2 m_protein was NaN at its only visit -> masked, not fabricated.
    assert not enc.mask[1, fi("serum_m_protein")]
    assert enc.features[1, fi("serum_m_protein")] == 0.0
    # hemoglobin observed for both at baseline visit.
    assert enc.features[0, fi("cbc_hemoglobin")] == 10.0
    assert enc.features[1, fi("cbc_hemoglobin")] == 9.0
