"""Tests for resistancemap.mortfm.schemas."""

from __future__ import annotations

import pytest
import torch

from resistancemap.mortfm.schemas import (
    DrugContext,
    MODALITY_ORDER,
    ModalityTensor,
    MORTBatch,
    MORTFMConfig,
    ModalityName,
    PatientCellSnapshot,
    ResistanceOutcome,
    TemporalTrainingPair,
)


def test_modality_tensor_shape_validation():
    v = torch.zeros(2, 5)
    mt = ModalityTensor(name="rna", values=v, feature_names=["g1", "g2", "g3", "g4", "g5"])
    assert mt.n_samples == 2 and mt.feature_dim == 5

    with pytest.raises(ValueError, match="feature dim is 5"):
        ModalityTensor(name="rna", values=v, feature_names=["g1", "g2", "g3"])

    with pytest.raises(ValueError, match="must be 2-D"):
        ModalityTensor(name="rna", values=torch.zeros(2, 3, 4, 5), feature_names=["a"] * 5)


def test_modality_tensor_mask_shape_validation():
    v = torch.zeros(2, 5)
    with pytest.raises(ValueError, match="mask shape"):
        ModalityTensor(name="rna", values=v, feature_names=["g"] * 5,
                       mask=torch.ones(3, 5, dtype=torch.bool))


def test_modality_order_is_stable():
    expected = ("rna", "atac", "methylation", "histone_ptm", "proteomics",
                "phosphoproteomics", "clinical", "drug")
    assert tuple(m.value for m in MODALITY_ORDER) == expected


def test_patient_snapshot_available_modalities():
    rna = ModalityTensor("rna", torch.zeros(1, 3), ["a", "b", "c"])
    snap = PatientCellSnapshot(patient_id="p1", sample_id="s1", disease="MM", rna=rna)
    assert snap.available_modalities() == [ModalityName.RNA]
    snap.drug = DrugContext("Bortezomib", "proteasome_inhibitor")
    assert ModalityName.DRUG in snap.available_modalities()


def test_resistance_outcome_label_methods():
    o = ResistanceOutcome(patient_id="p1", baseline_time=0.0, event_time=10.0, censored=False,
                          future_state=torch.zeros(5))
    assert o.has_survival_label() and o.has_future_state()
    o2 = ResistanceOutcome(patient_id="p1", baseline_time=0.0)
    assert not o2.has_survival_label() and not o2.has_future_state()


def test_temporal_pair_supervision_flag():
    snap = PatientCellSnapshot(patient_id="p1", sample_id="s1", disease="MM")
    out = ResistanceOutcome(patient_id="p1", baseline_time=0.0)
    pair = TemporalTrainingPair(x_t=snap, x_t_delta=None, outcome=out)
    assert not pair.has_any_supervision()
    out.event_time = 5.0
    assert pair.has_any_supervision()


def test_mortbatch_validate_for_loss_rejects_missing_labels():
    batch = MORTBatch(rna=torch.zeros(2, 3), patient_ids=["a", "b"])
    with pytest.raises(ValueError, match="event_time"):
        batch.validate_for_loss("survival")
    with pytest.raises(ValueError, match="future_state"):
        batch.validate_for_loss("trajectory")
    with pytest.raises(ValueError, match="resistance_label"):
        batch.validate_for_loss("resistance_state")
    with pytest.raises(ValueError, match="drug_response"):
        batch.validate_for_loss("drug_response")


def test_mortbatch_to_device_preserves_tensors():
    batch = MORTBatch(
        rna=torch.zeros(2, 3),
        modality_mask={"rna": torch.ones(2, dtype=torch.bool)},
        patient_ids=["a", "b"],
    )
    moved = batch.to(torch.device("cpu"))
    assert moved.rna is not None and moved.rna.device.type == "cpu"
    assert moved.modality_mask["rna"].device.type == "cpu"


def test_mortfmconfig_defaults():
    cfg = MORTFMConfig()
    assert cfg.d_latent > 0
    assert cfg.survival_kind in {"cox", "discrete", "competing_risks"}
    assert cfg.dynamics_kind in {"neural_ode", "neural_sde", "jump_sde"}
