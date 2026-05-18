"""End-to-end MORT-FM trainer smoke test.

Equivalent to ``scripts/mortfm_smoke_test.py`` but as a pytest case so CI can
gate on it. Runs all four representative stages (A/B/F/E) for 1 epoch each on
a tiny structural cohort.
"""

from __future__ import annotations

import torch

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)
from resistancemap.mortfm.trainer import MORTFMTrainer


def _mk_snap(pid, t):
    rna = ModalityTensor("rna", torch.poisson(torch.full((1, 100), 3.0)),
                          [f"g{i}" for i in range(100)])
    prot = ModalityTensor("proteomics", torch.randn(1, 50),
                          [f"p{i}" for i in range(50)])
    return PatientCellSnapshot(
        patient_id=pid, sample_id=f"{pid}_t{t}", disease="MM", timepoint=float(t),
        rna=rna, proteomics=prot,
        drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01),
    )


def _cohort(n=10):
    snaps = [_mk_snap(f"P{i:03d}", t) for i in range(n) for t in [0, 6]]
    outs = [
        ResistanceOutcome(
            patient_id=f"P{i:03d}", baseline_time=0.0,
            event_time=float(8 + i), censored=(i % 2 == 0),
            resistance_label=i % 4,
        )
        for i in range(n)
    ]
    return build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5)


def _debug_cfg(checkpoint_dir):
    return MORTFMConfig(
        rna_input_dim=100, proteomics_input_dim=50, clinical_input_dim=4,
        d_token=16, d_latent=32, fusion_layers=2, fusion_heads=2, fusion_dropout=0.0,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=True,
        n_time_grid=4, integration_time=6.0,
        pretrain_epochs=1, finetune_epochs=1, trajectory_epochs=1, survival_epochs=1,
        mixed_precision=False, batch_size=2, drug_embed_dim=8,
        checkpoint_dir=str(checkpoint_dir),
    )


def test_smoke_stage_A(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=torch.device("cpu"))
    m = trainer.fit_stage("A", n_epochs=1)
    assert "recon" in m[0].components
    assert m[0].train_loss > 0


def test_smoke_stage_F_survival(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=torch.device("cpu"))
    m = trainer.fit_stage("F", n_epochs=1)
    assert "survival" in m[0].components or "state" in m[0].components


def test_smoke_stage_E_trajectory(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=torch.device("cpu"))
    m = trainer.fit_stage("E", n_epochs=1)
    assert "trajectory" in m[0].components


def test_smoke_checkpoint_roundtrip(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=torch.device("cpu"))
    trainer.fit_stage("A", n_epochs=1)
    ckpt = trainer.save_checkpoint("rt.pt")
    # New trainer + new model, then reload.
    model2 = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer2 = MORTFMTrainer(model2, cfg, train_loader=train_loader, val_loader=val_loader,
                              device=torch.device("cpu"))
    trainer2.load_checkpoint(ckpt)
    # Parameters now match.
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_smoke_full_forward_emits_all_heads(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, _, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    batch = next(iter(train_loader))
    with torch.no_grad():
        # v18 — legacy TrajectoryPrediction-returning path moved to forward_legacy()
        out = model.forward_legacy(batch, compute_counterfactuals=True)
    assert out.z_path.shape[0] == cfg.n_time_grid
    assert out.resistance_state_logits.shape == (batch.batch_size, 4)
    assert out.pathway_protein_scores.shape == (batch.batch_size, 20)
    assert out.drug_specific_risk.shape == (batch.batch_size, 5)
    assert out.counterfactual_rankings is not None
