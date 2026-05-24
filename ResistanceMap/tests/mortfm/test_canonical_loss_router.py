"""tests/mortfm/test_canonical_loss_router.py

Unit tests for ``assemble_canonical_loss`` and the individual canonical
loss helpers in ``resistancemap.training.canonical_mortfm_losses``.

The decisive contracts:
  * ``assemble_canonical_loss`` dispatches the right *set* of terms for each
    stage (E vs F vs G vs H).
  * Every loss term returns a sub-dict with an ``n_supervised: int`` field.
  * When the required supervision is missing the term reports
    ``n_supervised == 0`` (and the trainer's strict mode is what surfaces it
    as ``MissingSupervisionError`` — that path is covered in the smoke test).
"""

from __future__ import annotations

import pytest
import torch

from resistancemap.mortfm.canonical_trainer import (
    CanonicalMORTFMTrainer,
    MissingSupervisionError,
)
from resistancemap.mortfm.schemas import MORTBatch
from resistancemap.training.canonical_mortfm_losses import (
    REQUIRED_SUPERVISION_PER_STAGE,
    STAGE_LOSS_SPEC,
    assemble_canonical_loss,
    basin_transition_loss,
    canonical_trajectory_loss,
    hitting_time_loss,
    lens_survival_loss,
    sde_path_loss,
)


# --------------------------------------------------------------------------
# Synthetic FORWARD outputs (deterministic — these are NOT metric values).
# Using zeros/ones for outputs is structural: we only assert
# "loss computed, n_supervised reported correctly", never any metric magnitude.
# --------------------------------------------------------------------------


def _mk_outputs(B: int = 4, T: int = 4, d: int = 8, K: int = None, n_basins: int = 5):
    # v19 Phase 7: hitting_cdf and survival_curve MUST share their time
    # axis. If the caller omits K, default to K=T so survival_hitting_consistency_loss
    # validates. Legacy callers may still pass K explicitly to exercise the
    # mismatch-detection error path.
    if K is None:
        K = T
    return {
        "z0":               torch.zeros(B, d),
        "z_traj":           torch.linspace(0.0, 1.0, T)
                                 .view(1, T, 1).expand(B, T, d).clone(),
        "z_samples":        torch.zeros(2, B, T, d),
        "t_grid":           torch.linspace(0.0, 6.0, T),
        "hazard":           torch.full((B, K), 1.0 / K),
        "survival_curve":   torch.linspace(1.0, 0.5, K).unsqueeze(0).expand(B, K).clone(),
        "cif_per_event":    torch.zeros(B, K, 1),
        "basin_probs":      torch.full((B, T, n_basins), 1.0 / n_basins),
        "hitting_cdf":      torch.linspace(0.1, 0.9, T).unsqueeze(0).expand(B, T).clone(),
        "hitting_mean_tau": torch.full((B,), 3.0),
        "hitting_frac_hit": torch.full((B,), 0.5),
    }


def _mk_batch_with_all_supervision(B: int = 4, d: int = 8) -> MORTBatch:
    return MORTBatch(
        event_time=torch.tensor([3.0, 4.0, 2.0, 5.0])[:B],
        event_observed=torch.tensor([1.0, 0.0, 1.0, 1.0])[:B],
        resistance_label=torch.tensor([0, 1, 2, 3])[:B],
        future_state=torch.zeros(B, d),
        patient_ids=[f"P{i}" for i in range(B)],
    )


def _mk_batch_with_no_supervision(B: int = 4) -> MORTBatch:
    return MORTBatch(patient_ids=[f"P{i}" for i in range(B)])


# --------------------------------------------------------------------------
# Individual losses — n_supervised contract
# --------------------------------------------------------------------------


def test_lens_survival_loss_reports_n_supervised():
    out = _mk_outputs()
    batch = _mk_batch_with_all_supervision()
    sub = lens_survival_loss(out, batch)
    assert "n_supervised" in sub
    assert sub["n_supervised"] == 4
    assert torch.isfinite(sub["loss"]).item()
    assert "loss_survival_nll" in sub and "loss_brier" in sub


def test_lens_survival_loss_returns_zero_when_unsupervised():
    out = _mk_outputs()
    batch = _mk_batch_with_no_supervision()
    sub = lens_survival_loss(out, batch)
    assert sub["n_supervised"] == 0
    assert sub["loss"].item() == 0.0


def test_hitting_time_loss_n_supervised_counts_observed_events_only():
    out = _mk_outputs()
    batch = _mk_batch_with_all_supervision()
    sub = hitting_time_loss(out, batch)
    # event_observed = [1,0,1,1] -> 3 observed rows
    assert sub["n_supervised"] == 3
    assert torch.isfinite(sub["loss"]).item()


def test_hitting_time_loss_returns_zero_when_no_observed():
    out = _mk_outputs()
    batch = MORTBatch(
        event_time=torch.tensor([3.0, 4.0]),
        event_observed=torch.zeros(2),
        patient_ids=["a", "b"],
    )
    sub = hitting_time_loss(out, batch)
    assert sub["n_supervised"] == 0
    assert sub["loss"].item() == 0.0


def test_basin_transition_loss_counts_valid_labels():
    out = _mk_outputs(n_basins=5)
    batch = _mk_batch_with_all_supervision()
    sub = basin_transition_loss(out, batch)
    # All 4 labels are < n_basins (5) so all valid.
    assert sub["n_supervised"] == 4
    assert torch.isfinite(sub["loss"]).item()


def test_basin_transition_loss_zero_when_no_label():
    out = _mk_outputs()
    batch = _mk_batch_with_no_supervision()
    sub = basin_transition_loss(out, batch)
    assert sub["n_supervised"] == 0
    assert sub["loss"].item() == 0.0


def test_sde_path_loss_is_always_finite():
    out = _mk_outputs()
    batch = _mk_batch_with_no_supervision()
    sub = sde_path_loss(out, batch)
    assert "n_supervised" in sub
    # sde_path is regularisation — n_supervised == B even with no labels.
    assert sub["n_supervised"] == out["z_traj"].shape[0]
    assert torch.isfinite(sub["loss"]).item()


def test_canonical_trajectory_loss_inv_dt_weighting():
    B, T, d = 4, 5, 8
    pred = torch.randn(B, T, d)
    fut = torch.randn(B, d)
    dt = torch.tensor([1.0, 2.0, 4.0, 0.5])
    sub = canonical_trajectory_loss(pred, fut, dt)
    assert sub["n_supervised"] == 4
    assert torch.isfinite(sub["loss"]).item()


def test_canonical_trajectory_loss_zero_on_nonpositive_dt():
    B, T, d = 3, 4, 5
    pred = torch.randn(B, T, d)
    fut = torch.randn(B, d)
    dt = torch.tensor([0.0, -1.0, float("nan")])
    sub = canonical_trajectory_loss(pred, fut, dt)
    assert sub["n_supervised"] == 0
    assert sub["loss"].item() == 0.0


# --------------------------------------------------------------------------
# Stage dispatcher
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage,expected_terms", [
    ("E", ("trajectory", "sde_path", "basin_transition")),
    # v19 Phase 7: stage F gains hitting_nll + surv_hit_consistency
    ("F", ("survival", "hitting", "hitting_nll", "surv_hit_consistency")),
    ("G", ("basin_transition", "pathway_attribution", "pathway_evidence")),
    ("H", ("survival", "counterfactual_consistency")),
])
def test_assemble_canonical_loss_dispatches_per_stage(stage, expected_terms):
    out = _mk_outputs()
    batch = _mk_batch_with_all_supervision()
    bundle = assemble_canonical_loss(stage, out, batch, weights=None)
    assert tuple(bundle["components"].keys()) == expected_terms
    for term in expected_terms:
        sub = bundle["components"][term]
        assert "n_supervised" in sub, f"term {term!r} missing n_supervised"
        assert "loss" in sub
    assert torch.isfinite(bundle["loss_total"]).item()


def test_assemble_canonical_loss_rejects_unknown_stage():
    out = _mk_outputs()
    batch = _mk_batch_with_all_supervision()
    with pytest.raises(ValueError):
        assemble_canonical_loss("Z", out, batch, weights=None)


def test_required_supervision_table_matches_stage_spec():
    """The strict-supervision table must reference terms present in the
    stage spec — otherwise the strict policy points at a term that will
    never be dispatched."""
    for stage, term in REQUIRED_SUPERVISION_PER_STAGE.items():
        assert stage in STAGE_LOSS_SPEC, f"stage {stage!r} missing from STAGE_LOSS_SPEC"
        assert term in STAGE_LOSS_SPEC[stage], (
            f"required term {term!r} for stage {stage!r} is not in its dispatch list "
            f"{STAGE_LOSS_SPEC[stage]}"
        )


# --------------------------------------------------------------------------
# Strict-mode trigger (cross-checks the trainer wiring).
# --------------------------------------------------------------------------


def test_trainer_strict_raises_when_required_term_unsupervised(tmp_path):
    """End-to-end: a strict trainer driving stage F on a batch without
    event_time/event_observed must raise MissingSupervisionError when
    _losses_for_batch is invoked."""
    # We don't need a real model here — just a stub object that returns
    # synthetic outputs in canonical-dict shape, so the test focuses on the
    # strict policy wiring itself.

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._param = torch.nn.Parameter(torch.zeros(1))

        def to(self, *args, **kwargs):  # no-op; we're CPU-only
            return self

        def forward(self, batch, *, clinical=None, drug=None, graph_emb=None):
            return _mk_outputs(B=2)

        def forward_legacy(self, *a, **k):  # pragma: no cover
            raise AssertionError("forward_legacy must not be called")

    class _Loader:
        def __iter__(self):
            return iter([])  # not exercised here

    model = _DummyModel()
    trainer = CanonicalMORTFMTrainer(
        model, cfg={}, train_loader=_Loader(), val_loader=None,
        device="cpu", strict=True,
    )
    # Empty batch — no event_time, no event_observed.
    batch = MORTBatch(patient_ids=["a", "b"])
    with pytest.raises(MissingSupervisionError):
        trainer._losses_for_batch(batch, "F")
