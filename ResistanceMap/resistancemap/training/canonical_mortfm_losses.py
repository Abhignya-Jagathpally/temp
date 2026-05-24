"""
resistancemap/training/canonical_mortfm_losses.py
=================================================
Loss functions that consume the canonical ``MORTFM.forward()`` LENS dict.

These are the **only** loss helpers that touch the canonical dict surface
(``z_traj``, ``hazard``, ``survival_curve``, ``cif_per_event``, ``basin_probs``,
``hitting_cdf``, ``hitting_mean_tau``, ``t_grid`` etc.). The legacy
:mod:`resistancemap.training.mortfm_losses` module still serves the
``TrajectoryPrediction`` dataclass surface used by
:meth:`MORTFM.forward_legacy`.

Each function in this module returns a dict::

    {
        "loss":          torch.Tensor scalar (or zero if no supervision),
        "n_supervised":  int — how many rows in the batch actually contributed,
        # ... optionally additional named sub-terms.
    }

Returning ``n_supervised`` is mandatory so the v18.2 strict supervision
coverage policy implemented in :mod:`resistancemap.mortfm.trainer` can audit
which losses had real labels each step.

Honest-behaviour invariants
---------------------------
* No ``np.random`` or ``torch.randn``-based label fabrication appears anywhere.
* When the supervision is absent the loss returns a *zero scalar with
  ``n_supervised == 0``*; the trainer raises if that situation persists across
  too many batches in a "strict" stage.
* NaN-target rows are masked out before the loss is taken.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from resistancemap.mortfm.schemas import MORTBatch

logger = logging.getLogger(__name__)

# Public symbols (used by canonical_trainer and tests).
__all__ = [
    "lens_survival_loss",
    "hitting_time_loss",
    "basin_transition_loss",
    "sde_path_loss",
    "canonical_trajectory_loss",
    "latent_future_state_loss",
    "hitting_time_nll",
    "survival_hitting_consistency_loss",
    "pathway_attribution_loss",
    "counterfactual_consistency_loss",
    "pathway_evidence_regularizer",
    "assemble_canonical_loss",
    "STAGE_LOSS_SPEC",
    "REQUIRED_SUPERVISION_PER_STAGE",
]


# ---------------------------------------------------------------------------
# Stage spec — which losses fire in each canonical patient-level stage.
# ---------------------------------------------------------------------------

#: Maps each patient-level stage to the canonical loss terms it dispatches.
#: Stages outside this dict are foundation stages and use the legacy losses.
#: v19 Phase 7: stage F gains ``hitting_nll`` (the canonical discrete-NLL on
#: the shared t_grid, replacing the prototype ``hitting_time_loss`` MSE) and
#: ``surv_hit_consistency`` (the directional invariant). The legacy
#: ``hitting`` term is kept so old configs still resolve to a valid loss.
STAGE_LOSS_SPEC: Dict[str, tuple] = {
    "E": ("trajectory", "sde_path", "basin_transition"),
    "F": ("survival", "hitting", "hitting_nll", "surv_hit_consistency"),
    "G": ("basin_transition", "pathway_attribution", "pathway_evidence"),
    "H": ("survival", "counterfactual_consistency"),
}


#: Which loss terms must have ``n_supervised > 0`` for the stage to count as
#: supervised under the v18.2 strict policy. A stage may dispatch *more*
#: losses than it requires; only the listed terms gate the strict check.
#: Note: ``pathway_attribution`` and ``counterfactual_consistency`` are
#: optional enrichment terms — they do NOT gate their stages because the
#: required labels (``pathway_targets``, ``counterfactual_rankings``) are
#: not available in every dataset. The strict gating remains on the core
#: terms (``basin_transition`` for G, ``survival`` for H).
REQUIRED_SUPERVISION_PER_STAGE: Dict[str, str] = {
    "E": "trajectory",
    "F": "survival",
    "G": "basin_transition",
    "H": "survival",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_term(reference: torch.Tensor) -> Dict[str, Any]:
    """Return a zero-loss term that matches ``reference``'s device + dtype."""
    return {
        "loss": torch.zeros((), device=reference.device, dtype=reference.dtype),
        "n_supervised": 0,
    }


def _get_field(source: Any, name: str) -> Any:
    """Read ``name`` from either a Mapping or a dataclass-like object."""
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


# ---------------------------------------------------------------------------
# Individual canonical losses
# ---------------------------------------------------------------------------


def lens_survival_loss(
    out: Mapping[str, Any],
    batch: Any,
    *,
    integration_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Discrete-time NLL + Brier-style proper score on the LENS survival surface.

    Reads ``out["hazard"]``, ``out["survival_curve"]``, ``out["cif_per_event"]``.
    Reads ``batch.event_time`` and ``batch.event_observed`` (or the equivalent
    mapping keys).

    Parameters
    ----------
    out :
        Canonical-dict output of ``MORTFM.forward``.
    batch :
        Either a :class:`MORTBatch` instance or a ``Mapping`` with the
        equivalent supervision fields.
    integration_time :
        Optional overrride for the integration horizon used to bucketise
        ``event_time``. When omitted the per-batch maximum is used.

    Returns
    -------
    dict
        ``{"loss": Tensor, "loss_survival_nll": Tensor, "loss_brier": Tensor,
        "n_supervised": int}``. The loss term is a sum of NLL + Brier.
    """
    hazard = out["hazard"]                       # (B, K) or (B, K, n_events)
    surv = out["survival_curve"]                 # (B, K)
    et = _get_field(batch, "event_time")
    eo = _get_field(batch, "event_observed")

    # The competing-risk head emits hazard as (B, K, n_events); collapse
    # to a single composite hazard by summing across events so we get one
    # discrete-time NLL per row even in the multi-event configuration.
    if hazard.dim() == 3:
        hazard = hazard.sum(dim=-1)              # (B, K)

    if et is None or eo is None:
        return {
            "loss": torch.zeros((), device=hazard.device, dtype=hazard.dtype),
            "loss_survival_nll": torch.zeros((), device=hazard.device, dtype=hazard.dtype),
            "loss_brier": torch.zeros((), device=hazard.device, dtype=hazard.dtype),
            "n_supervised": 0,
        }

    # Cast event-time / event-observed to the hazard's dtype/device.
    et = et.to(device=hazard.device, dtype=hazard.dtype)
    eo = eo.to(device=hazard.device, dtype=hazard.dtype)

    # Only rows with a finite event_time can contribute.
    valid = torch.isfinite(et)
    if valid.sum() == 0:
        return {
            "loss": torch.zeros((), device=hazard.device, dtype=hazard.dtype),
            "loss_survival_nll": torch.zeros((), device=hazard.device, dtype=hazard.dtype),
            "loss_brier": torch.zeros((), device=hazard.device, dtype=hazard.dtype),
            "n_supervised": 0,
        }

    B, K = hazard.shape
    if integration_time is None:
        # Use the largest observed event time as the upper edge so every row
        # falls into a real bin; never invent labels beyond what's measured.
        integration_time = float(et[valid].max().item())
        if integration_time <= 0.0:
            integration_time = 1.0

    # Bin edges: K+1 boundaries from 0 to integration_time.
    edges = torch.linspace(0.0, integration_time, K + 1, device=hazard.device, dtype=hazard.dtype)
    # bucketize returns the index of the first edge that is > et; subtract 1
    # to get the bin index in [0, K-1].
    bin_idx = torch.bucketize(et, edges[1:]).clamp_(0, K - 1)

    eps = 1e-8
    h_at_t = hazard.gather(1, bin_idx[:, None]).squeeze(-1)         # (B,)
    s_at_t = surv.gather(1, bin_idx[:, None]).squeeze(-1).clamp_min(eps)  # (B,)

    log_h = torch.log(h_at_t.clamp_min(eps))
    log_s_prev = torch.log(s_at_t)

    # NLL: for an observed event we pay -log h + -log S(t_{idx-1});
    # for a censored row we pay -log S(t_{idx}).
    nll_per_row = -(eo * (log_h + log_s_prev) + (1.0 - eo) * torch.log(s_at_t))
    nll_per_row = nll_per_row[valid]
    nll = nll_per_row.mean()

    # Brier-style proper score on the survival at t: target is 1-eo (still
    # surviving means S(t)=1 if censored at t; for observed events the
    # target is 0). Uses observed rows only.
    obs_mask = (valid & (eo > 0.5))
    if obs_mask.sum() > 0:
        brier = (s_at_t[obs_mask] - 0.0).pow(2).mean()
    else:
        brier = torch.zeros((), device=hazard.device, dtype=hazard.dtype)

    total = nll + brier
    return {
        "loss": total,
        "loss_survival_nll": nll,
        "loss_brier": brier,
        "n_supervised": int(valid.sum().item()),
    }


def hitting_time_loss(
    out: Mapping[str, Any],
    batch: Any,
) -> Dict[str, Any]:
    """Discrete-NLL of observed event time against ``out['hitting_cdf']``.

    Reads:
        out['hitting_cdf']      : (B, T) CDF over ``t_grid``
        out['hitting_mean_tau'] : (B,)   predicted mean hitting time
        out['t_grid']           : (T,)   integration grid
        batch.event_time        : (B,)   observed time-to-resistance
        batch.event_observed    : (B,)   0/1

    Returns ``{"loss": Tensor, "n_supervised": int}``. If the batch lacks
    ``event_time`` / ``event_observed`` returns a zero loss with
    ``n_supervised=0`` (no exception).
    """
    cdf = out["hitting_cdf"]                  # (B, T)
    t_grid = out["t_grid"]                    # (T,)
    mean_tau = out["hitting_mean_tau"]        # (B,)
    et = _get_field(batch, "event_time")
    eo = _get_field(batch, "event_observed")

    if et is None or eo is None:
        return _zero_term(cdf)

    et = et.to(device=cdf.device, dtype=cdf.dtype)
    eo = eo.to(device=cdf.device, dtype=cdf.dtype)

    # Observed rows only — censoring contributes via the lens_survival_loss.
    obs_mask = torch.isfinite(et) & (eo > 0.5)
    n_obs = int(obs_mask.sum().item())
    if n_obs == 0:
        return _zero_term(cdf)

    # Map observed times onto the t_grid; ``bucketize`` is monotone in et.
    bin_idx = torch.bucketize(et, t_grid).clamp_(0, cdf.shape[1] - 1)
    cdf_at_t = cdf.gather(1, bin_idx[:, None]).squeeze(-1).clamp(1e-8, 1.0 - 1e-8)
    # Discrete NLL: maximise CDF(t) for observed event rows.
    nll = -torch.log(cdf_at_t[obs_mask]).mean()
    # Plus a small MSE regulariser on mean_tau so the predicted mean tracks
    # observed time (cheap, identifiable supervision signal).
    mse = F.mse_loss(mean_tau[obs_mask], et[obs_mask])
    return {
        "loss": nll + mse,
        "loss_hitting_nll": nll,
        "loss_hitting_mse": mse,
        "n_supervised": n_obs,
    }


def basin_transition_loss(
    out: Mapping[str, Any],
    batch: Any,
) -> Dict[str, Any]:
    """Cross-entropy on the terminal basin probability vs ``batch.resistance_label``.

    Reads ``out['basin_probs']`` (shape ``(B, T, n_basins)``) and
    ``batch.resistance_label`` (shape ``(B,)`` integer in
    ``[0, n_basins)``). Returns zero loss + ``n_supervised=0`` if no labels.
    """
    basin_probs = out["basin_probs"]                # (B, T, n_basins)
    rl = _get_field(batch, "resistance_label")
    if rl is None:
        return _zero_term(basin_probs)

    terminal = basin_probs[:, -1, :].clamp_min(1e-8)         # (B, n_basins)
    targets = rl.to(device=terminal.device).long()
    valid = (targets >= 0) & (targets < terminal.shape[-1])
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return _zero_term(basin_probs)

    log_probs = torch.log(terminal)
    loss = F.nll_loss(log_probs[valid], targets[valid])
    return {"loss": loss, "n_supervised": n_valid}


def sde_path_loss(
    out: Mapping[str, Any],
    batch: Any,
    *,
    path_length_weight: float = 1e-2,
    smoothness_weight: float = 1e-2,
) -> Dict[str, Any]:
    """Lightweight regulariser on the SDE trajectory.

    This term is *always* finite: it touches only the model's own ``z_traj``
    output, so it has no batch supervision dependence. It penalises
    excessive path length and discrete-time second-difference roughness
    along the latent trajectory, which keeps the SDE from drifting to
    pathological energy basins in early training.

    Returns ``{"loss": ..., "n_supervised": B}`` because every row of
    ``z_traj`` contributed.
    """
    z_traj = out["z_traj"]                          # (B, T, d_latent)
    if z_traj.shape[1] < 2:
        return _zero_term(z_traj)

    # First difference (path length).
    deltas = z_traj[:, 1:, :] - z_traj[:, :-1, :]   # (B, T-1, d_latent)
    path_len = deltas.pow(2).sum(dim=-1).mean()

    if z_traj.shape[1] >= 3:
        # Second difference (smoothness).
        second = z_traj[:, 2:, :] - 2.0 * z_traj[:, 1:-1, :] + z_traj[:, :-2, :]
        roughness = second.pow(2).sum(dim=-1).mean()
    else:
        roughness = torch.zeros((), device=z_traj.device, dtype=z_traj.dtype)

    loss = path_length_weight * path_len + smoothness_weight * roughness
    return {
        "loss": loss,
        "loss_path_length": path_len,
        "loss_smoothness": roughness,
        "n_supervised": int(z_traj.shape[0]),
    }


def canonical_trajectory_loss(
    predicted_z_traj: torch.Tensor,
    encoded_z_future: torch.Tensor,
    delta_t: torch.Tensor,
) -> Dict[str, Any]:
    """Per-row latent-space loss normalised by per-row time gap.

    This is the ``latent_future_state_loss`` from the phase_1_6_7 doc: for
    every patient we have an observed future encoded state and a delta_t
    (months) between baseline and that future. We compare the predicted
    terminal latent to the encoded future, weighting each row by
    ``1 / max(delta_t, eps)`` so rows with shorter gaps (which should be
    easier to predict) contribute proportionally more.

    Parameters
    ----------
    predicted_z_traj :
        ``(B, T, d_latent)`` SDE trajectory; the terminal slice is used.
    encoded_z_future :
        ``(B, d_latent)`` foundation-encoded future state.
    delta_t :
        ``(B,)`` per-row time gap in months.

    Returns
    -------
    dict with ``loss`` (scalar Tensor) and ``n_supervised`` (int).
    """
    if predicted_z_traj.ndim != 3:
        raise ValueError(
            f"predicted_z_traj must be (B, T, d_latent); got {tuple(predicted_z_traj.shape)}"
        )
    if encoded_z_future.ndim != 2:
        raise ValueError(
            f"encoded_z_future must be (B, d_latent); got {tuple(encoded_z_future.shape)}"
        )

    pred_T = predicted_z_traj[:, -1, :]              # (B, d_latent)
    if pred_T.shape != encoded_z_future.shape:
        raise ValueError(
            f"predicted terminal shape {tuple(pred_T.shape)} does not match "
            f"encoded_z_future shape {tuple(encoded_z_future.shape)}"
        )

    valid = torch.isfinite(delta_t) & (delta_t > 0.0)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return _zero_term(predicted_z_traj)

    eps = 1e-3
    inv_dt = (1.0 / delta_t.clamp_min(eps))[valid]                # (n_valid,)
    sq = (pred_T[valid] - encoded_z_future[valid]).pow(2).sum(dim=-1)
    loss = (inv_dt * sq).mean()
    return {"loss": loss, "n_supervised": n_valid}


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, *, sigma: float = 1.0) -> torch.Tensor:
    """Radial-basis-function MMD^2 between two same-d_latent point sets.

    Both ``x`` and ``y`` are ``(N, d)``. Symmetric, biased-but-consistent
    estimator (no diagonal removal); we use it as a regulariser, not as a
    statistical test, so the bias is acceptable.
    """
    def k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.exp(-d2 / (2.0 * sigma * sigma))

    return k(x, x).mean() + k(y, y).mean() - 2.0 * k(x, y).mean()


def latent_future_state_loss(
    predicted_z_traj: torch.Tensor,
    encoded_z_future: torch.Tensor,
    delta_t_days: torch.Tensor,
    *,
    mmd_min_batch: int = 16,
    mmd_sigma: float = 1.0,
    delta_t_floor: float = 1.0,
) -> Dict[str, Any]:
    """Per-row L2 in latent space, normalised by per-row Δt (in days).

    This is Phase 1's canonical trajectory loss. It compares the SDE-evolved
    terminal latent against the **encoder's own** view of the follow-up
    snapshot — the only dimensionally-consistent comparison for a latent
    SDE. The v17 "truncate raw RNA to d_latent" path is dead.

    Per-row mask: only rows with finite, positive ``delta_t_days`` and finite
    target latents contribute. Rows that fail the mask receive zero gradient.

    The optional MMD bonus is gated by ``len(batch_supervised) >= mmd_min_batch``
    so we do not pay an O(B^2) RBF kernel on a 2-row batch.

    Returns
    -------
    dict
        ``{"loss_traj_latent": Tensor, "loss_traj_mmd": Tensor,
        "loss": Tensor, "n_supervised": int}``. ``loss`` is the sum of the
        two; the trainer multiplies it by the stage weight.
    """
    if predicted_z_traj.ndim != 3:
        raise ValueError(
            f"predicted_z_traj must be (B, T, d_latent); got "
            f"{tuple(predicted_z_traj.shape)}"
        )
    if encoded_z_future.ndim != 2:
        raise ValueError(
            f"encoded_z_future must be (B, d_latent); got "
            f"{tuple(encoded_z_future.shape)}"
        )
    pred_T = predicted_z_traj[:, -1, :]  # (B, d_latent)
    if pred_T.shape != encoded_z_future.shape:
        raise ValueError(
            f"latent_future_state_loss: predicted terminal shape "
            f"{tuple(pred_T.shape)} != encoded_z_future shape "
            f"{tuple(encoded_z_future.shape)}. Both MUST be the same encoder's "
            f"output."
        )

    device = predicted_z_traj.device
    dtype = predicted_z_traj.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if delta_t_days.shape != (pred_T.shape[0],):
        raise ValueError(
            f"delta_t_days must have shape (B,); got {tuple(delta_t_days.shape)}"
        )

    dt = delta_t_days.to(device=device, dtype=dtype)
    valid = (
        torch.isfinite(dt)
        & (dt > 0.0)
        & torch.isfinite(encoded_z_future).all(dim=-1)
        & torch.isfinite(pred_T).all(dim=-1)
    )
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {
            "loss_traj_latent": zero,
            "loss_traj_mmd": zero,
            "loss": zero,
            "n_supervised": 0,
        }

    pred_m = pred_T[valid]
    targ_m = encoded_z_future[valid]
    dt_m = dt[valid].clamp_min(delta_t_floor)
    sq = (pred_m - targ_m).pow(2).sum(dim=-1)               # (n_valid,)
    l2 = (sq / dt_m).mean()

    if n_valid >= mmd_min_batch:
        mmd = _mmd_rbf(pred_m, targ_m, sigma=mmd_sigma)
    else:
        mmd = zero

    return {
        "loss_traj_latent": l2,
        "loss_traj_mmd": mmd,
        "loss": l2 + mmd,
        "n_supervised": n_valid,
    }


def hitting_time_nll(
    hitting_cdf: torch.Tensor,
    t_grid: torch.Tensor,
    event_time: torch.Tensor,
    event_observed: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Discrete NLL on the empirical hitting CDF, on the shared canonical grid.

    For an uncensored row with event at grid bin ``k``:
        L = -log(F(k) - F(k-1) + eps)
    For a censored row with last-known-event-free time at bin ``k``:
        L = -log(1 - F(k) + eps)

    This is the Phase 7 invariant: the CDF over ``t_grid`` is interpreted as
    a true survival predictor on the *same* time axis as the survival head,
    so the NLL is comparable across the two subsystems.

    Returns
    -------
    dict
        ``{"loss_hit_nll": Tensor, "loss": Tensor, "n_supervised": int}``.
    """
    if hitting_cdf.ndim != 2:
        raise ValueError(
            f"hitting_cdf must be (B, T); got {tuple(hitting_cdf.shape)}"
        )
    B, T = hitting_cdf.shape
    if t_grid.numel() != T:
        raise ValueError(
            f"t_grid length {t_grid.numel()} != hitting_cdf T={T}; the SDE "
            f"and survival head must share the canonical_time_grid."
        )

    device = hitting_cdf.device
    dtype = hitting_cdf.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if event_time is None or event_observed is None:
        return {"loss_hit_nll": zero, "loss": zero, "n_supervised": 0}

    et = event_time.to(device=device, dtype=dtype)
    eo = event_observed.to(device=device, dtype=dtype)
    valid = torch.isfinite(et)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {"loss_hit_nll": zero, "loss": zero, "n_supervised": 0}

    # Snap each row's event_time onto the nearest grid bin.
    tg = t_grid.to(device=device, dtype=dtype)
    diff = (et.unsqueeze(-1) - tg.unsqueeze(0)).abs()
    k = diff.argmin(dim=-1).clamp_(0, T - 1)                    # (B,)

    cdf_at_k = hitting_cdf.gather(1, k.unsqueeze(-1)).squeeze(-1)
    k_prev = (k - 1).clamp_(min=0)
    cdf_prev = hitting_cdf.gather(1, k_prev.unsqueeze(-1)).squeeze(-1)
    # At k=0 there is no preceding bin; the implicit F(-1)=0.
    cdf_prev = torch.where(k > 0, cdf_prev, torch.zeros_like(cdf_prev))

    pmf_at_k = (cdf_at_k - cdf_prev).clamp_min(eps)
    surv_at_k = (1.0 - cdf_at_k).clamp_min(eps)

    nll_per_row = -(eo * pmf_at_k.log() + (1.0 - eo) * surv_at_k.log())
    nll = nll_per_row[valid].mean()
    return {"loss_hit_nll": nll, "loss": nll, "n_supervised": n_valid}


def survival_hitting_consistency_loss(
    survival_curve: torch.Tensor,
    hitting_cdf: torch.Tensor,
    cif_other_events: Optional[torch.Tensor] = None,
    *,
    margin: float = 0.0,
) -> Dict[str, Any]:
    """Hinge penalty enforcing ``S(t) <= 1 - F_hit(t) + sum(c_other(t))``.

    The directional consistency invariant from Phase 7: a patient who is
    still event-free at time t must not have already entered the resistant
    basin. With competing-risk corrections allowed, the upper bound on
    S(t) is ``1 - F_hit(t) + sum_r c_other_r(t)``.

    Shape contract: both ``survival_curve`` and ``hitting_cdf`` must be
    ``(B, T)`` on the same canonical grid. ``cif_other_events`` may be
    ``(B, T)`` or ``(B, T, R)`` (summed over R internally).

    Returns
    -------
    dict
        ``{"loss_surv_hit_consistency": Tensor, "loss": Tensor,
        "n_supervised": int}``. ``n_supervised`` is always batch size
        because this is an unsupervised internal-consistency penalty.
    """
    if survival_curve.shape != hitting_cdf.shape:
        raise ValueError(
            f"survival_curve {tuple(survival_curve.shape)} != hitting_cdf "
            f"{tuple(hitting_cdf.shape)} — grids unaligned. Wire both through "
            f"the canonical_time_grid from resistancemap.mortfm.trajectory.grid."
        )

    upper = 1.0 - hitting_cdf
    if cif_other_events is not None:
        cif = cif_other_events.to(device=upper.device, dtype=upper.dtype)
        if cif.dim() == 3:
            cif = cif.sum(dim=-1)
        if cif.shape != upper.shape:
            raise ValueError(
                f"cif_other_events {tuple(cif.shape)} must match "
                f"{tuple(upper.shape)}"
            )
        upper = upper + cif

    excess = (survival_curve - upper - margin).clamp(min=0.0)
    loss = excess.mean()
    return {
        "loss_surv_hit_consistency": loss,
        "loss": loss,
        "n_supervised": int(survival_curve.shape[0]),
    }


# ---------------------------------------------------------------------------
# Pathway & counterfactual losses (v19 Phase 8)
# ---------------------------------------------------------------------------


def pathway_attribution_loss(
    out: Mapping[str, Any],
    batch: Any,
    *,
    sparsity_weight: float = 1e-3,
) -> Dict[str, Any]:
    """BCE + L1 sparsity on ``protein_scores`` vs ``pathway_targets``.

    Reads:
        out['protein_scores']    : (B, n_proteins) — sigmoid-ready attribution
                                   logits from the pathway head.
        batch.pathway_targets    : (B, n_proteins) — binary 0/1 labels
                                   indicating known pathway membership.

    When ``pathway_targets`` is absent or all-NaN the loss returns zero with
    ``n_supervised=0`` (no exception).

    Returns
    -------
    dict
        ``{"loss": Tensor, "loss_bce": Tensor, "loss_sparsity": Tensor,
        "n_supervised": int}``.
    """
    scores = out.get("protein_scores")
    if scores is None:
        # Model head not wired — return a zero term anchored to z_traj.
        return _zero_term(out["z_traj"])

    targets = _get_field(batch, "pathway_targets")
    if targets is None:
        return _zero_term(scores)

    targets = targets.to(device=scores.device, dtype=scores.dtype)

    # Mask out rows where targets are entirely NaN (unsupervised patients).
    finite_mask = torch.isfinite(targets)
    row_valid = finite_mask.any(dim=-1)                     # (B,)
    n_valid = int(row_valid.sum().item())
    if n_valid == 0:
        return _zero_term(scores)

    s_valid = scores[row_valid]
    t_valid = targets[row_valid]
    # Per-element finite mask (some proteins may lack labels in some rows).
    elem_mask = torch.isfinite(t_valid)
    # Replace NaN targets with 0 for the BCE call (masked out below).
    t_safe = torch.where(elem_mask, t_valid, torch.zeros_like(t_valid))

    bce_unreduced = F.binary_cross_entropy_with_logits(
        s_valid, t_safe, reduction="none",
    )
    # Zero out elements where the target was NaN.
    bce_unreduced = bce_unreduced * elem_mask.float()
    n_elems = elem_mask.sum().clamp_min(1)
    bce = bce_unreduced.sum() / n_elems.float()

    # L1 sparsity on the raw logits — encourages the model to zero out
    # proteins that are NOT pathway members instead of hedging.
    sparsity = s_valid.abs().mean()

    loss = bce + sparsity_weight * sparsity
    return {
        "loss": loss,
        "loss_bce": bce,
        "loss_sparsity": sparsity,
        "n_supervised": n_valid,
    }


def counterfactual_consistency_loss(
    out: Mapping[str, Any],
    batch: Any,
    *,
    margin: float = 0.1,
) -> Dict[str, Any]:
    """Pairwise margin loss on predicted vs known drug-target rankings.

    Reads:
        out['counterfactual_pred'] : (B, n_drugs) — predicted efficacy scores
                                      (higher = more effective).
        batch.counterfactual_rankings : (B, n_drugs) — ordinal ranking labels
                                        (lower rank = more effective, i.e.
                                        rank 1 > rank 2). NaN indicates an
                                        unlabelled drug for that patient.

    For every ordered pair ``(i, j)`` of drugs within a row where
    ``rank_i < rank_j`` (drug *i* is more effective), we enforce::

        pred_i - pred_j >= margin

    via a hinge loss. This loss is *not* supervision-gated in the strict
    sense — when ``counterfactual_rankings`` is absent, it returns zero with
    ``n_supervised=0``.

    Returns
    -------
    dict
        ``{"loss": Tensor, "n_supervised": int, "n_pairs": int}``.
    """
    pred = out.get("counterfactual_pred")
    if pred is None:
        return _zero_term(out["z_traj"])

    rankings = _get_field(batch, "counterfactual_rankings")
    if rankings is None:
        return _zero_term(pred)

    rankings = rankings.to(device=pred.device, dtype=pred.dtype)

    B, D = pred.shape
    total_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    total_pairs = 0
    n_supervised = 0

    for b in range(B):
        r = rankings[b]                                     # (D,)
        valid_drugs = torch.isfinite(r)
        if valid_drugs.sum() < 2:
            continue
        n_supervised += 1
        idx = torch.where(valid_drugs)[0]
        r_valid = r[idx]
        p_valid = pred[b, idx]
        # All ordered pairs: i beats j iff r_valid[i] < r_valid[j].
        # Expand into pairwise comparison matrices.
        r_i = r_valid.unsqueeze(1)                          # (m, 1)
        r_j = r_valid.unsqueeze(0)                          # (1, m)
        pair_mask = (r_i < r_j)                             # i is better
        if pair_mask.sum() == 0:
            continue
        p_i = p_valid.unsqueeze(1).expand_as(pair_mask)
        p_j = p_valid.unsqueeze(0).expand_as(pair_mask)
        # Hinge: want pred_i - pred_j >= margin
        hinge = (margin - (p_i - p_j)).clamp_min(0.0)
        total_loss = total_loss + hinge[pair_mask].sum()
        total_pairs += int(pair_mask.sum().item())

    if total_pairs == 0:
        return _zero_term(pred)

    loss = total_loss / float(total_pairs)
    return {"loss": loss, "n_supervised": n_supervised, "n_pairs": total_pairs}


def pathway_evidence_regularizer(
    out: Mapping[str, Any],
    batch: Any,
    *,
    top_k: int = 20,
    target_overlap_weight: float = 1.0,
    crispr_overlap_weight: float = 0.5,
) -> Dict[str, Any]:
    """Soft penalty encouraging top-k attributed proteins to overlap with
    known drug targets and CRISPR-essential genes.

    Reads:
        out['protein_scores']          : (B, n_proteins) — attribution logits.
        batch.drug_target_mask         : (B, n_proteins) — binary 0/1 indicating
                                          known direct drug targets.
        batch.crispr_essential_mask    : (B, n_proteins) — binary 0/1 indicating
                                          CRISPR-essential genes.

    When neither mask is present, returns a small zero-gradient term so the
    loss is always finite.

    The penalty is ``1 - mean_overlap`` where ``mean_overlap`` is the
    soft-Jaccard between the top-k proteins (by attribution score) and the
    union of drug-target / CRISPR-essential masks. This encourages — but does
    not force — the model to attribute resistance to biologically plausible
    proteins.

    Returns
    -------
    dict
        ``{"loss": Tensor, "loss_target_overlap": Tensor,
        "loss_crispr_overlap": Tensor, "n_supervised": int}``.
    """
    scores = out.get("protein_scores")
    if scores is None:
        return _zero_term(out["z_traj"])

    device = scores.device
    dtype = scores.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    B, P = scores.shape

    drug_targets = _get_field(batch, "drug_target_mask")
    crispr_mask = _get_field(batch, "crispr_essential_mask")

    has_drug = drug_targets is not None
    has_crispr = crispr_mask is not None

    if not has_drug and not has_crispr:
        # No evidence masks in batch — return zero loss, always finite.
        return {
            "loss": zero,
            "loss_target_overlap": zero,
            "loss_crispr_overlap": zero,
            "n_supervised": 0,
        }

    # Soft top-k selection via sigmoid on scores (differentiable proxy).
    # Use a temperature-scaled sigmoid so that the top-k selection is smooth.
    # First, compute the k-th largest score per row as a soft threshold.
    k = min(top_k, P)
    # topk returns values in descending order.
    topk_vals, _ = scores.topk(k, dim=-1)                    # (B, k)
    threshold = topk_vals[:, -1:]                             # (B, 1)
    # Soft indicator: proteins above the threshold get weight ~1.
    soft_topk = torch.sigmoid(scores - threshold)             # (B, P)

    loss_target = zero
    loss_crispr = zero
    n_supervised = 0

    if has_drug:
        dt = drug_targets.to(device=device, dtype=dtype)
        # Mask NaN rows.
        dt_valid = torch.isfinite(dt).all(dim=-1)
        if dt_valid.sum() > 0:
            s = soft_topk[dt_valid]
            d = dt[dt_valid]
            # Soft overlap: sum(min(s, d)) / sum(max(s, d)) per row, averaged.
            intersection = torch.min(s, d).sum(dim=-1)
            union = torch.max(s, d).sum(dim=-1).clamp_min(1e-8)
            overlap = (intersection / union).mean()
            loss_target = 1.0 - overlap
            n_supervised += int(dt_valid.sum().item())

    if has_crispr:
        cm = crispr_mask.to(device=device, dtype=dtype)
        cm_valid = torch.isfinite(cm).all(dim=-1)
        if cm_valid.sum() > 0:
            s = soft_topk[cm_valid]
            c = cm[cm_valid]
            intersection = torch.min(s, c).sum(dim=-1)
            union = torch.max(s, c).sum(dim=-1).clamp_min(1e-8)
            overlap = (intersection / union).mean()
            loss_crispr = 1.0 - overlap
            n_supervised += int(cm_valid.sum().item())

    loss = target_overlap_weight * loss_target + crispr_overlap_weight * loss_crispr
    return {
        "loss": loss,
        "loss_target_overlap": loss_target if isinstance(loss_target, torch.Tensor) else zero,
        "loss_crispr_overlap": loss_crispr if isinstance(loss_crispr, torch.Tensor) else zero,
        "n_supervised": n_supervised,
    }


# ---------------------------------------------------------------------------
# Stage-level dispatcher
# ---------------------------------------------------------------------------


def _trajectory_term_from_dict(out: Mapping[str, Any], batch: Any) -> Dict[str, Any]:
    """Convenience adapter so the stage dispatcher can call the trajectory
    loss in the same ``(out, batch)`` shape as the others."""
    z_traj = out["z_traj"]
    fs = _get_field(batch, "future_state")
    if fs is None:
        return _zero_term(z_traj)
    fs = fs.to(device=z_traj.device, dtype=z_traj.dtype)
    if fs.ndim != 2 or fs.shape[0] != z_traj.shape[0]:
        return _zero_term(z_traj)
    d = z_traj.shape[-1]
    if fs.shape[-1] != d:
        fs = fs[..., :d]
    # Approximate per-row delta_t from the t_grid if it's available; else 1.
    t_grid = out.get("t_grid")
    if t_grid is not None and t_grid.numel() >= 1:
        dt_scalar = float(t_grid[-1].item() - t_grid[0].item()) or 1.0
    else:
        dt_scalar = 1.0
    delta_t = torch.full(
        (z_traj.shape[0],), dt_scalar, device=z_traj.device, dtype=z_traj.dtype
    )
    return canonical_trajectory_loss(z_traj, fs, delta_t)


def assemble_canonical_loss(
    stage: str,
    out: Mapping[str, Any],
    batch: Any,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Dispatch the canonical loss terms for ``stage`` and return their bundle.

    Parameters
    ----------
    stage :
        One of "E", "F", "G", "H". Unknown stages raise ``ValueError``.
    out :
        Canonical-dict output of ``MORTFM.forward``.
    batch :
        A :class:`MORTBatch` instance or equivalent mapping with
        ``event_time``, ``event_observed``, ``resistance_label``,
        ``future_state`` keys.
    weights :
        Optional per-term scalar weight map, e.g.
        ``{"survival": 1.0, "hitting": 0.5}``. Missing keys default to 1.0.

    Returns
    -------
    dict
        Bundle containing:
          * ``loss_total``           — weighted sum of every term that fired
            (zero scalar if nothing fired);
          * ``components``           — ``{term: sub_dict_returned_by_term}``;
          * ``per_term_n_supervised``— ``{term: int}``;
          * ``required_term``        — the strict-supervision gating term
            for this stage (or ``None`` for stages outside the strict policy).
    """
    if stage not in STAGE_LOSS_SPEC:
        raise ValueError(
            f"assemble_canonical_loss: stage {stage!r} is not a canonical "
            f"patient-level stage; expected one of {tuple(STAGE_LOSS_SPEC)}"
        )
    w = dict(weights or {})

    components: Dict[str, Dict[str, Any]] = {}
    per_term_n: Dict[str, int] = {}
    total: Optional[torch.Tensor] = None

    for term in STAGE_LOSS_SPEC[stage]:
        if term == "survival":
            sub = lens_survival_loss(out, batch)
        elif term == "hitting":
            sub = hitting_time_loss(out, batch)
        elif term == "hitting_nll":
            # v19 Phase 7: discrete-NLL of the empirical hitting CDF on the
            # SHARED canonical t_grid. Requires event_time + event_observed.
            sub = hitting_time_nll(
                out["hitting_cdf"],
                out["t_grid"],
                _get_field(batch, "event_time"),
                _get_field(batch, "event_observed"),
            ) if _get_field(batch, "event_time") is not None and _get_field(batch, "event_observed") is not None else _zero_term(out["hitting_cdf"])
        elif term == "surv_hit_consistency":
            # v19 Phase 7: directional invariant S(t) <= 1 - F_hit(t) + sum_c c_other(t).
            # Always finite (no batch supervision needed).
            cif_other = out.get("cif_per_event")
            sub = survival_hitting_consistency_loss(
                out["survival_curve"], out["hitting_cdf"],
                cif_other_events=cif_other,
            )
        elif term == "basin_transition":
            sub = basin_transition_loss(out, batch)
        elif term == "sde_path":
            sub = sde_path_loss(out, batch)
        elif term == "trajectory":
            sub = _trajectory_term_from_dict(out, batch)
        elif term == "pathway_attribution":
            sub = pathway_attribution_loss(out, batch)
        elif term == "pathway_evidence":
            sub = pathway_evidence_regularizer(out, batch)
        elif term == "counterfactual_consistency":
            sub = counterfactual_consistency_loss(out, batch)
        else:  # defensive — STAGE_LOSS_SPEC must agree with this dispatcher
            raise ValueError(f"assemble_canonical_loss: unknown term {term!r}")

        components[term] = sub
        per_term_n[term] = int(sub.get("n_supervised", 0))
        weight = float(w.get(term, 1.0))
        if weight != 0.0:
            contrib = weight * sub["loss"]
            total = contrib if total is None else total + contrib

    # Anchor the loss to *some* tensor even if every term was zero.
    if total is None:
        z = out["z_traj"]
        total = torch.zeros((), device=z.device, dtype=z.dtype)

    return {
        "loss_total": total,
        "components": components,
        "per_term_n_supervised": per_term_n,
        "required_term": REQUIRED_SUPERVISION_PER_STAGE.get(stage),
    }
