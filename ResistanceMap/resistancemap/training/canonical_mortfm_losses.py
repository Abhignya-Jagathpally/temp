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
    "assemble_canonical_loss",
    "STAGE_LOSS_SPEC",
    "REQUIRED_SUPERVISION_PER_STAGE",
]


# ---------------------------------------------------------------------------
# Stage spec — which losses fire in each canonical patient-level stage.
# ---------------------------------------------------------------------------

#: Maps each patient-level stage to the canonical loss terms it dispatches.
#: Stages outside this dict are foundation stages and use the legacy losses.
STAGE_LOSS_SPEC: Dict[str, tuple] = {
    "E": ("trajectory", "sde_path", "basin_transition"),
    "F": ("survival", "hitting"),
    "G": ("basin_transition",),
    "H": ("survival",),
}


#: Which loss terms must have ``n_supervised > 0`` for the stage to count as
#: supervised under the v18.2 strict policy. A stage may dispatch *more*
#: losses than it requires; only the listed terms gate the strict check.
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
        elif term == "basin_transition":
            sub = basin_transition_loss(out, batch)
        elif term == "sde_path":
            sub = sde_path_loss(out, batch)
        elif term == "trajectory":
            sub = _trajectory_term_from_dict(out, batch)
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
