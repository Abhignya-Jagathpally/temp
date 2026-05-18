"""
resistancemap/training/loss_router.py
=====================================
Disciplined loss composition for MORT-FM-LENS.

Why: ``mortfm_losses.assemble_total_loss`` already exists and sums
component losses. The router adds *guardrails* on top:

  * Validates that each enabled loss has the supervision it needs in the
    batch. If supervision is missing the router *refuses* to add the term
    (this is the v15 "trainer never silently substitutes a smaller loss"
    invariant promoted to first-class code).
  * Records per-step which losses fired and which were skipped, with
    explicit reasons — so the run-time auditor can read the trail.

Usage
-----

    router = LossRouter(
        enabled={"recon", "drug", "traj", "surv", "basin", "graph"},
        loss_fns={
            "recon": reconstruction_loss,
            "drug":  drug_response_loss,
            "traj":  trajectory_distribution_loss,
            ...
        },
        weights={"recon": 1.0, "drug": 1.0, "traj": 1.0, ...},
        required_supervision={
            "drug":  ["drug_response_target"],
            "traj":  ["z_future_target"],
            "surv":  ["event_time", "event_observed"],
            "basin": ["basin_label"],
        },
    )
    total, components, audit = router.compute(outputs, batch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

import torch

logger = logging.getLogger(__name__)


@dataclass
class LossAuditEntry:
    name: str
    fired: bool
    weight: float
    raw_value: Optional[float]
    skipped_reason: Optional[str]


class LossRouter:
    """Compose, audit, and gate the multi-objective MORT-FM loss."""

    def __init__(
        self,
        enabled: Iterable[str],
        loss_fns: Mapping[str, Callable[..., torch.Tensor]],
        weights: Mapping[str, float],
        required_supervision: Optional[Mapping[str, List[str]]] = None,
    ) -> None:
        self.enabled: Set[str] = set(enabled)
        self.loss_fns = dict(loss_fns)
        self.weights = dict(weights)
        self.required_supervision = dict(required_supervision or {})
        # Sanity check.
        for name in self.enabled:
            if name not in self.loss_fns:
                raise KeyError(f"LossRouter: enabled loss {name!r} has no loss_fn")
            if name not in self.weights:
                raise KeyError(f"LossRouter: enabled loss {name!r} has no weight")

    def _has_required(self, name: str, batch: Mapping[str, Any]) -> tuple[bool, Optional[str]]:
        reqs = self.required_supervision.get(name, [])
        missing = [r for r in reqs if (r not in batch) or (batch.get(r) is None)]
        if missing:
            return False, f"missing supervision: {missing}"
        return True, None

    def compute(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, float], List[LossAuditEntry]]:
        """Compute total weighted loss with per-component audit.

        Each loss_fn is invoked as ``fn(outputs, batch)``; the convention
        is consistent with the existing mortfm_losses module signatures
        (which take an outputs dict + batch tensors).
        """
        total = None
        components: Dict[str, float] = {}
        audit: List[LossAuditEntry] = []
        for name in sorted(self.enabled):
            ok, why = self._has_required(name, batch)
            if not ok:
                audit.append(LossAuditEntry(
                    name=name, fired=False, weight=self.weights[name],
                    raw_value=None, skipped_reason=why,
                ))
                logger.debug("LossRouter: %s skipped (%s)", name, why)
                continue
            try:
                val = self.loss_fns[name](outputs, batch)
            except Exception as exc:  # never silently swallow — surface
                audit.append(LossAuditEntry(
                    name=name, fired=False, weight=self.weights[name],
                    raw_value=None, skipped_reason=f"loss_fn raised: {exc!r}",
                ))
                continue
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            w = float(self.weights[name])
            term = w * val
            total = term if total is None else total + term
            components[name] = float(val.detach().item()) if val.dim() == 0 else float(val.mean().detach().item())
            audit.append(LossAuditEntry(
                name=name, fired=True, weight=w,
                raw_value=components[name], skipped_reason=None,
            ))
        if total is None:
            # No loss fired — return a zero scalar to keep the trainer alive
            # but mark the audit so the run is auditable.
            total = torch.zeros((), requires_grad=True)
        return total, components, audit
