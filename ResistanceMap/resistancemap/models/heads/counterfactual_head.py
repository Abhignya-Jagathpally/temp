"""
resistancemap/models/heads/counterfactual_head.py
=================================================
Counterfactual intervention head.

Simulates "what if we blocked protein X / added drug Y / inhibited pathway Z?"
and reports a ranked list of interventions plus the predicted change in
resistance trajectory.

This module is more orchestration than learnable parameters — it holds a
reference to the dynamics module + the foundation encoder and *re-runs them*
under modified inputs. The model itself is not retrained per query.

Honest behaviour
----------------
* This head returns *predictions under counterfactual inputs*, not actual
  causal effects. For real causal validation we need perturbation experiments
  (CRISPR / drug screens) — see the causal-inference-auditor agent.
* The :meth:`rank_interventions` method should be paired with an oracle
  (CRISPR / drug screen) to assert the rankings agree with measured biology
  before publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from resistancemap.models.dynamics.trajectory_sampler import TrajectorySampler


@dataclass
class CounterfactualResult:
    intervention: Dict[str, Any]
    delta_resistance: float           # change in predicted resistance score
    delta_time_to_resistance: float   # change in expected event time (months)
    intervention_score: float         # composite ranking score


class CounterfactualInterventionHead(nn.Module):
    """Counterfactual intervention simulator.

    Parameters
    ----------
    sampler :
        Dynamics sampler used to roll out trajectories under perturbed inputs.
    risk_head :
        A module that turns a final-time latent into a scalar resistance score
        (typically :class:`DrugSpecificTrajectoryRiskHead` or the survival head).
    """

    def __init__(
        self,
        sampler: TrajectorySampler,
        risk_head: nn.Module,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.risk_head = risk_head

    @torch.no_grad()
    def simulate_node_blockade(
        self,
        z0: torch.Tensor,
        node_idx: int,
        time_grid: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
        zero_value: float = 0.0,
    ) -> torch.Tensor:
        """Set ``z0[:, node_idx] = zero_value`` and roll out the trajectory.

        Returns the final-time latent ``z(T)``.
        """
        z0_cf = z0.clone()
        z0_cf[:, node_idx] = zero_value
        out = self.sampler.sample_paths(z0_cf, time_grid, drug=drug, graph=graph)
        return out.mean_path[-1]

    @torch.no_grad()
    def simulate_drug_combination(
        self,
        z0: torch.Tensor,
        drug_tokens: Sequence[torch.Tensor],
        time_grid: torch.Tensor,
        *,
        graph: Any = None,
    ) -> torch.Tensor:
        """Roll out with the *sum* of drug tokens as the combination's effect."""
        combo_drug = torch.stack(list(drug_tokens), dim=0).sum(dim=0)
        out = self.sampler.sample_paths(z0, time_grid, drug=combo_drug, graph=graph)
        return out.mean_path[-1]

    @torch.no_grad()
    def rank_interventions(
        self,
        z0: torch.Tensor,
        candidate_nodes: Sequence[int],
        time_grid: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> List[CounterfactualResult]:
        """Score each candidate intervention by predicted reduction in risk."""
        # Baseline.
        out_baseline = self.sampler.sample_paths(z0, time_grid, drug=drug, graph=graph)
        z_T_base = out_baseline.mean_path[-1]
        risk_base = self.risk_head(z_T_base).squeeze(-1) if self.risk_head is not None else z_T_base.norm(dim=-1)
        # Reduce across batch to a scalar.
        risk_base_scalar = risk_base.mean()

        results: List[CounterfactualResult] = []
        for node_idx in candidate_nodes:
            z_T_cf = self.simulate_node_blockade(z0, node_idx, time_grid, drug=drug, graph=graph)
            risk_cf = self.risk_head(z_T_cf).squeeze(-1) if self.risk_head is not None else z_T_cf.norm(dim=-1)
            delta_r = float((risk_cf.mean() - risk_base_scalar).item())
            results.append(
                CounterfactualResult(
                    intervention={"kind": "node_blockade", "node": int(node_idx)},
                    delta_resistance=delta_r,
                    delta_time_to_resistance=0.0,   # filled by survival head if available
                    intervention_score=-delta_r,    # larger = better intervention
                )
            )
        results.sort(key=lambda r: r.intervention_score, reverse=True)
        return results
