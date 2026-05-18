"""
resistancemap/mortfm/causal/counterfactual_runner.py
=====================================================
Per-edge counterfactual effect on resistance probability.

For each candidate edge e in the biological graph, compute

    Δ_e = E[ P(τ^R ≤ T) | do(e=1) ] − E[ P(τ^R ≤ T) | do(e=0) ]

by:
  1. Producing the baseline graph_emb (no intervention).
  2. Applying the InterventionGraph's knockout to get graph_emb_minus.
  3. Applying the InterventionGraph's amplify to get graph_emb_plus.
  4. Rolling the SDE under each variant.
  5. Reading P(τ^R ≤ T) from HittingTime and averaging over patients.

Δ_e is positive for edges whose presence accelerates resistance; negative
for protective edges. The output is a tidy DataFrame ranking edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd
import torch

from resistancemap.mortfm.causal.intervention_graph import (
    EdgeIntervention, InterventionGraph,
)
from resistancemap.mortfm.trajectory import (
    GraphEnergyResistanceSDE, HittingTime, ResistanceBasin,
)

logger = logging.getLogger(__name__)


@dataclass
class EdgeEffect:
    edge_id: str
    p_resistance_baseline: float
    p_resistance_under_knockout: float
    p_resistance_under_amplify: float
    delta: float
    note: Optional[str] = None


class CounterfactualRunner:
    """Score each edge's effect on resistance probability."""

    def __init__(
        self,
        sde: GraphEnergyResistanceSDE,
        basin: ResistanceBasin,
        hitting: HittingTime,
        intervention_graph: InterventionGraph,
    ) -> None:
        self.sde = sde
        self.basin = basin
        self.hitting = hitting
        self.graph = intervention_graph

    @torch.no_grad()
    def _roll(self, z0: torch.Tensor, graph_emb: torch.Tensor,
              drug: torch.Tensor, clinical: torch.Tensor) -> float:
        out = self.sde(z0, graph_emb, drug, clinical, return_samples=True)
        h = self.hitting(out["z_samples"], out["t_grid"])
        # P(tau^R <= T_max) — last column of cdf
        return float(h["cdf"][:, -1].mean().item())

    @torch.no_grad()
    def run(
        self,
        z0: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
        edge_ids: Sequence[str],
        amplify_factor: float = 2.0,
    ) -> pd.DataFrame:
        B = z0.shape[0]
        base_emb = self.graph.base_embedding(B).to(z0.device)
        p_base = self._roll(z0, base_emb, drug, clinical)
        records: List[EdgeEffect] = []
        for eid in edge_ids:
            try:
                ko_emb = self.graph.intervene(base_emb, [EdgeIntervention(eid, "knockout")])
                amp_emb = self.graph.intervene(
                    base_emb, [EdgeIntervention(eid, "amplify", factor=amplify_factor)]
                )
                p_ko = self._roll(z0, ko_emb, drug, clinical)
                p_amp = self._roll(z0, amp_emb, drug, clinical)
                records.append(EdgeEffect(
                    edge_id=eid, p_resistance_baseline=p_base,
                    p_resistance_under_knockout=p_ko,
                    p_resistance_under_amplify=p_amp,
                    delta=p_amp - p_ko,
                ))
            except KeyError as exc:
                records.append(EdgeEffect(
                    edge_id=eid, p_resistance_baseline=p_base,
                    p_resistance_under_knockout=float("nan"),
                    p_resistance_under_amplify=float("nan"),
                    delta=float("nan"),
                    note=str(exc),
                ))
        df = pd.DataFrame([e.__dict__ for e in records])
        df = df.sort_values("delta", ascending=False, na_position="last")
        logger.info("Counterfactual runner scored %d edges", len(df))
        return df
