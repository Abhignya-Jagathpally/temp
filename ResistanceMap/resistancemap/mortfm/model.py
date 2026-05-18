"""
resistancemap/mortfm/model.py
=============================
End-to-end MORT-FM model.

Composition (encoder -> dynamics -> heads):

    1. ModalityTokenizer       — modality-specific encoders -> tokens
    2. MultiOmicFoundationFusion — cross-modal transformer -> z0
    3. TrajectorySampler        — graph-conditioned ODE / SDE / Jump-SDE
    4. Heads:
        - ResistanceStateHead
        - TimeToResistanceHead (Cox or discrete-time)
        - TrajectoryDistributionHead
        - PathwayRouteHead
        - DrugSpecificTrajectoryRiskHead
        - EvidentialUncertaintyHead
        - CounterfactualInterventionHead

The model exposes a single ``forward(batch) -> TrajectoryPrediction`` API.
The trainer is responsible for selecting which heads contribute to the loss
at each training stage.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from resistancemap.landscape.potential import WaddingtonPotential
from resistancemap.models.dynamics.trajectory_sampler import TrajectorySampler
from resistancemap.models.foundation_fusion import MultiOmicFoundationFusion
from resistancemap.models.heads.counterfactual_head import CounterfactualInterventionHead
from resistancemap.models.heads.drug_risk_head import DrugSpecificTrajectoryRiskHead
from resistancemap.models.heads.pathway_route_head import PathwayRouteHead
from resistancemap.models.heads.resistance_state_head import ResistanceStateHead
from resistancemap.models.heads.time_to_resistance_head import TimeToResistanceHead
from resistancemap.models.heads.trajectory_head import TrajectoryDistributionHead
from resistancemap.models.heads.uncertainty_head import EvidentialUncertaintyHead
from resistancemap.mortfm.schemas import (
    FoundationState,
    MORTBatch,
    MORTFMConfig,
    TrajectoryPrediction,
)


class MORTFM(nn.Module):
    """End-to-end MORT-FM."""

    def __init__(
        self,
        config: MORTFMConfig,
        *,
        n_pathway_proteins: int = 500,
        n_pathway_edges: int = 0,
        n_pathways: int = 50,
        n_drug_candidates: int = 11,
        n_resistance_states: int = 4,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        # Encoder + fusion.
        self.fusion = MultiOmicFoundationFusion(config, return_attention=return_attention)

        # The drug *token* coming out of the foundation fusion has width
        # ``d_token`` (every modality token does). The dynamics + potential
        # consume that token directly, so they must size their drug-input
        # dimension to ``d_token``, NOT ``drug_embed_dim`` (which is an
        # internal hyperparameter of :class:`DrugEncoder`'s identity embedding).
        self._dynamics_drug_dim = config.d_token if config.use_drug else 0

        # Waddington potential — owned by the model so heads can call it.
        self.potential = (
            WaddingtonPotential(
                d_latent=config.d_latent,
                n_attractors=config.n_attractors,
                drug_dim=self._dynamics_drug_dim,
                hidden=config.potential_hidden,
            )
            if config.use_potential
            else None
        )

        # Dynamics.
        self.sampler = TrajectorySampler(
            d_latent=config.d_latent,
            kind=config.dynamics_kind,
            drug_dim=self._dynamics_drug_dim,
            n_graph_nodes=0,  # set by trainer when graph is wired
            potential=self.potential,
        )

        # Heads.
        self.state_head = ResistanceStateHead(config.d_latent, n_states=n_resistance_states)
        self.survival_head = TimeToResistanceHead(
            config.d_latent, kind=config.survival_kind, n_bins=config.survival_n_bins
        )
        self.trajectory_head = TrajectoryDistributionHead(
            config.d_latent, d_output=config.d_latent
        )
        self.pathway_head = PathwayRouteHead(
            config.d_latent,
            n_proteins=n_pathway_proteins,
            n_edges=n_pathway_edges,
            n_pathways=n_pathways,
        )
        self.drug_risk_head = DrugSpecificTrajectoryRiskHead(
            config.d_latent, n_drugs=n_drug_candidates, drug_embed_dim=config.drug_embed_dim
        )
        self.uncertainty_head = EvidentialUncertaintyHead(
            config.d_latent, n_classes=n_resistance_states
        )
        self.counterfactual_head = CounterfactualInterventionHead(
            sampler=self.sampler, risk_head=self.drug_risk_head
        )

    # ------------------------------------------------------------------
    # Sub-graph forward passes (used by the trainer for stage selection).
    # ------------------------------------------------------------------

    def encode(self, batch: MORTBatch) -> FoundationState:
        return self.fusion(batch)

    def rollout(
        self,
        state: FoundationState,
        time_grid: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
        n_samples: int = 16,
    ):
        return self.sampler.sample_paths(state.z0, time_grid, drug=drug, graph=graph, n_samples=n_samples)

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: MORTBatch,
        *,
        time_grid: Optional[torch.Tensor] = None,
        n_traj_samples: int = 16,
        graph: Any = None,
        compute_counterfactuals: bool = False,
    ) -> TrajectoryPrediction:
        state = self.encode(batch)
        if time_grid is None:
            time_grid = torch.linspace(
                0.0, self.config.integration_time, self.config.n_time_grid, device=state.z0.device,
            )

        drug_token = state.modality_embeddings.get("drug")
        traj = self.rollout(state, time_grid, drug=drug_token, graph=graph, n_samples=n_traj_samples)

        z_T = traj.mean_path[-1]
        state_logits = self.state_head(z_T)
        survival_out = self.survival_head(z_T)
        traj_mean, traj_log_var = self.trajectory_head(traj.mean_path)
        protein_scores, edge_scores, pathway_scores = self.pathway_head(z_T)
        drug_risk = self.drug_risk_head(z_T)  # (N, n_drug_candidates)

        prediction = TrajectoryPrediction(
            z_path=traj.mean_path,
            z_samples=traj.samples,
            resistance_state_logits=state_logits,
            hazard=survival_out if self.config.survival_kind == "cox" else None,
            survival_curve=(
                self.survival_head.head.survival_curve(survival_out)
                if self.config.survival_kind == "discrete"
                else None
            ),
            trajectory_mean=traj_mean,
            trajectory_cov=traj_log_var.exp(),
            pathway_protein_scores=protein_scores,
            pathway_edge_scores=edge_scores,
            drug_specific_risk=drug_risk,
            counterfactual_rankings=None,
            uncertainty={"trajectory_std": traj.std_path} if traj.std_path is not None else None,
        )

        if compute_counterfactuals:
            results = self.counterfactual_head.rank_interventions(
                state.z0, candidate_nodes=list(range(min(8, state.z0.shape[-1]))),
                time_grid=time_grid, drug=drug_token, graph=graph,
            )
            prediction.counterfactual_rankings = [[r.__dict__ for r in results]]
        return prediction
