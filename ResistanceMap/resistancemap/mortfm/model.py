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
# v18.3 — TrajectorySampler + v15 task heads live under the legacy namespace.
# They are exercised ONLY via forward_legacy() and via historical checkpoint
# loaders. The canonical v18 forward() uses the v16 LENS submodules instead.
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.trajectory_sampler import TrajectorySampler
from resistancemap.models.foundation_fusion import MultiOmicFoundationFusion
from resistancemap.legacy.v15_planned_mortfm.models.heads.counterfactual_head import CounterfactualInterventionHead
from resistancemap.legacy.v15_planned_mortfm.models.heads.drug_risk_head import DrugSpecificTrajectoryRiskHead
from resistancemap.legacy.v15_planned_mortfm.models.heads.pathway_route_head import PathwayRouteHead
from resistancemap.legacy.v15_planned_mortfm.models.heads.resistance_state_head import ResistanceStateHead
from resistancemap.legacy.v15_planned_mortfm.models.heads.time_to_resistance_head import TimeToResistanceHead
from resistancemap.legacy.v15_planned_mortfm.models.heads.trajectory_head import TrajectoryDistributionHead
from resistancemap.legacy.v15_planned_mortfm.models.heads.uncertainty_head import EvidentialUncertaintyHead
from resistancemap.mortfm.schemas import (
    FoundationState,
    MORTBatch,
    MORTFMConfig,
    TrajectoryPrediction,
)

# v17 — first-class import of the v16 LENS modules. The classic
# `forward()` method still uses the legacy TrajectorySampler+WaddingtonPotential
# path for backward compatibility with checkpoints saved before v16. The
# new `forward_lens()` method uses GraphEnergyResistanceSDE + CompetingRiskHead
# + ResistanceBasin + HittingTime — the v16 path actually trained on
# MMRF + BeatAML + scRNA. New trainers should call forward_lens().
from resistancemap.mortfm.trajectory import (  # noqa: E402
    GraphEnergyResistanceSDE,
    HittingTime,
    LatentToGraphProjector,
    ResistanceBasin,
)
from resistancemap.mortfm.survival import CompetingRiskHead  # noqa: E402

# v19 Phase 2+3 — causal / pathway attribution imports.
# Guarded by the has_intervention_graph flag set in __init__; never
# imported from legacy namespace.
from resistancemap.mortfm.causal.intervention_graph import InterventionGraph
from resistancemap.mortfm.causal.counterfactual_runner import CounterfactualRunner


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
        intervention_graph: Optional[InterventionGraph] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # v19 Phase 2+3 — optional causal attribution graph.
        # When wired, forward() can compute protein_scores, edge_scores,
        # pathway_scores, and counterfactual_rankings. When None,
        # those outputs are returned as None with zero overhead.
        self.intervention_graph = intervention_graph
        self.has_intervention_graph: bool = intervention_graph is not None

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
        # v17 — v16 LENS submodules. Always allocated so checkpoints
        # serialise both paths; forward_lens() chooses among them at call
        # time. Hyperparameters mirror what mortfm_train_lens_resistance.py
        # actually used (n_basins=5, n_time_grid=config.n_time_grid).
        # ------------------------------------------------------------------
        # v19 Phase 7 — share ONE canonical time grid between the SDE and
        # the competing-risk head. config.integration_time is in months;
        # n_time_grid is the canonical step count. The legacy SDE default
        # of integration_time=1.0 (unitless rollout) is replaced with the
        # config value so the SDE and survival head live on the same axis.
        from resistancemap.mortfm.trajectory.grid import CanonicalTimeGridConfig
        self._canonical_time_grid_config = CanonicalTimeGridConfig(
            t_max_months=float(config.integration_time),
            n_steps=int(config.n_time_grid),
        )

        # v19 Bug B19 — explicit d_graph contract.
        # GraphEnergyResistanceSDE consumes a (B, d_graph) embedding from
        # the GraphConditionedDrift's first-layer Linear. LatentToGraphProjector
        # must emit the SAME d_graph. Both default to 16 today; we pin them
        # to a single local constant so any future tweak (e.g. d_graph=32
        # for richer graph context) updates them in lock-step instead of
        # leaving them silently de-synced. The pin is asserted at the end
        # of __init__ to fail fast at construction time.
        _D_GRAPH = 16
        self.lens_sde = GraphEnergyResistanceSDE(
            d_latent=config.d_latent,
            d_graph=_D_GRAPH,
            d_drug=8,
            d_clinical=5,
            n_basins=5,
            n_mc_samples=4,
            time_grid_config=self._canonical_time_grid_config,
        )
        self.lens_basin = ResistanceBasin(
            d_latent=config.d_latent, n_basins=5,
            share_centroids_with=self.lens_sde.potential,
        )
        self.lens_hitting = HittingTime(
            self.lens_basin, resistant_basin_index=4, prob_threshold=0.3,
        )
        self.lens_graph_projector = LatentToGraphProjector(
            d_latent=config.d_latent, d_graph=_D_GRAPH,
        )
        # v19 Bug B19 verification: SDE consumes the projector's output, so
        # they MUST agree. The drift's first Linear has in_features =
        # d_latent + d_graph + d_drug + d_clinical; we read its actual
        # in_features and back out the SDE's runtime d_graph as a sanity
        # check, then equality-test against the projector's last Linear
        # out_features (also LayerNorm normalized_shape).
        _sde_in = self.lens_sde.drift.net[0].in_features
        _sde_d_graph = _sde_in - config.d_latent - 8 - 5
        _proj_out = self.lens_graph_projector.net[-1].out_features
        assert _sde_d_graph == _D_GRAPH == _proj_out, (
            f"v19 Bug B19 — GraphEnergyResistanceSDE expects d_graph="
            f"{_sde_d_graph} but LatentToGraphProjector emits d_graph="
            f"{_proj_out}. The shared local constant _D_GRAPH={_D_GRAPH} "
            f"must propagate to BOTH constructors."
        )
        self.lens_competing_risk = CompetingRiskHead(
            d_latent=config.d_latent,
            n_bins=config.survival_n_bins,
            event_names=["progression"],
            time_grid_config=self._canonical_time_grid_config,
        )

        # v19 Phase 2+3 — CounterfactualRunner (only usable when an
        # InterventionGraph is wired). Non-None self.counterfactual_runner
        # gates the causal outputs in forward().
        self.counterfactual_runner: Optional[CounterfactualRunner] = None
        if self.has_intervention_graph:
            self.counterfactual_runner = CounterfactualRunner(
                sde=self.lens_sde,
                basin=self.lens_basin,
                hitting=self.lens_hitting,
                intervention_graph=self.intervention_graph,
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

    def forward_foundation(self, batch: MORTBatch) -> FoundationState:
        """Lightweight encode-only path for stages A/B/C (foundation pretraining)."""
        return self.encode(batch)

    def forward_legacy(
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

    # ------------------------------------------------------------------
    # v19 Phase 2+3 helpers: pathway attribution + counterfactuals
    # ------------------------------------------------------------------

    def _compute_protein_scores(
        self,
        z0: torch.Tensor,
        graph_emb: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Per-protein attribution: gradient of soft resistance-basin
        probability w.r.t. graph_emb.

        The HittingTime CDF uses hard thresholding (argmax, cummax) and is
        not differentiable. Instead we use the *soft* basin probability of
        the designated resistant basin (index 4) at every SDE timepoint,
        averaged over MC samples and time steps. This is a differentiable
        proxy for "how much does each graph_emb dimension affect the
        probability of landing in the resistant basin".

        Returns (B, d_graph) or None if the backward pass yields no gradient.
        Uses ``torch.enable_grad()`` to ensure the local gradient tape works
        even when called from an outer ``torch.no_grad()`` context. All
        inputs are ``.detach()``-ed so gradients do not leak into the caller.
        """
        graph_emb_var = graph_emb.detach().requires_grad_(True)
        with torch.enable_grad():
            sde_out = self.lens_sde(
                z0.detach(), graph_emb_var, drug.detach(), clinical.detach(),
                return_samples=True,
            )
            # z_samples: (S, B, T, D)
            z_samples = sde_out["z_samples"]
            S, B, T, D = z_samples.shape
            # Soft basin probs via ResistanceBasin (differentiable softmax).
            flat = z_samples.reshape(S * B * T, D)
            probs = self.lens_basin(flat).reshape(S, B, T, -1)  # (S, B, T, K)
            # Resistant basin probability, averaged over samples and time.
            resist_idx = self.lens_hitting.resistant_basin_index
            p_resist = probs[..., resist_idx].mean(dim=(0, 2))  # (B,)
            scalar = p_resist.sum()
            scalar.backward()
        # (B, d_graph) — per-protein sensitivity
        return graph_emb_var.grad.detach() if graph_emb_var.grad is not None else None

    def _compute_edge_scores(
        self,
        protein_scores: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Per-edge attribution from InterventionGraph basis + protein_scores.

        Projects the (B, d_graph) protein_scores onto the edge basis to
        get a (B, n_edges) attribution. Returns None if intervention_graph
        has no loaded edge basis.
        """
        if (
            self.intervention_graph is None
            or self.intervention_graph._edge_basis is None
        ):
            return None
        # edge_basis: (n_edges, d_graph), protein_scores: (B, d_graph)
        edge_basis = self.intervention_graph._edge_basis.to(protein_scores.device)
        # (B, n_edges) = matmul of (B, d_graph) x (d_graph, n_edges)
        return torch.matmul(protein_scores, edge_basis.t())

    @staticmethod
    def _compute_pathway_scores_from_protein(
        protein_scores: torch.Tensor,
        n_pathways: int,
    ) -> torch.Tensor:
        """Aggregate protein-level scores into pathway-level scores.

        Uses a simple chunk-based aggregation: d_graph features are
        partitioned into ``n_pathways`` equal-width bins and the mean
        absolute attribution per bin gives a pathway-level score.

        Returns (B, n_pathways).
        """
        B, D = protein_scores.shape
        if n_pathways <= 0:
            n_pathways = 1
        # Pad D to be divisible by n_pathways for clean chunking.
        chunk = max(D // n_pathways, 1)
        scores = protein_scores.abs()
        pathway_list = []
        for i in range(n_pathways):
            start = i * chunk
            end = min(start + chunk, D)
            if start >= D:
                # More pathways than features — pad with zeros.
                pathway_list.append(scores.new_zeros(B))
            else:
                pathway_list.append(scores[:, start:end].mean(dim=-1))
        return torch.stack(pathway_list, dim=-1)  # (B, n_pathways)

    # ------------------------------------------------------------------
    # CANONICAL v18 forward — the SINGLE OFFICIAL patient-level path.
    # Calls the v16/v17 LENS modules directly. Returns a dict (not a
    # dataclass) to discourage drift back to the legacy TrajectoryPrediction
    # surface. forward_legacy() above remains for old-checkpoint compat.
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: MORTBatch,
        *,
        clinical: Optional[torch.Tensor] = None,
        drug: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        use_graph_projector: bool = True,
        compute_pathway_scores: bool = False,
    ) -> dict:
        """LENS-style forward: SDE rollout + competing-risk hazard + basin probs.

        Parameters
        ----------
        batch : MORTBatch (used only for encoding to z0)
        clinical : (B, 5) clinical covariates from encode_for_lens, or None
                   (then zero-filled). MUST match d_clinical=5.
        drug : (B, 8) drug context (one-hot-ish), or None.
        graph_emb : (B, 16) precomputed biological graph embedding, or
                    None — in which case use_graph_projector decides
                    whether to learn graph_emb = projector(z0) or feed zeros.
        compute_pathway_scores : if True, compute protein_scores,
                    edge_scores, pathway_scores, and counterfactual_rankings.
                    Default False to avoid extra backward-pass overhead
                    during training.

        Returns
        -------
        dict with keys:
            z0, z_traj, z_samples, t_grid, hazard, survival_curve,
            cif_per_event, basin_probs, hitting_cdf, hitting_mean_tau,
            hitting_frac_hit,
            protein_scores, edge_scores, pathway_scores,
            counterfactual_rankings
        The last four are None when compute_pathway_scores is False or
        when the required causal modules are not available.
        """
        state = self.encode(batch)
        z0 = state.z0
        B = z0.shape[0]
        if drug is None:
            drug = z0.new_zeros((B, 8))
        if clinical is None:
            clinical = z0.new_zeros((B, 5))
        if graph_emb is None:
            graph_emb = (
                self.lens_graph_projector(z0)
                if use_graph_projector
                else z0.new_zeros((B, 16))
            )
        sde_out = self.lens_sde(z0, graph_emb, drug, clinical, return_samples=True)
        z_final = sde_out["z_traj"][:, -1, :]
        head_out = self.lens_competing_risk(z_final)
        basin_probs = self.lens_basin.trajectory_probs(sde_out["z_traj"])
        hit_out = self.lens_hitting(sde_out["z_samples"], sde_out["t_grid"])

        # ----- v19 Phase 2+3: optional causal / pathway attribution -----
        protein_scores: Optional[torch.Tensor] = None
        edge_scores: Optional[torch.Tensor] = None
        pathway_scores: Optional[torch.Tensor] = None
        counterfactual_rankings = None

        if compute_pathway_scores:
            # Protein-level attribution via gradient of hitting_cdf w.r.t.
            # graph_emb. Always available (uses the projector output).
            protein_scores = self._compute_protein_scores(
                z0, graph_emb, drug, clinical,
            )

            if protein_scores is not None:
                # Edge-level attribution (requires InterventionGraph).
                if self.has_intervention_graph:
                    edge_scores = self._compute_edge_scores(protein_scores)

                # Pathway-level aggregation (always available from protein_scores).
                pathway_scores = self._compute_pathway_scores_from_protein(
                    protein_scores, n_pathways=50,
                )

            # Counterfactual rankings (requires CounterfactualRunner + graph).
            if self.counterfactual_runner is not None and self.has_intervention_graph:
                edge_ids = self.intervention_graph.list_edges(limit=50)
                if edge_ids:
                    cf_df = self.counterfactual_runner.run(
                        z0, drug, clinical, edge_ids=edge_ids,
                    )
                    # Convert to list of (edge_id, delta_resistance) tuples,
                    # ranked by absolute delta descending.
                    counterfactual_rankings = [
                        (row.edge_id, float(row.delta))
                        for row in cf_df.itertuples(index=False)
                    ]

        return {
            "z0": z0,
            "z_traj": sde_out["z_traj"],
            "z_samples": sde_out["z_samples"],
            "t_grid": sde_out["t_grid"],
            "hazard": head_out["hazard"],
            "survival_curve": head_out["survival_curve"],
            "cif_per_event": head_out["cif_per_event"],
            "basin_probs": basin_probs,
            "hitting_cdf": hit_out["cdf"],
            "hitting_mean_tau": hit_out["mean_tau"],
            "hitting_frac_hit": hit_out["frac_hit"],
            # v19 Phase 2+3 — causal / pathway attribution (None when disabled)
            "protein_scores": protein_scores,
            "edge_scores": edge_scores,
            "pathway_scores": pathway_scores,
            "counterfactual_rankings": counterfactual_rankings,
        }
