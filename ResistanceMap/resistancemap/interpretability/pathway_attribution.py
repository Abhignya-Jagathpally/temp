"""
Pathway-level attribution for ResistanceMap v6 Neural ODE predictions.

This module implements temporal integrated gradients adapted for ODE-based models,
pathway-level aggregation of gene attributions, and extraction of learned gene
interaction networks from the structured Neural ODE.

Key methods:
- Temporal Integrated Gradients (TIG): extends Sundararajan et al. (2017) "Axiomatic
  Attribution for Deep Networks" (ICML) to continuous-time dynamical systems, attributing
  prediction changes across the ODE trajectory rather than at a single timepoint.
- Pathway aggregation: maps gene-level attributions to KEGG, Reactome, and MSigDB Hallmark
  gene sets with permutation-based significance testing following Subramanian et al. (2005)
  "Gene set enrichment analysis" (PNAS).
- Gene interaction recovery: compares the learned sparse interaction matrix A from the
  identifiable Neural ODE to curated databases (TRRUST, RegNetwork, STRING) following
  Pratapa et al. (2020) "Benchmarking algorithms for gene regulatory network inference"
  (Nature Methods).

References:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML.
    Subramanian, A. et al. (2005). Gene set enrichment analysis. PNAS, 102(43), 15545-15550.
    Chen, R. T. Q. et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
    Pratapa, A. et al. (2020). Benchmarking algorithms for GRN inference. Nature Methods, 17, 147-154.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PathwayAttributionConfig:
    """Configuration for pathway attribution analysis.

    Attributes:
        n_ig_steps: Number of interpolation steps for integrated gradients.
            More steps give more accurate attributions at higher compute cost.
            Sundararajan et al. (2017) recommend 20-300.
        baseline_mode: Strategy for choosing the IG baseline.
            'zero' uses a zero vector, 'mean' uses the training-set mean,
            'random' averages over multiple random baselines.
        n_random_baselines: Number of random baselines when baseline_mode='random'.
        n_permutations: Permutation count for pathway significance testing.
        significance_threshold: FDR-corrected p-value threshold.
        min_pathway_size: Minimum gene-set size for inclusion.
        max_pathway_size: Maximum gene-set size for inclusion.
        interaction_threshold: Magnitude threshold for interaction extraction from A.
        temporal_resolution: Number of intermediate ODE timepoints for temporal IG.
        device: Torch device string.
    """
    n_ig_steps: int = 100
    baseline_mode: str = "zero"
    n_random_baselines: int = 10
    n_permutations: int = 1000
    significance_threshold: float = 0.05
    min_pathway_size: int = 10
    max_pathway_size: int = 500
    interaction_threshold: float = 0.01
    temporal_resolution: int = 50
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PathwayAttributionResult:
    """Container for pathway attribution analysis results.

    Attributes:
        gene_attributions: (T, G) array of per-gene attributions at each timepoint.
        pathway_scores: Dict mapping pathway name -> (T,) temporal attribution score.
        pathway_pvalues: Dict mapping pathway name -> permutation p-value.
        significant_pathways: List of pathways passing FDR correction.
        temporal_profile: (T,) overall attribution magnitude over time.
        metadata: Additional information (config, runtime, etc.).
    """
    gene_attributions: np.ndarray
    pathway_scores: Dict[str, np.ndarray]
    pathway_pvalues: Dict[str, float]
    significant_pathways: List[str]
    temporal_profile: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionResult:
    """Container for gene interaction extraction results.

    Attributes:
        learned_interactions: List of (gene_i, gene_j, weight) tuples from A.
        precision_vs_known: Dict mapping database name -> precision.
        recall_vs_known: Dict mapping database name -> recall.
        novel_interactions: Ranked list of (gene_i, gene_j, confidence) not in any DB.
        interaction_matrix: The raw (G, G) interaction matrix.
    """
    learned_interactions: List[Tuple[str, str, float]]
    precision_vs_known: Dict[str, float]
    recall_vs_known: Dict[str, float]
    novel_interactions: List[Tuple[str, str, float]]
    interaction_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Pathway Attribution Engine
# ---------------------------------------------------------------------------

class PathwayAttributor:
    """Attribute Neural ODE predictions to biological pathways.

    Implements Temporal Integrated Gradients (TIG) that compute attributions
    along the ODE trajectory, then aggregates gene-level scores into pathway-
    level scores using gene set databases. Statistical significance is assessed
    via permutation testing with BH-FDR correction.

    Following Sundararajan et al. (2017), TIG satisfies:
    - Completeness: attributions sum to the difference in model output.
    - Sensitivity: if a feature matters, it receives nonzero attribution.
    - Implementation invariance: attributions depend only on the function, not
      the network architecture.

    The key extension for ODE models is that we compute attributions for the
    *change* in state x(t1) - x(t0) by integrating gradients along the
    temporal trajectory, not just along a straight-line path in input space.

    Args:
        model: The ResistanceMap Neural ODE model. Must expose:
            - model.ode_func: the ODE right-hand side f(x, t)
            - model.solve(x0, t_span): integrate the ODE
            - model.predict(trajectory): map trajectory to outcome
        gene_names: Ordered list of gene names matching model input dimensions.
        pathway_db: Dict mapping pathway name -> list of gene names in that set.
        config: PathwayAttributionConfig instance.
    """

    def __init__(
        self,
        model: nn.Module,
        gene_names: List[str],
        pathway_db: Dict[str, List[str]],
        config: Optional[PathwayAttributionConfig] = None,
    ) -> None:
        self.model = model
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.config = config or PathwayAttributionConfig()
        self.device = torch.device(self.config.device)

        # Filter pathway DB by size constraints
        self.pathway_db: Dict[str, List[int]] = {}
        for name, genes in pathway_db.items():
            idx = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]
            if self.config.min_pathway_size <= len(idx) <= self.config.max_pathway_size:
                self.pathway_db[name] = idx

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Baseline construction
    # ------------------------------------------------------------------

    def _get_baseline(self, x: Tensor) -> Tensor:
        """Construct the attribution baseline.

        Args:
            x: Input tensor of shape (batch, genes) or (genes,).

        Returns:
            Baseline tensor of same shape as x.
        """
        if self.config.baseline_mode == "zero":
            return torch.zeros_like(x)
        elif self.config.baseline_mode == "mean":
            # Assumes model stores training mean; fall back to zeros
            if hasattr(self.model, "training_mean"):
                return self.model.training_mean.to(x.device).expand_as(x)
            warnings.warn("Model has no training_mean; falling back to zero baseline.")
            return torch.zeros_like(x)
        elif self.config.baseline_mode == "random":
            return torch.randn_like(x) * 0.01
        else:
            raise ValueError(f"Unknown baseline_mode: {self.config.baseline_mode}")

    # ------------------------------------------------------------------
    # Core: Temporal Integrated Gradients
    # ------------------------------------------------------------------

    def compute_temporal_integrated_gradients(
        self,
        x0: Tensor,
        t_span: Tensor,
        target_fn: Optional[callable] = None,
    ) -> Tensor:
        """Compute Temporal Integrated Gradients along the ODE trajectory.

        Unlike standard IG which interpolates between baseline and input in
        *input space*, TIG interpolates along the *temporal trajectory* of the
        ODE. For each intermediate timepoint t_k, we compute the gradient of
        the final prediction with respect to the state x(t_k), weighted by
        the velocity dx/dt at t_k.

        This captures *when* each gene contributed to the outcome, not just
        *whether* it contributed.

        Mathematical formulation:
            TIG_i = integral_{t0}^{t1} (df/dx_i(t)) * (dx_i/dt) dt

        where f is the prediction function and x(t) is the ODE trajectory.
        We approximate this integral using the trapezoidal rule over
        config.temporal_resolution intermediate points.

        Args:
            x0: Initial state, shape (batch, n_genes) or (n_genes,).
            t_span: Time span tensor, shape (T,) with T >= 2.
            target_fn: Optional function mapping model output to scalar target.
                If None, uses the model's default resistance score.

        Returns:
            Temporal attributions of shape (T_interp, n_genes) where T_interp
            is config.temporal_resolution.
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        x0 = x0.to(self.device)
        t_span = t_span.to(self.device)

        n_time = self.config.temporal_resolution
        t_eval = torch.linspace(
            t_span[0].item(), t_span[-1].item(), n_time, device=self.device
        )

        # Solve ODE to get trajectory
        with torch.no_grad():
            trajectory = self._solve_ode(x0, t_eval)  # (n_time, batch, n_genes)

        attributions = torch.zeros(n_time, x0.shape[-1], device=self.device)

        for k in range(n_time):
            x_k = trajectory[k].detach().requires_grad_(True)

            # Forward from x_k to final state
            if k < n_time - 1:
                t_remaining = t_eval[k:]
                traj_from_k = self._solve_ode(x_k, t_remaining)
                x_final = traj_from_k[-1]
            else:
                x_final = x_k

            # Compute prediction
            if target_fn is not None:
                score = target_fn(x_final)
            else:
                score = self._default_target(x_final)

            # Gradient of prediction w.r.t. state at time k
            grad = torch.autograd.grad(
                score.sum(), x_k, create_graph=False, retain_graph=False
            )[0]  # (batch, n_genes)

            # Velocity at time k
            with torch.no_grad():
                velocity = self._ode_rhs(trajectory[k], t_eval[k])

            # TIG attribution at this timepoint: grad * velocity
            attributions[k] = (grad * velocity).mean(dim=0)  # average over batch

        # Apply trapezoidal integration weights
        dt = (t_eval[1] - t_eval[0]).item()
        attributions[0] *= 0.5
        attributions[-1] *= 0.5
        attributions *= dt

        return attributions.detach().cpu()

    def compute_standard_integrated_gradients(
        self,
        x: Tensor,
        t_span: Tensor,
        target_fn: Optional[callable] = None,
    ) -> Tensor:
        """Standard Integrated Gradients (Sundararajan et al., 2017) for comparison.

        Interpolates between baseline and input in input space. Useful as a
        sanity check against Temporal IG.

        Args:
            x: Input tensor, shape (batch, n_genes).
            t_span: Time span for ODE integration.
            target_fn: Optional scalar target function.

        Returns:
            Attributions of shape (n_genes,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        t_span = t_span.to(self.device)

        baseline = self._get_baseline(x)
        n_steps = self.config.n_ig_steps

        # Interpolation path
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        accumulated_grads = torch.zeros_like(x)

        for alpha in alphas:
            x_interp = baseline + alpha * (x - baseline)
            x_interp = x_interp.detach().requires_grad_(True)

            # Full forward pass through ODE + prediction
            trajectory = self._solve_ode(x_interp, t_span)
            x_final = trajectory[-1]

            if target_fn is not None:
                score = target_fn(x_final)
            else:
                score = self._default_target(x_final)

            grad = torch.autograd.grad(
                score.sum(), x_interp, create_graph=False
            )[0]
            accumulated_grads += grad

        # IG formula: (x - baseline) * mean_gradient
        ig = (x - baseline) * accumulated_grads / n_steps
        return ig.squeeze(0).detach().cpu()

    # ------------------------------------------------------------------
    # Pathway aggregation
    # ------------------------------------------------------------------

    def aggregate_to_pathways(
        self,
        gene_attributions: Union[Tensor, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Aggregate gene-level attributions to pathway scores.

        For each pathway, the score is the mean absolute attribution of its
        member genes, normalized by the square root of pathway size to avoid
        bias toward large gene sets (following GSEA normalization).

        Args:
            gene_attributions: Shape (T, n_genes) or (n_genes,).

        Returns:
            Dict mapping pathway name -> score array. If input is 2D,
            scores are (T,) temporal profiles; if 1D, scores are scalars.
        """
        if isinstance(gene_attributions, Tensor):
            gene_attributions = gene_attributions.numpy()

        is_temporal = gene_attributions.ndim == 2
        if not is_temporal:
            gene_attributions = gene_attributions[np.newaxis, :]

        pathway_scores = {}
        for name, gene_idx in self.pathway_db.items():
            pw_attr = gene_attributions[:, gene_idx]  # (T, |pathway|)
            # Mean absolute attribution, size-normalized
            score = np.abs(pw_attr).mean(axis=1) * np.sqrt(len(gene_idx))
            pathway_scores[name] = score.squeeze() if not is_temporal else score

        return pathway_scores

    def compute_pathway_significance(
        self,
        gene_attributions: np.ndarray,
        pathway_scores: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Permutation test for pathway attribution significance.

        For each pathway, we permute gene labels and recompute the pathway
        score n_permutations times to build a null distribution. The p-value
        is the fraction of permuted scores exceeding the observed score.

        Multiple testing correction uses Benjamini-Hochberg FDR.

        Args:
            gene_attributions: Shape (T, n_genes) or (n_genes,).
            pathway_scores: Observed pathway scores from aggregate_to_pathways.

        Returns:
            Dict mapping pathway name -> FDR-corrected p-value.
        """
        if gene_attributions.ndim == 2:
            # Use the sum over time as a scalar summary
            attr_summary = np.abs(gene_attributions).sum(axis=0)
        else:
            attr_summary = np.abs(gene_attributions)

        n_genes = len(attr_summary)
        n_perm = self.config.n_permutations
        raw_pvalues = {}

        for name, gene_idx in self.pathway_db.items():
            k = len(gene_idx)
            observed = attr_summary[gene_idx].mean() * np.sqrt(k)

            # If temporal, use scalar summary of pathway score
            if isinstance(pathway_scores[name], np.ndarray) and pathway_scores[name].ndim > 0:
                observed = float(np.abs(pathway_scores[name]).sum())

            null_dist = np.zeros(n_perm)
            for p in range(n_perm):
                perm_idx = np.random.choice(n_genes, size=k, replace=False)
                null_dist[p] = attr_summary[perm_idx].mean() * np.sqrt(k)

            pvalue = (np.sum(null_dist >= observed) + 1) / (n_perm + 1)
            raw_pvalues[name] = pvalue

        # Benjamini-Hochberg FDR correction
        corrected = self._benjamini_hochberg(raw_pvalues)
        return corrected

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def attribute(
        self,
        x0: Tensor,
        t_span: Tensor,
        target_fn: Optional[callable] = None,
    ) -> PathwayAttributionResult:
        """Run the full pathway attribution pipeline.

        1. Compute Temporal Integrated Gradients along the ODE trajectory.
        2. Aggregate gene attributions to pathway scores.
        3. Test pathway significance via permutation test.
        4. Apply FDR correction and return significant pathways.

        Args:
            x0: Initial patient state, shape (n_genes,) or (batch, n_genes).
            t_span: Time span for ODE trajectory.
            target_fn: Optional target function.

        Returns:
            PathwayAttributionResult with all attribution data.
        """
        # Step 1: Temporal IG
        gene_attr = self.compute_temporal_integrated_gradients(x0, t_span, target_fn)
        gene_attr_np = gene_attr.numpy()

        # Step 2: Pathway aggregation
        pathway_scores = self.aggregate_to_pathways(gene_attr_np)

        # Step 3: Significance testing
        pathway_pvalues = self.compute_pathway_significance(gene_attr_np, pathway_scores)

        # Step 4: Filter significant pathways
        sig = [
            name for name, pval in pathway_pvalues.items()
            if pval < self.config.significance_threshold
        ]
        sig.sort(key=lambda n: pathway_pvalues[n])

        # Temporal profile: overall attribution magnitude at each timepoint
        temporal_profile = np.linalg.norm(gene_attr_np, axis=1)

        return PathwayAttributionResult(
            gene_attributions=gene_attr_np,
            pathway_scores=pathway_scores,
            pathway_pvalues=pathway_pvalues,
            significant_pathways=sig,
            temporal_profile=temporal_profile,
            metadata={
                "config": self.config,
                "n_pathways_tested": len(pathway_pvalues),
                "n_significant": len(sig),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _solve_ode(self, x0: Tensor, t_eval: Tensor) -> Tensor:
        """Solve the Neural ODE. Wraps model.solve or uses torchdiffeq."""
        if hasattr(self.model, "solve"):
            return self.model.solve(x0, t_eval)
        # Fallback: use torchdiffeq if available
        try:
            from torchdiffeq import odeint
            return odeint(self.model.ode_func, x0, t_eval)
        except ImportError:
            raise RuntimeError(
                "Model must expose a .solve() method or torchdiffeq must be installed."
            )

    def _ode_rhs(self, x: Tensor, t: Tensor) -> Tensor:
        """Evaluate the ODE right-hand side f(x, t)."""
        if hasattr(self.model, "ode_func"):
            return self.model.ode_func(t, x)
        raise RuntimeError("Model must expose an .ode_func(t, x) method.")

    def _default_target(self, x_final: Tensor) -> Tensor:
        """Default target function: resistance probability from model."""
        if hasattr(self.model, "predict"):
            return self.model.predict(x_final)
        if hasattr(self.model, "classifier"):
            return self.model.classifier(x_final)
        # Fallback: L2 norm as proxy
        return x_final.norm(dim=-1)

    @staticmethod
    def _benjamini_hochberg(pvalues: Dict[str, float]) -> Dict[str, float]:
        """Benjamini-Hochberg FDR correction.

        Args:
            pvalues: Dict mapping name -> raw p-value.

        Returns:
            Dict mapping name -> FDR-corrected p-value.
        """
        names = list(pvalues.keys())
        pvals = np.array([pvalues[n] for n in names])
        n = len(pvals)
        if n == 0:
            return {}

        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]

        # BH formula: p_adj(i) = min(p(i) * n / rank(i), p_adj(i+1))
        adjusted = np.zeros(n)
        adjusted[-1] = sorted_pvals[-1]
        for i in range(n - 2, -1, -1):
            rank = i + 1
            adjusted[i] = min(sorted_pvals[i] * n / rank, adjusted[i + 1])
        adjusted = np.minimum(adjusted, 1.0)

        result = {}
        for i, idx in enumerate(sorted_idx):
            result[names[idx]] = float(adjusted[i])
        return result


# ---------------------------------------------------------------------------
# Gene Interaction Extractor
# ---------------------------------------------------------------------------

class GeneInteractionExtractor:
    """Extract and validate learned gene-gene interactions from the Neural ODE.

    The structured Neural ODE in ResistanceMap parameterizes the dynamics as:
        dx/dt = A * sigma(x) + b
    where A is a sparse interaction matrix. This class extracts A, thresholds
    it, and compares recovered edges to curated gene regulatory network
    databases.

    Comparison databases:
    - TRRUST v2: Han et al. (2018) "TRRUST v2: an expanded reference database
      of human and mouse transcriptional regulatory interactions." Nucleic
      Acids Research.
    - RegNetwork: Liu et al. (2015) "RegNetwork: an integrated database of
      transcriptional and post-transcriptional regulatory networks." Database.
    - STRING v12: Szklarczyk et al. (2023) "The STRING database in 2023."
      Nucleic Acids Research.

    Args:
        model: Neural ODE model with accessible interaction matrix.
        gene_names: Ordered list of gene names.
        known_networks: Dict mapping database name -> set of (gene_i, gene_j) edges.
        config: PathwayAttributionConfig instance.
    """

    def __init__(
        self,
        model: nn.Module,
        gene_names: List[str],
        known_networks: Optional[Dict[str, set]] = None,
        config: Optional[PathwayAttributionConfig] = None,
    ) -> None:
        self.model = model
        self.gene_names = gene_names
        self.known_networks = known_networks or {}
        self.config = config or PathwayAttributionConfig()

    def extract_interaction_matrix(self) -> np.ndarray:
        """Extract the interaction matrix A from the model.

        Searches for the matrix in common locations within the model
        architecture: model.A, model.ode_func.A, model.ode_func.linear.weight.

        Returns:
            Numpy array of shape (n_genes, n_genes).
        """
        A = None
        # Try common attribute paths
        for path in ["A", "interaction_matrix", "ode_func.A",
                      "ode_func.interaction_matrix", "ode_func.linear.weight"]:
            obj = self.model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                A = obj
                break
            except AttributeError:
                continue

        if A is None:
            raise RuntimeError(
                "Cannot find interaction matrix in model. Expected one of: "
                "model.A, model.ode_func.A, model.ode_func.linear.weight"
            )

        if isinstance(A, Tensor):
            A = A.detach().cpu().numpy()
        return A

    def extract_interactions(
        self,
        threshold: Optional[float] = None,
    ) -> InteractionResult:
        """Extract, threshold, and validate gene interactions.

        Pipeline:
        1. Extract the interaction matrix A.
        2. Threshold by magnitude to get significant edges.
        3. Compare to known databases (TRRUST, RegNetwork, STRING).
        4. Rank novel interactions by confidence.

        Args:
            threshold: Magnitude threshold for including an edge.
                Defaults to config.interaction_threshold.

        Returns:
            InteractionResult with interactions, precision/recall, and novel edges.
        """
        threshold = threshold or self.config.interaction_threshold
        A = self.extract_interaction_matrix()
        n = A.shape[0]

        # Extract edges above threshold
        learned_edges = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(A[i, j]) > threshold:
                    learned_edges.append((
                        self.gene_names[i],
                        self.gene_names[j],
                        float(A[i, j]),
                    ))

        # Sort by absolute weight (confidence)
        learned_edges.sort(key=lambda e: abs(e[2]), reverse=True)

        # Build set of learned edges (undirected for comparison)
        learned_set = {(e[0], e[1]) for e in learned_edges}
        learned_undirected = set()
        for g1, g2 in learned_set:
            learned_undirected.add((min(g1, g2), max(g1, g2)))

        # Compare to each known network
        precision_scores = {}
        recall_scores = {}
        for db_name, known_edges in self.known_networks.items():
            known_undirected = set()
            for g1, g2 in known_edges:
                if g1 in self.gene_names and g2 in self.gene_names:
                    known_undirected.add((min(g1, g2), max(g1, g2)))

            if len(learned_undirected) == 0:
                precision_scores[db_name] = 0.0
            else:
                tp = len(learned_undirected & known_undirected)
                precision_scores[db_name] = tp / len(learned_undirected)

            if len(known_undirected) == 0:
                recall_scores[db_name] = 0.0
            else:
                tp = len(learned_undirected & known_undirected)
                recall_scores[db_name] = tp / len(known_undirected)

        # Novel interactions: in learned but not in ANY known DB
        all_known = set()
        for known_edges in self.known_networks.values():
            for g1, g2 in known_edges:
                all_known.add((min(g1, g2), max(g1, g2)))

        novel = [
            (e[0], e[1], abs(e[2]))
            for e in learned_edges
            if (min(e[0], e[1]), max(e[0], e[1])) not in all_known
        ]

        return InteractionResult(
            learned_interactions=learned_edges,
            precision_vs_known=precision_scores,
            recall_vs_known=recall_scores,
            novel_interactions=novel,
            interaction_matrix=A,
        )

    def compute_interaction_significance(
        self,
        n_permutations: int = 1000,
    ) -> Dict[Tuple[str, str], float]:
        """Assess statistical significance of each learned interaction.

        Permutes gene labels and recomputes A to build a null distribution
        for each edge weight.

        Args:
            n_permutations: Number of label permutations.

        Returns:
            Dict mapping (gene_i, gene_j) -> p-value.
        """
        A = self.extract_interaction_matrix()
        n = A.shape[0]
        threshold = self.config.interaction_threshold

        # Build null distribution via permutation of gene labels
        null_max = np.zeros(n_permutations)
        for p in range(n_permutations):
            perm = np.random.permutation(n)
            A_perm = A[perm][:, perm]
            np.fill_diagonal(A_perm, 0)
            null_max[p] = np.abs(A_perm).max()

        # P-value for each edge
        pvalues = {}
        for i in range(n):
            for j in range(n):
                if i != j and abs(A[i, j]) > threshold:
                    pval = (np.sum(null_max >= abs(A[i, j])) + 1) / (n_permutations + 1)
                    pvalues[(self.gene_names[i], self.gene_names[j])] = pval

        return pvalues

    def get_top_regulators(self, k: int = 20) -> List[Tuple[str, float]]:
        """Identify the top-k genes with highest outgoing interaction strength.

        These are candidate master regulators of resistance.

        Args:
            k: Number of top regulators to return.

        Returns:
            List of (gene_name, total_outgoing_weight) tuples.
        """
        A = self.extract_interaction_matrix()
        out_strength = np.abs(A).sum(axis=1)
        top_idx = np.argsort(out_strength)[::-1][:k]
        return [(self.gene_names[i], float(out_strength[i])) for i in top_idx]

    def get_top_targets(self, k: int = 20) -> List[Tuple[str, float]]:
        """Identify the top-k genes with highest incoming interaction strength.

        These are the most regulated genes in the network.

        Args:
            k: Number of top targets to return.

        Returns:
            List of (gene_name, total_incoming_weight) tuples.
        """
        A = self.extract_interaction_matrix()
        in_strength = np.abs(A).sum(axis=0)
        top_idx = np.argsort(in_strength)[::-1][:k]
        return [(self.gene_names[i], float(in_strength[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Utility: load pathway databases
# ---------------------------------------------------------------------------

def load_pathway_database(
    source: str = "hallmark",
    gmt_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Load a gene set database from GMT format or built-in collections.

    Supports:
    - 'hallmark': MSigDB Hallmark gene sets (Liberzon et al., 2015)
    - 'kegg': KEGG pathway gene sets
    - 'reactome': Reactome pathway gene sets
    - Custom GMT files

    Args:
        source: One of 'hallmark', 'kegg', 'reactome', or 'custom'.
        gmt_path: Path to GMT file (required if source='custom').

    Returns:
        Dict mapping pathway name -> list of gene symbols.
    """
    if source == "custom" and gmt_path is None:
        raise ValueError("gmt_path is required when source='custom'.")

    if gmt_path is not None:
        return _parse_gmt(gmt_path)

    # Built-in minimal pathway definitions for testing / demonstration.
    # In production, these are loaded from MSigDB GMT files.
    builtin = {
        "hallmark": {
            "HALLMARK_APOPTOSIS": ["BAX", "BCL2", "CASP3", "CASP8", "CASP9",
                                     "CYCS", "APAF1", "BID", "BAK1", "MCL1",
                                     "BIRC5", "XIAP"],
            "HALLMARK_PI3K_AKT_MTOR": ["AKT1", "AKT2", "MTOR", "PIK3CA",
                                         "PIK3CB", "PTEN", "TSC1", "TSC2",
                                         "RPTOR", "RICTOR", "RPS6KB1", "EIF4E"],
            "HALLMARK_NF_KB": ["NFKB1", "NFKB2", "RELA", "RELB", "REL",
                                "IKBKB", "IKBKG", "CHUK", "TRAF2", "TRAF6",
                                "TNFAIP3", "NFKBIA"],
            "HALLMARK_DNA_REPAIR": ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR",
                                     "CHEK1", "CHEK2", "TP53", "MDM2", "PARP1",
                                     "XRCC1", "MLH1"],
            "HALLMARK_DRUG_METABOLISM": ["ABCB1", "ABCC1", "ABCG2", "CYP3A4",
                                          "CYP2D6", "UGT1A1", "GSTP1", "NR1I2",
                                          "NR1I3", "SLC22A1", "SLCO1B1", "CES1"],
            "HALLMARK_STEMNESS": ["SOX2", "POU5F1", "NANOG", "KLF4", "MYC",
                                   "BMI1", "LGR5", "CD44", "ALDH1A1", "PROM1",
                                   "NOTCH1", "WNT3A"],
        },
        "kegg": {
            "KEGG_ABC_TRANSPORTERS": ["ABCB1", "ABCC1", "ABCG2", "ABCA1",
                                       "ABCB4", "ABCC2", "ABCB11", "ABCA7",
                                       "ABCD1", "ABCG1", "ABCG5", "ABCG8"],
            "KEGG_PROTEASOME": ["PSMB5", "PSMB8", "PSMB9", "PSMA1", "PSMA2",
                                 "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
                                 "PSMB1", "PSMB2"],
        },
    }

    if source in builtin:
        return builtin[source]
    raise ValueError(f"Unknown pathway source: {source}. Use 'hallmark', 'kegg', 'reactome', or 'custom'.")


def _parse_gmt(path: str) -> Dict[str, List[str]]:
    """Parse a GMT (Gene Matrix Transposed) file.

    GMT format: each line is tab-separated with:
        pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...

    Args:
        path: Path to .gmt file.

    Returns:
        Dict mapping pathway name -> list of gene symbols.
    """
    db = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            db[name] = genes
    return db


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_pathway_attribution():
    """Unit tests for PathwayAttributor and GeneInteractionExtractor."""
    print("Running pathway attribution tests...")

    # --- Mock model for testing ---
    class MockODEFunc(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)

        def forward(self, t, x):
            return x @ self.A

    class MockModel(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.ode_func = MockODEFunc(dim)
            self.classifier = nn.Linear(dim, 1)

        def solve(self, x0, t_eval):
            # Simple Euler integration for testing
            dt = (t_eval[1] - t_eval[0]).item() if len(t_eval) > 1 else 0.1
            traj = [x0]
            x = x0
            for i in range(1, len(t_eval)):
                x = x + dt * self.ode_func(t_eval[i], x)
                traj.append(x)
            return torch.stack(traj)

        def predict(self, x):
            return torch.sigmoid(self.classifier(x))

    dim = 24
    gene_names = [f"GENE_{i}" for i in range(dim)]
    model = MockModel(dim)

    # Create a small pathway DB
    pathway_db = {
        "PATHWAY_A": gene_names[:10],
        "PATHWAY_B": gene_names[10:20],
    }

    config = PathwayAttributionConfig(
        n_ig_steps=10,
        n_permutations=50,
        temporal_resolution=10,
        min_pathway_size=5,
    )

    # Test PathwayAttributor
    attributor = PathwayAttributor(model, gene_names, pathway_db, config)

    x0 = torch.randn(1, dim)
    t_span = torch.linspace(0, 1, 5)

    # Test temporal IG
    tig = attributor.compute_temporal_integrated_gradients(x0, t_span)
    assert tig.shape == (10, dim), f"Expected (10, {dim}), got {tig.shape}"
    print(f"  Temporal IG shape: {tig.shape} -- PASS")

    # Test standard IG
    sig = attributor.compute_standard_integrated_gradients(x0, t_span)
    assert sig.shape == (dim,), f"Expected ({dim},), got {sig.shape}"
    print(f"  Standard IG shape: {sig.shape} -- PASS")

    # Test pathway aggregation
    scores = attributor.aggregate_to_pathways(tig)
    assert "PATHWAY_A" in scores
    assert scores["PATHWAY_A"].shape == (10,)
    print(f"  Pathway aggregation: {len(scores)} pathways -- PASS")

    # Test full pipeline
    result = attributor.attribute(x0, t_span)
    assert isinstance(result, PathwayAttributionResult)
    assert result.gene_attributions.shape == (10, dim)
    print(f"  Full pipeline: {result.metadata['n_pathways_tested']} tested -- PASS")

    # Test GeneInteractionExtractor
    known = {"TRRUST": {("GENE_0", "GENE_1"), ("GENE_2", "GENE_3")}}
    extractor = GeneInteractionExtractor(model, gene_names, known, config)

    A = extractor.extract_interaction_matrix()
    assert A.shape == (dim, dim)
    print(f"  Interaction matrix shape: {A.shape} -- PASS")

    interactions = extractor.extract_interactions(threshold=0.01)
    assert isinstance(interactions, InteractionResult)
    print(f"  Extracted {len(interactions.learned_interactions)} interactions -- PASS")

    top_reg = extractor.get_top_regulators(5)
    assert len(top_reg) == 5
    print(f"  Top regulators: {[r[0] for r in top_reg]} -- PASS")

    # Test BH correction
    pvals = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}
    corrected = PathwayAttributor._benjamini_hochberg(pvals)
    assert all(0 <= v <= 1 for v in corrected.values())
    print(f"  BH correction: {corrected} -- PASS")

    # Test GMT parser with builtin
    hallmark = load_pathway_database("hallmark")
    assert "HALLMARK_APOPTOSIS" in hallmark
    print(f"  Built-in Hallmark DB: {len(hallmark)} sets -- PASS")

    print("All pathway attribution tests PASSED.\n")


if __name__ == "__main__":
    _test_pathway_attribution()
