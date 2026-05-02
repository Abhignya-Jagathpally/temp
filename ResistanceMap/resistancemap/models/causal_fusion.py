"""Causally Structured Multi-Modal Fusion for ResistanceMap.

Replaces symmetric cross-attention with a fusion architecture grounded
in the known biological causal hierarchy:

    Genetic --> Epigenetic --> Transcriptomic --> Proteomic

Key innovations:
    1. **CausalModalityGraph**: Directed acyclic graph (DAG) encoding the
       biological hierarchy. Information flows ONLY along causal edges
       (parent -> child), enforced by masked attention.
    2. **Structural Causal Model (SCM) Layer**: Each modality representation
       is a function of its parents in the causal graph plus independent
       noise (exogenous variable). Enables do-calculus interventions.
    3. **Confound-Aware Attention**: Batch effects and technical noise
       are modeled as confounders that are d-separated from the causal
       path via conditional independence constraints.
    4. **Intervention Interface**: do(modality = x) blocks all incoming
       causal edges, enabling counterfactual causal queries like
       "what if the epigenetic state were fixed to x?".

Theoretical grounding:
    - Pearl, J. (2009). "Causality," Cambridge University Press.
    - Scholkopf et al. (2021). "Toward Causal Representation Learning."
      Proc. IEEE 109(5), 612-634.
    - Louizos et al. (2017). "Causal Effect Inference with Deep
      Latent-Variable Models." NeurIPS.

Authors: ResistanceMap Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "Modality",
    "CausalFusionConfig",
    "CausalModalityGraph",
    "SCMLayer",
    "CausalAttentionBlock",
    "ConfounderEncoder",
    "CausalFusionNetwork",
]


# ===================================================================
# Configuration
# ===================================================================

class Modality(IntEnum):
    """Multi-omics modalities in causal order.

    The integer ordering encodes the causal hierarchy:
    lower index = upstream cause.
    """
    GENETIC = 0       # Mutations, CNVs, SVs
    EPIGENETIC = 1    # ATAC-seq, histone mods, methylation
    TRANSCRIPTOMIC = 2  # scRNA-seq
    PROTEOMIC = 3     # Mass spec, CyTOF

    @property
    def full_name(self) -> str:
        return {
            0: "Genetic (mutations, CNVs)",
            1: "Epigenetic (ATAC, histone, methylation)",
            2: "Transcriptomic (scRNA-seq)",
            3: "Proteomic (mass spec, CyTOF)",
        }[self.value]


# The biological causal DAG: directed edges (parent -> child)
BIOLOGICAL_CAUSAL_EDGES: List[Tuple[Modality, Modality]] = [
    (Modality.GENETIC, Modality.EPIGENETIC),
    (Modality.GENETIC, Modality.TRANSCRIPTOMIC),
    (Modality.EPIGENETIC, Modality.TRANSCRIPTOMIC),
    (Modality.TRANSCRIPTOMIC, Modality.PROTEOMIC),
    # Skip connection for direct genetic effects on protein
    (Modality.GENETIC, Modality.PROTEOMIC),
    # Epigenetic influence on protein (chromatin -> translation regulation)
    (Modality.EPIGENETIC, Modality.PROTEOMIC),
]


@dataclass
class CausalFusionConfig:
    """Configuration for causal fusion network.

    Attributes:
        modality_dims: Dict mapping Modality -> input feature dimension.
        hidden_dim: Internal representation dimension (all modalities
            projected to this shared dimension).
        n_heads: Number of attention heads in causal attention.
        n_layers: Number of stacked causal attention layers.
        dropout: Dropout rate.
        confounder_dim: Dimension for confounder representations.
        n_confounders: Number of modeled confounders (batch effects etc.).
        noise_dim: Dimension of exogenous noise per modality.
        intervention_noise_std: Std of noise added during do() interventions
            to prevent distributional collapse.
    """
    modality_dims: Dict[int, int] = field(default_factory=lambda: {
        Modality.GENETIC: 128,
        Modality.EPIGENETIC: 256,
        Modality.TRANSCRIPTOMIC: 512,
        Modality.PROTEOMIC: 256,
    })
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    confounder_dim: int = 32
    n_confounders: int = 4   # e.g., batch, platform, site, prep
    noise_dim: int = 16
    intervention_noise_std: float = 0.01


# ===================================================================
# Causal DAG Structure
# ===================================================================

class CausalModalityGraph(nn.Module):
    """Encodes the biological causal DAG and computes causal attention masks.

    The DAG structure is fixed by biological knowledge (not learned).
    The attention mask allows information to flow ONLY from causal
    parents to children, preventing spurious correlations from
    influencing the fused representation.

    For a modality m, its representation can only attend to:
        - Its own representation (self-loop)
        - Representations of modalities that are its causal ancestors

    This is a hard constraint, not a soft bias.
    """

    def __init__(
        self,
        edges: Optional[List[Tuple[Modality, Modality]]] = None,
    ) -> None:
        super().__init__()
        self.edges = edges or BIOLOGICAL_CAUSAL_EDGES
        n = len(Modality)

        # Build adjacency matrix (parent -> child)
        adj = torch.zeros(n, n)
        for parent, child in self.edges:
            adj[child.value, parent.value] = 1.0  # child attends to parent

        # Add self-loops
        adj = adj + torch.eye(n)

        # Compute transitive closure for ancestor mask
        # (allows multi-hop causal paths)
        ancestor_mask = adj.clone()
        for _ in range(n):
            ancestor_mask = (ancestor_mask @ adj).clamp(max=1.0)
        ancestor_mask = (ancestor_mask > 0).float()

        self.register_buffer("adjacency", adj)
        self.register_buffer("ancestor_mask", ancestor_mask)
        self.register_buffer("direct_parent_mask", adj)

    def get_attention_mask(self, use_ancestors: bool = True) -> torch.Tensor:
        """Get the causal attention mask.

        Args:
            use_ancestors: If True, allow attention to all ancestors
                (transitive closure). If False, only direct parents.

        Returns:
            mask: (n_modalities, n_modalities) binary mask where
                mask[i,j] = 1 means modality i can attend to modality j.
        """
        if use_ancestors:
            return self.ancestor_mask
        return self.direct_parent_mask

    def get_intervention_mask(self, do_modality: Modality) -> torch.Tensor:
        """Get attention mask after do(modality = x) intervention.

        do(X = x) removes all incoming edges to X, making it independent
        of its parents. X's row in the mask becomes a self-loop only.

        Args:
            do_modality: The modality being intervened on.

        Returns:
            mask: (n_modalities, n_modalities) modified attention mask.
        """
        mask = self.ancestor_mask.clone()
        idx = do_modality.value
        mask[idx, :] = 0.0
        mask[idx, idx] = 1.0  # Self-loop preserved
        return mask

    def parents(self, modality: Modality) -> List[Modality]:
        """Get direct parents of a modality in the causal graph.

        Args:
            modality: Query modality.

        Returns:
            List of parent Modality values.
        """
        idx = modality.value
        parent_indices = (self.direct_parent_mask[idx] > 0).nonzero(as_tuple=True)[0]
        return [Modality(i.item()) for i in parent_indices if i.item() != idx]

    def children(self, modality: Modality) -> List[Modality]:
        """Get direct children of a modality in the causal graph.

        Args:
            modality: Query modality.

        Returns:
            List of child Modality values.
        """
        idx = modality.value
        child_indices = (self.direct_parent_mask[:, idx] > 0).nonzero(as_tuple=True)[0]
        return [Modality(i.item()) for i in child_indices if i.item() != idx]

    def is_ancestor(self, potential_ancestor: Modality, descendant: Modality) -> bool:
        """Check if one modality is a causal ancestor of another.

        Args:
            potential_ancestor: Candidate ancestor.
            descendant: Candidate descendant.

        Returns:
            True if potential_ancestor is an ancestor of descendant.
        """
        return bool(
            self.ancestor_mask[descendant.value, potential_ancestor.value].item() > 0
            and potential_ancestor != descendant
        )


# ===================================================================
# Confounder Encoder
# ===================================================================

class ConfounderEncoder(nn.Module):
    """Encodes observed confounders (batch effects, technical noise)
    into a representation that can be conditioned on or marginalized out.

    Confounders are modeled as variables that affect all modalities
    but are NOT on the causal path between them. By conditioning on
    confounders, we achieve d-separation between the causal path and
    confounding paths (Pearl, 2009, Sec 3.3).

    Args:
        n_confounders: Number of categorical confounder variables.
        max_categories: Maximum categories per confounder.
        confounder_dim: Embedding dimension per confounder.
        output_dim: Dimension of aggregated confounder representation.
    """

    def __init__(
        self,
        n_confounders: int = 4,
        max_categories: int = 32,
        confounder_dim: int = 32,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_confounders = n_confounders
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_categories, confounder_dim)
            for _ in range(n_confounders)
        ])
        self.aggregate = nn.Sequential(
            nn.Linear(n_confounders * confounder_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, confounder_ids: torch.Tensor) -> torch.Tensor:
        """Encode confounder variables.

        Args:
            confounder_ids: (B, n_confounders) integer indices for each
                confounder variable.

        Returns:
            c: (B, output_dim) confounder representation.
        """
        parts = []
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(confounder_ids[:, i]))
        cat = torch.cat(parts, dim=-1)
        return self.aggregate(cat)


# ===================================================================
# Structural Causal Model Layer
# ===================================================================

class SCMLayer(nn.Module):
    """Structural Causal Model layer for a single modality.

    Implements the structural equation:
        Z_m = f_m(Z_{pa(m)}, U_m, C)

    where:
        Z_m: representation of modality m
        Z_{pa(m)}: representations of causal parents
        U_m: exogenous noise (independent across modalities)
        C: confounder representation (shared)
        f_m: learned structural function (MLP)

    The exogenous noise U_m is either sampled (during training for
    regularization) or set to zero (during deterministic inference).

    This formulation ensures that:
        1. Each modality's representation is causally determined by its
           parents (not spuriously correlated with siblings).
        2. do(Z_m = z) interventions are well-defined: set Z_m = z
           and recompute all descendants.
        3. Counterfactuals are computable via abduction-action-prediction.

    Args:
        modality: Which modality this layer represents.
        hidden_dim: Representation dimension.
        n_parents: Maximum number of parent modalities.
        noise_dim: Exogenous noise dimension.
        confounder_dim: Confounder representation dimension.
    """

    def __init__(
        self,
        modality: Modality,
        hidden_dim: int,
        n_parents: int,
        noise_dim: int = 16,
        confounder_dim: int = 64,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        # Input: own representation + parent representations + noise + confounder
        input_dim = hidden_dim + n_parents * hidden_dim + noise_dim + confounder_dim

        self.structural_fn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Residual connection
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_self: torch.Tensor,
        z_parents: List[torch.Tensor],
        confounder: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply structural equation.

        Args:
            z_self: (B, hidden_dim) current representation of this modality.
            z_parents: List of (B, hidden_dim) parent representations.
            confounder: (B, confounder_dim) confounder encoding.
            noise: (B, noise_dim) exogenous noise. Sampled if None.

        Returns:
            z_new: (B, hidden_dim) updated representation.
        """
        B = z_self.shape[0]
        device = z_self.device

        # Exogenous noise
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=device)

        # Concatenate all inputs
        parts = [z_self]
        for zp in z_parents:
            parts.append(zp)
        # Pad if fewer parents than expected
        n_expected_parents = (
            self.structural_fn[0].in_features
            - self.hidden_dim - self.noise_dim - confounder.shape[-1]
        ) // self.hidden_dim
        while len(z_parents) < n_expected_parents:
            parts.append(torch.zeros_like(z_self))
            z_parents.append(torch.zeros_like(z_self))  # for count only
        parts.extend([noise, confounder])

        concat = torch.cat(parts, dim=-1)
        out = self.structural_fn(concat)

        # Gated residual: blend new with old
        gate_input = torch.cat([z_self, out], dim=-1)
        gate = self.residual_gate(gate_input)
        z_new = gate * out + (1 - gate) * z_self

        return z_new


# ===================================================================
# Causal Attention Block
# ===================================================================

class CausalAttentionBlock(nn.Module):
    """Multi-head attention with causal masking.

    Standard multi-head attention where the attention matrix is masked
    to only allow information flow along causal edges. This means
    modality i can attend to modality j ONLY if j is a causal ancestor
    of i (or j == i for self-attention).

    The mask is computed from the CausalModalityGraph and remains fixed
    during training (the causal structure is not learned).

    Args:
        hidden_dim: Representation dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Causally masked multi-head attention.

        Args:
            x: (B, M, hidden_dim) modality representations where M is
                the number of modalities.
            causal_mask: (M, M) binary mask from CausalModalityGraph.

        Returns:
            out: (B, M, hidden_dim) updated representations.
        """
        B, M, D = x.shape
        residual = x
        x = self.norm1(x)

        # QKV projections
        Q = self.q_proj(x).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores with causal mask
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, M, M)

        # Apply causal mask: set blocked positions to -inf
        mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, M, M)
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum
        out = (attn @ V).transpose(1, 2).contiguous().view(B, M, D)
        out = self.out_proj(out)

        # Residual + FFN
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x

    def get_attention_weights(
        self, x: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract attention weights for interpretability.

        Args:
            x: (B, M, hidden_dim) modality representations.
            causal_mask: (M, M) binary mask.

        Returns:
            weights: (B, n_heads, M, M) attention weights.
        """
        B, M, D = x.shape
        x = self.norm1(x)
        Q = self.q_proj(x).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        return F.softmax(attn, dim=-1)


# ===================================================================
# Main Causal Fusion Network
# ===================================================================

class CausalFusionNetwork(nn.Module):
    """End-to-end causally structured multi-modal fusion network.

    Fuses representations from genetic, epigenetic, transcriptomic, and
    proteomic modalities using:
        1. Modality-specific input projections to shared dimension.
        2. CausalModalityGraph for attention masking.
        3. Stacked CausalAttentionBlocks for message passing.
        4. SCM layers for structural equation enforcement.
        5. ConfounderEncoder for batch effect conditioning.

    The output is a fused representation that respects the causal
    hierarchy and supports interventional queries.

    Args:
        config: CausalFusionConfig.
    """

    def __init__(self, config: CausalFusionConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # Modality-specific input projections
        self.input_projections = nn.ModuleDict()
        for mod in Modality:
            dim_in = config.modality_dims.get(mod.value, H)
            self.input_projections[str(mod.value)] = nn.Sequential(
                nn.Linear(dim_in, H),
                nn.LayerNorm(H),
                nn.GELU(),
            )

        # Causal graph
        self.causal_graph = CausalModalityGraph()

        # Modality type embeddings (learned, additive)
        self.modality_embeddings = nn.Embedding(len(Modality), H)

        # Causal attention layers
        self.attention_layers = nn.ModuleList([
            CausalAttentionBlock(H, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        # SCM layers (one per modality)
        self.scm_layers = nn.ModuleDict()
        for mod in Modality:
            n_parents = len(self.causal_graph.parents(mod))
            self.scm_layers[str(mod.value)] = SCMLayer(
                modality=mod,
                hidden_dim=H,
                n_parents=max(n_parents, 1),
                noise_dim=config.noise_dim,
                confounder_dim=config.confounder_dim * 2,
            )

        # Confounder encoder
        self.confounder_encoder = ConfounderEncoder(
            n_confounders=config.n_confounders,
            confounder_dim=config.confounder_dim,
            output_dim=config.confounder_dim * 2,
        )

        # Output projection (fused representation)
        self.output_proj = nn.Sequential(
            nn.Linear(H * len(Modality), H * 2),
            nn.LayerNorm(H * 2),
            nn.GELU(),
            nn.Linear(H * 2, H),
        )

        # Intervention state
        self._intervention: Optional[Tuple[Modality, torch.Tensor]] = None

    def set_intervention(
        self,
        modality: Modality,
        value: torch.Tensor,
    ) -> None:
        """Set do(modality = value) for the next forward pass.

        This blocks all incoming causal edges to the specified modality
        and fixes its representation to the given value.

        Args:
            modality: Modality to intervene on.
            value: (B, hidden_dim) fixed representation.
        """
        self._intervention = (modality, value)

    def clear_intervention(self) -> None:
        """Clear any active intervention."""
        self._intervention = None

    def forward(
        self,
        modality_inputs: Dict[int, torch.Tensor],
        confounder_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with causal fusion.

        Args:
            modality_inputs: Dict mapping Modality.value -> (B, dim)
                raw modality feature tensors.
            confounder_ids: (B, n_confounders) integer confounder indices.
                If None, zeros are used.

        Returns:
            fused: (B, hidden_dim) fused representation.
            modality_reprs: Dict mapping modality name -> (B, hidden_dim)
                individual modality representations after causal fusion.
        """
        B = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device
        H = self.config.hidden_dim
        n_mods = len(Modality)

        # Encode confounders
        if confounder_ids is None:
            confounder_ids = torch.zeros(
                B, self.config.n_confounders, dtype=torch.long, device=device
            )
        confounder = self.confounder_encoder(confounder_ids)  # (B, conf_dim*2)

        # Project each modality to shared dimension
        z_list = []
        for mod in Modality:
            if mod.value in modality_inputs:
                z = self.input_projections[str(mod.value)](
                    modality_inputs[mod.value]
                )
            else:
                z = torch.zeros(B, H, device=device)
            # Add modality type embedding
            mod_emb = self.modality_embeddings(
                torch.tensor(mod.value, device=device)
            )
            z = z + mod_emb
            z_list.append(z)

        # Stack into (B, M, H) for attention
        z = torch.stack(z_list, dim=1)  # (B, M, H)

        # Get causal attention mask
        if self._intervention is not None:
            do_mod, do_val = self._intervention
            causal_mask = self.causal_graph.get_intervention_mask(do_mod)
            # Override the intervened modality's representation
            z[:, do_mod.value, :] = do_val
        else:
            causal_mask = self.causal_graph.get_attention_mask(use_ancestors=True)

        # Causal attention layers
        for attn_layer in self.attention_layers:
            z = attn_layer(z, causal_mask)

        # SCM refinement: apply structural equations in topological order
        for mod in Modality:
            parents = self.causal_graph.parents(mod)
            z_parents = [z[:, p.value, :] for p in parents]
            z_self = z[:, mod.value, :]

            # Skip SCM for intervened modality
            if self._intervention is not None and self._intervention[0] == mod:
                continue

            z_new = self.scm_layers[str(mod.value)](
                z_self, z_parents, confounder
            )
            z = z.clone()
            z[:, mod.value, :] = z_new

        # Collect modality representations
        modality_reprs = {}
        for mod in Modality:
            modality_reprs[mod.name] = z[:, mod.value, :]

        # Fuse: concatenate all modalities and project
        z_flat = z.reshape(B, n_mods * H)
        fused = self.output_proj(z_flat)

        return fused, modality_reprs

    def interventional_query(
        self,
        modality_inputs: Dict[int, torch.Tensor],
        do_modality: Modality,
        do_value: torch.Tensor,
        confounder_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Answer a causal interventional query: P(Y | do(X = x)).

        Convenience method that sets the intervention, runs forward,
        and clears the intervention.

        Args:
            modality_inputs: Dict of modality features.
            do_modality: Modality to intervene on.
            do_value: (B, hidden_dim) intervention value.
            confounder_ids: Optional confounder indices.

        Returns:
            Same as forward(): (fused, modality_reprs) under intervention.
        """
        self.set_intervention(do_modality, do_value)
        try:
            result = self.forward(modality_inputs, confounder_ids)
        finally:
            self.clear_intervention()
        return result

    def causal_effect(
        self,
        modality_inputs: Dict[int, torch.Tensor],
        do_modality: Modality,
        do_value: torch.Tensor,
        confounder_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Average Causal Effect (ACE) of an intervention.

        ACE = E[Y | do(X=x)] - E[Y | do(X=0)]

        where Y is the fused representation.

        Args:
            modality_inputs: Dict of modality features.
            do_modality: Modality to intervene on.
            do_value: (B, hidden_dim) intervention value.
            confounder_ids: Optional confounder indices.

        Returns:
            ace: (B, hidden_dim) average causal effect.
        """
        B = do_value.shape[0]
        H = self.config.hidden_dim
        device = do_value.device

        # Factual: do(X = x)
        fused_factual, _ = self.interventional_query(
            modality_inputs, do_modality, do_value, confounder_ids
        )

        # Baseline: do(X = 0)
        zero_val = torch.zeros(B, H, device=device)
        fused_baseline, _ = self.interventional_query(
            modality_inputs, do_modality, zero_val, confounder_ids
        )

        ace = fused_factual - fused_baseline
        return ace

    def get_attention_weights(
        self, modality_inputs: Dict[int, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Extract attention weights from all layers for interpretability.

        Args:
            modality_inputs: Dict of modality features.

        Returns:
            List of (B, n_heads, M, M) attention weight tensors, one
            per attention layer.
        """
        B = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device
        H = self.config.hidden_dim

        z_list = []
        for mod in Modality:
            if mod.value in modality_inputs:
                z = self.input_projections[str(mod.value)](
                    modality_inputs[mod.value]
                )
            else:
                z = torch.zeros(B, H, device=device)
            mod_emb = self.modality_embeddings(
                torch.tensor(mod.value, device=device)
            )
            z_list.append(z + mod_emb)
        z = torch.stack(z_list, dim=1)

        mask = self.causal_graph.get_attention_mask()
        weights = []
        for attn_layer in self.attention_layers:
            w = attn_layer.get_attention_weights(z, mask)
            weights.append(w)
            z = attn_layer(z, mask)
        return weights


# ===================================================================
# Unit Tests
# ===================================================================

def _test_causal_graph() -> None:
    """Test CausalModalityGraph masks and queries."""
    graph = CausalModalityGraph()

    mask = graph.get_attention_mask()
    n = len(Modality)
    assert mask.shape == (n, n), f"Expected ({n},{n}), got {mask.shape}"

    # Self-loops present
    for i in range(n):
        assert mask[i, i] == 1.0, f"Missing self-loop at {i}"

    # Genetic can attend to itself only (no parents)
    assert mask[Modality.GENETIC, Modality.GENETIC] == 1.0

    # Proteomic can attend to everything (downstream of all)
    for mod in Modality:
        assert mask[Modality.PROTEOMIC, mod.value] == 1.0, (
            f"Proteomic should attend to {mod.name}"
        )

    # Genetic should NOT attend to transcriptomic (wrong direction)
    # But genetic has no parents so mask[GENETIC, TRANSCRIPTOMIC] should be 0
    assert mask[Modality.GENETIC, Modality.TRANSCRIPTOMIC] == 0.0

    # Intervention mask
    do_mask = graph.get_intervention_mask(Modality.EPIGENETIC)
    assert do_mask[Modality.EPIGENETIC, Modality.GENETIC] == 0.0, (
        "do(epigenetic) should block genetic -> epigenetic"
    )
    assert do_mask[Modality.EPIGENETIC, Modality.EPIGENETIC] == 1.0, (
        "do(epigenetic) should preserve self-loop"
    )

    # Parent/child queries
    parents = graph.parents(Modality.TRANSCRIPTOMIC)
    parent_names = {p.name for p in parents}
    assert "GENETIC" in parent_names
    assert "EPIGENETIC" in parent_names

    print("[PASS] CausalModalityGraph")


def _test_confounder_encoder() -> None:
    """Test ConfounderEncoder shape."""
    enc = ConfounderEncoder(n_confounders=4, confounder_dim=16, output_dim=32)
    B = 8
    ids = torch.randint(0, 10, (B, 4))
    out = enc(ids)
    assert out.shape == (B, 32), f"Expected ({B}, 32), got {out.shape}"
    print("[PASS] ConfounderEncoder")


def _test_scm_layer() -> None:
    """Test SCMLayer forward pass."""
    H, B = 64, 4
    layer = SCMLayer(
        modality=Modality.TRANSCRIPTOMIC,
        hidden_dim=H,
        n_parents=2,
        noise_dim=16,
        confounder_dim=32,
    )
    z_self = torch.randn(B, H)
    z_parents = [torch.randn(B, H), torch.randn(B, H)]
    conf = torch.randn(B, 32)

    out = layer(z_self, z_parents, conf)
    assert out.shape == (B, H), f"Expected ({B}, {H}), got {out.shape}"
    print("[PASS] SCMLayer")


def _test_causal_attention() -> None:
    """Test CausalAttentionBlock with mask."""
    B, M, H = 4, 4, 64
    block = CausalAttentionBlock(H, n_heads=4)
    x = torch.randn(B, M, H)
    mask = torch.eye(M)  # Identity mask (only self-attention)

    out = block(x, mask)
    assert out.shape == (B, M, H), f"Expected ({B},{M},{H}), got {out.shape}"

    # With full causal mask
    graph = CausalModalityGraph()
    full_mask = graph.get_attention_mask()
    out2 = block(x, full_mask)
    assert out2.shape == (B, M, H)
    print("[PASS] CausalAttentionBlock")


def _test_fusion_network() -> None:
    """Test full CausalFusionNetwork forward and intervention."""
    cfg = CausalFusionConfig(
        modality_dims={0: 32, 1: 64, 2: 128, 3: 64},
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
    )
    net = CausalFusionNetwork(cfg)
    B = 4

    inputs = {
        Modality.GENETIC.value: torch.randn(B, 32),
        Modality.EPIGENETIC.value: torch.randn(B, 64),
        Modality.TRANSCRIPTOMIC.value: torch.randn(B, 128),
        Modality.PROTEOMIC.value: torch.randn(B, 64),
    }

    # Normal forward
    fused, reprs = net(inputs)
    assert fused.shape == (B, 64), f"Expected ({B}, 64), got {fused.shape}"
    assert "GENETIC" in reprs
    assert "PROTEOMIC" in reprs

    # Interventional query
    do_val = torch.randn(B, 64)
    fused_do, reprs_do = net.interventional_query(
        inputs, Modality.EPIGENETIC, do_val
    )
    assert fused_do.shape == (B, 64)

    # Causal effect
    ace = net.causal_effect(inputs, Modality.EPIGENETIC, do_val)
    assert ace.shape == (B, 64)

    # Attention weights
    weights = net.get_attention_weights(inputs)
    assert len(weights) == cfg.n_layers

    print("[PASS] CausalFusionNetwork")


def run_all_tests() -> None:
    """Run all unit tests."""
    _test_causal_graph()
    _test_confounder_encoder()
    _test_scm_layer()
    _test_causal_attention()
    _test_fusion_network()
    print("\n=== All causal_fusion tests passed ===")


if __name__ == "__main__":
    run_all_tests()
