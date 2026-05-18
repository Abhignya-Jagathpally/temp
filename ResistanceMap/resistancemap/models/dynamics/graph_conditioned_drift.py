"""
resistancemap/models/dynamics/graph_conditioned_drift.py
========================================================
Graph-conditioned drift network.

Given a current latent state ``z(t)`` and a *fixed* biological graph
``G = (V, E)`` whose nodes are proteins/genes, this module computes:

  g_theta(z, G, d) = MLP([z; pool(GNN(z_node, G, d))])

Where ``z_node`` is a projection of ``z`` back onto the node space. The graph
contribution is what makes the dynamics "PPI-aware" — without it, the drift
collapses to a generic neural ODE over the latent.

This implementation uses a simple 2-layer message-passing block written in
pure PyTorch (no torch-geometric required) so the dynamics module stays
importable in minimal environments. For production use, a torch-geometric
:class:`GATConv` / :class:`GCNConv` is a drop-in replacement — the
:class:`ProteinNetwork` in :mod:`resistancemap.models.protein_network` already
does this.

A ``graph=None`` argument turns the module into a no-op identity, so the
neural-ODE / SDE wrappers can call this unconditionally.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConditionedDrift(nn.Module):
    """Latent-state drift conditioned on (z, t, drug, graph).

    Parameters
    ----------
    d_latent :
        Latent state dimension (matches ``MORTFMConfig.d_latent``).
    drug_dim :
        Drug token dimension. Set to 0 to disable drug conditioning.
    n_graph_nodes :
        If > 0, project ``z`` to this many node-features and run two rounds
        of message passing on the supplied graph. Set to 0 to disable the
        graph branch.
    hidden :
        Hidden width of the drift MLP.
    """

    def __init__(
        self,
        d_latent: int,
        *,
        drug_dim: int = 0,
        n_graph_nodes: int = 0,
        graph_hidden: int = 32,
        hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.drug_dim = drug_dim
        self.n_graph_nodes = n_graph_nodes
        self.graph_hidden = graph_hidden

        in_dim = d_latent + 1 + drug_dim  # +1 for time
        if n_graph_nodes > 0:
            self.z_to_node = nn.Linear(d_latent, n_graph_nodes * graph_hidden)
            self.node_msg1 = nn.Linear(graph_hidden, graph_hidden)
            self.node_msg2 = nn.Linear(graph_hidden, graph_hidden)
            self.graph_to_z = nn.Linear(graph_hidden, d_latent)
            in_dim += d_latent  # concatenated graph readout

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_latent),
        )

    def _message_pass(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One round of additive message passing over a sparse edge list.

        Parameters
        ----------
        node_features :
            ``(N_batch, n_nodes, graph_hidden)``.
        edge_index :
            ``(2, E)`` long tensor of (source, target) node indices.
        edge_weight :
            Optional ``(E,)`` float tensor. Default uniform weights.
        """
        N, V, H = node_features.shape
        src = edge_index[0]
        dst = edge_index[1]
        if edge_weight is None:
            edge_weight = torch.ones(src.shape[0], device=node_features.device)
        msg = node_features[:, src, :] * edge_weight.view(1, -1, 1)
        # Scatter-add into destination nodes.
        out = torch.zeros_like(node_features)
        # ``index_add_`` works on a fixed dim, so reshape:
        out_flat = out.view(N * V, H)
        offsets = torch.arange(N, device=node_features.device) * V
        dst_flat = dst.unsqueeze(0) + offsets.unsqueeze(1)  # (N, E)
        msg_flat = msg.reshape(N * src.shape[0], H)
        out_flat.index_add_(0, dst_flat.reshape(-1), msg_flat)
        return out_flat.view(N, V, H)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> torch.Tensor:
        """Compute ``dz/dt`` (deterministic drift)."""
        N = z.shape[0]
        t_in = t.expand(N, 1) if t.ndim == 0 else (t.unsqueeze(-1) if t.ndim == 1 else t)
        parts = [z, t_in.to(z.dtype)]
        if self.drug_dim > 0:
            if drug is None:
                drug = z.new_zeros(N, self.drug_dim)
            parts.append(drug)

        if self.n_graph_nodes > 0 and graph is not None:
            nodes = self.z_to_node(z).view(N, self.n_graph_nodes, self.graph_hidden)
            edge_index = graph.get("edge_index") if isinstance(graph, dict) else None
            edge_weight = graph.get("edge_weight") if isinstance(graph, dict) else None
            if edge_index is not None:
                msgs = self._message_pass(nodes, edge_index, edge_weight)
                nodes = F.gelu(self.node_msg1(nodes + msgs))
                msgs2 = self._message_pass(nodes, edge_index, edge_weight)
                nodes = F.gelu(self.node_msg2(nodes + msgs2))
            graph_readout = self.graph_to_z(nodes.mean(dim=1))
            parts.append(graph_readout)
        elif self.n_graph_nodes > 0:
            parts.append(z.new_zeros(N, self.d_latent))

        x = torch.cat(parts, dim=-1)
        return self.mlp(x)
