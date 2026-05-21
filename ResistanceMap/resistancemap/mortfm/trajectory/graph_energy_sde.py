"""
resistancemap/mortfm/trajectory/graph_energy_sde.py
====================================================
Graph-conditioned, drug-pressured neural SDE on a learned Waddington
landscape. The model rolls out a latent state ``z_t`` over a fixed time
grid via Euler-Maruyama.

Math (v15 plan section 5):

    dz_t = [-grad U_theta(z_t, d) + f_theta(z_t, g_t, d, c)] dt
           + sigma_theta(z_t, d) dW_t

  * U_theta : R^d_latent x R^d_drug -> R     (scalar potential)
  * f_theta : R^d_latent x R^d_graph x R^d_drug x R^d_clin -> R^d_latent
              (graph-conditioned drift)
  * sigma_theta : R^d_latent x R^d_drug -> R^{d_latent x d_latent}
              (state- and drug-dependent diffusion)

The SDE outputs:
  * z_traj   — (B, T, d_latent) deterministic mean trajectory
  * z_samples — (S, B, T, d_latent) Monte-Carlo samples for hitting-time
                and uncertainty estimation.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn

from resistancemap.mortfm.trajectory.grid import (
    CanonicalTimeGridConfig,
    canonical_time_grid,
)

logger = logging.getLogger(__name__)


def _mlp(d_in: int, d_hidden: int, d_out: int, n_hidden: int = 2,
         act: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(d_hidden, d_hidden), act()]
    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)


class WaddingtonPotential(nn.Module):
    """Scalar potential U_theta(z, d) with K learned basin centroids.

    Implemented as a quadratic-around-centroids energy plus a learned MLP
    residual. The negative gradient -grad U pulls states toward basins
    in a drug-modulated way (the basin weights are drug-conditioned).
    """

    def __init__(
        self,
        d_latent: int,
        d_drug: int,
        n_basins: int = 5,
        d_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_basins = n_basins
        # Initialize centroids spread on a learnable simplex.
        self.basin_centroids = nn.Parameter(torch.randn(n_basins, d_latent) * 0.3)
        self.basin_width = nn.Parameter(torch.ones(n_basins))
        self.drug_to_basin_weight = nn.Linear(d_drug, n_basins)
        self.residual = _mlp(d_latent + d_drug, d_hidden, 1, n_hidden=2)

    def forward(self, z: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """Return U(z, d) of shape (B,)."""
        # Per-basin squared distance, weighted by drug-conditioned weights.
        # z:        (B, d_latent)
        # centroids: (K, d_latent)
        diff = z.unsqueeze(1) - self.basin_centroids.unsqueeze(0)
        sqd = (diff ** 2).sum(dim=-1)                          # (B, K)
        widths = torch.clamp(self.basin_width, min=0.1) ** 2
        per_basin_energy = sqd / widths.unsqueeze(0)           # (B, K)
        w = torch.softmax(self.drug_to_basin_weight(drug), dim=-1)  # (B, K)
        quadratic = (w * per_basin_energy).sum(dim=-1)          # (B,)
        residual = self.residual(torch.cat([z, drug], dim=-1)).squeeze(-1)
        return quadratic + residual


class GraphConditionedDrift(nn.Module):
    """f_theta(z, g, d, c) — graph + drug + clinical conditioned drift."""

    def __init__(
        self,
        d_latent: int,
        d_graph: int,
        d_drug: int,
        d_clinical: int,
        d_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.net = _mlp(
            d_latent + d_graph + d_drug + d_clinical,
            d_hidden, d_latent, n_hidden=2,
        )

    def forward(
        self,
        z: torch.Tensor,
        graph_emb: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([z, graph_emb, drug, clinical], dim=-1))


class _Diffusion(nn.Module):
    """sigma_theta(z, d) — diagonal, positive, state + drug dependent."""

    def __init__(self, d_latent: int, d_drug: int, d_hidden: int = 64,
                 min_sigma: float = 1e-3, max_sigma: float = 0.5) -> None:
        super().__init__()
        self.net = _mlp(d_latent + d_drug, d_hidden, d_latent, n_hidden=2)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(self, z: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([z, drug], dim=-1))
        return self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(raw)


class GraphEnergyResistanceSDE(nn.Module):
    """The full latent dynamics module.

    .. note::
        ``graph_emb`` is assumed to be precomputed by the existing
        biological-graph GNN (Block A/D produce it). The SDE does NOT
        recompute graph context inside the rollout loop.

    Parameters
    ----------
    d_latent
        Latent state dimension (matches MORTFMConfig.d_latent).
    d_graph
        Graph embedding dimension.
    d_drug
        Drug context dimension.
    d_clinical
        Clinical context dimension (can be 0).
    n_basins
        Number of resistance basins (sensitive, persister, MRD, relapse, ...).
    integration_time
        Total integration time T (days or scaled units; the loss/metrics
        decide the unit).
    n_time_grid
        Number of Euler-Maruyama steps.
    n_mc_samples
        Number of Monte-Carlo trajectory samples per batch element.
    """

    def __init__(
        self,
        d_latent: int = 64,
        d_graph: int = 64,
        d_drug: int = 16,
        d_clinical: int = 4,
        n_basins: int = 5,
        integration_time: float = 1.0,
        n_time_grid: int = 8,
        n_mc_samples: int = 4,
        time_grid_config: Optional[CanonicalTimeGridConfig] = None,
    ) -> None:
        super().__init__()
        self.potential = WaddingtonPotential(d_latent, d_drug, n_basins=n_basins)
        self.drift = GraphConditionedDrift(d_latent, d_graph, d_drug, max(d_clinical, 1))
        self.diffusion = _Diffusion(d_latent, d_drug)
        self.n_mc_samples = n_mc_samples
        self._has_clinical = d_clinical > 0

        # v19 Phase 7: when a CanonicalTimeGridConfig is supplied, the SDE's
        # internal time grid is replaced by the shared canonical grid so the
        # survival head and the SDE rollout live on the same time axis.
        # If no config is given, the legacy linspace(0, integration_time,
        # n_time_grid) grid is preserved for backward compatibility with
        # v15/v17 checkpoints — but we emit a DeprecationWarning so callers
        # migrate before the dual-grid mode is removed in v20.
        if time_grid_config is not None:
            grid = time_grid_config.build()
            self.integration_time = float(time_grid_config.t_max_months)
            self.n_time_grid = int(time_grid_config.n_steps)
            self.register_buffer("t_grid", grid, persistent=False)
            self._uses_canonical_grid = True
        else:
            warnings.warn(
                "GraphEnergyResistanceSDE constructed without "
                "time_grid_config; falling back to legacy "
                f"linspace(0, {integration_time}, {n_time_grid}). The "
                "directional-consistency invariant between hitting CDF and "
                "competing-risk survival cannot be evaluated until both "
                "subsystems share a CanonicalTimeGridConfig. This fallback "
                "is preserved for v15/v17 checkpoint compatibility only.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.integration_time = integration_time
            self.n_time_grid = n_time_grid
            grid = torch.linspace(0.0, float(integration_time), int(n_time_grid))
            self.register_buffer("t_grid", grid, persistent=False)
            self._uses_canonical_grid = False

    def _grad_u(self, z: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """Compute grad_z U(z, d). Always run under enable_grad so the
        SDE works inside torch.no_grad() rollouts (e.g. evaluation)."""
        with torch.enable_grad():
            z_ = z.detach().requires_grad_(True)
            u = self.potential(z_, drug).sum()
            (grad,) = torch.autograd.grad(u, z_, create_graph=z.requires_grad)
        return grad

    def _step(
        self,
        z: torch.Tensor,
        graph_emb: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
        dt: float,
        deterministic: bool = False,
    ) -> torch.Tensor:
        f = self.drift(z, graph_emb, drug, clinical)
        grad_u = self._grad_u(z, drug)
        drift_term = -grad_u + f
        if deterministic:
            return z + drift_term * dt
        sigma = self.diffusion(z, drug)
        # Euler-Maruyama: z_{t+dt} = z_t + drift*dt + sigma*sqrt(dt)*xi, xi~N(0,I)
        noise = torch.randn_like(z) * math.sqrt(dt)
        return z + drift_term * dt + sigma * noise

    def forward(
        self,
        z0: torch.Tensor,
        graph_emb: torch.Tensor,
        drug: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        *,
        return_samples: bool = True,
    ) -> dict:
        """Roll out the SDE over the configured time grid.

        Parameters
        ----------
        z0 : (B, d_latent)
        graph_emb : (B, d_graph)
        drug : (B, d_drug)
        clinical : (B, d_clinical) or None

        Returns
        -------
        dict with keys:
            "z_traj"     — (B, T, d_latent) deterministic mean trajectory
            "z_samples"  — (S, B, T, d_latent) MC samples (if return_samples)
            "t_grid"     — (T,) absolute time grid
            "drift_norm" — (B, T-1) mean |drift| per step (diagnostic)
        """
        B = z0.shape[0]
        if clinical is None:
            clinical = z0.new_zeros((B, 1))
        T = self.n_time_grid
        dt = self.integration_time / max(T - 1, 1)
        # v19 Phase 7: prefer the registered buffer (canonical or legacy)
        # so this forward never reconstructs the grid; downstream code can
        # equality-test sde.t_grid against survival_head.t_grid.
        t_grid = self.t_grid.to(device=z0.device)

        # Deterministic mean trajectory.
        z_traj = [z0]
        drift_norms = []
        z = z0
        for _ in range(T - 1):
            z_next = self._step(z, graph_emb, drug, clinical, dt, deterministic=True)
            drift_norms.append((z_next - z).norm(dim=-1))
            z = z_next
            z_traj.append(z)
        z_traj = torch.stack(z_traj, dim=1)                        # (B, T, d_latent)
        drift_norm = torch.stack(drift_norms, dim=1) if drift_norms else None

        out = {"z_traj": z_traj, "t_grid": t_grid, "drift_norm": drift_norm}
        if return_samples and self.n_mc_samples > 0:
            samples = []
            for _ in range(self.n_mc_samples):
                z_s = [z0]
                z = z0
                for _ in range(T - 1):
                    z = self._step(z, graph_emb, drug, clinical, dt, deterministic=False)
                    z_s.append(z)
                samples.append(torch.stack(z_s, dim=1))
            out["z_samples"] = torch.stack(samples, dim=0)         # (S, B, T, d_latent)
        return out
