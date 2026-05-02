"""
First-Principles Mathematical Reasoning for ResistanceMap
==========================================================

This module implements fundamental theoretical bounds and stability analysis
from information theory, dynamical systems, network topology, and cognitive
science. Every method is grounded in peer-reviewed mathematics.

Key Insight: We don't just make predictions—we prove bounds on what's
theoretically possible, then verify our model respects those bounds.

References:
  [1] Cover & Thomas (2006). Elements of Information Theory (2nd ed). Wiley.
  [2] Strogatz (2018). Nonlinear Dynamics and Chaos (2nd ed). CRC Press.
  [3] Barabasi & Albert (1999). Emergence of scaling in random networks. Science.
  [4] Tversky (1977). Features of Similarity. Psychological Review.
  [5] Alberts et al. (2015). Molecular Biology of the Cell (6th ed). Garland.
  [6] Lane (2016). Power, Sex, Suicide: Mitochondria and the Meaning of Life. OUP.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform


class InformationTheoreticBounds:
    """Information-theoretic bounds on prediction accuracy."""

    @staticmethod
    def compute_fano_bound(features, labels, n_classes):
        features = features.detach().cpu()
        labels = labels.detach().cpu()
        unique_labels, counts = torch.unique(labels, return_counts=True)
        py = counts.float() / len(labels)
        hy = -(py * torch.log2(py + 1e-10)).sum().item()
        k = min(3, len(features) // n_classes)
        hyx = InformationTheoreticBounds._knn_entropy_estimate(features, labels, k, n_classes)
        log_classes = np.log2(n_classes)
        fano_bound = max(0, (hyx - 1) / log_classes) if log_classes > 0 else 0
        achievable_acc = 1 - fano_bound
        return {
            "conditional_entropy": float(hyx),
            "marginal_entropy": float(hy),
            "fano_bound": float(fano_bound),
            "achievable_accuracy_upper": float(achievable_acc),
            "n_classes": n_classes,
            "n_samples": len(features),
        }

    @staticmethod
    def _knn_entropy_estimate(features, labels, k, n_classes):
        features_np = features.cpu().numpy()
        dist_matrix = squareform(pdist(features_np, metric='euclidean'))
        hyx = 0.0
        n_samples = len(features)
        for i in range(n_samples):
            neighbors_idx = np.argsort(dist_matrix[i])[1:k+1]
            neighbor_labels = labels[neighbors_idx].numpy()
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            p = counts / len(neighbor_labels)
            h_neighbors = -np.sum(p * np.log2(p + 1e-10))
            hyx += h_neighbors
        return hyx / n_samples

    @staticmethod
    def mutual_information(features, labels):
        features = features.detach().cpu()
        labels = labels.detach().cpu()
        unique_labels, counts = torch.unique(labels, return_counts=True)
        py = counts.float() / len(labels)
        hy = -(py * torch.log2(py + 1e-10)).sum().item()
        k = min(3, len(features) // len(unique_labels))
        hyx = InformationTheoreticBounds._knn_entropy_estimate(features, labels, k, len(unique_labels))
        mi = hy - hyx
        return max(0, float(mi))

    @staticmethod
    def information_bottleneck_bound(input_dim, latent_dim, output_classes):
        compression_ratio = input_dim / latent_dim
        target_info = np.log2(output_classes)
        latent_capacity = latent_dim * np.log2(np.e)
        is_feasible = latent_capacity >= target_info
        margin = latent_capacity - target_info
        return {
            "compression_ratio": float(compression_ratio),
            "bits_per_sample": float(target_info),
            "max_latent_capacity": float(latent_capacity),
            "is_feasible": bool(is_feasible),
            "estimated_margin": float(margin),
        }


class DynamicalSystemsTheory:
    """Dynamical systems analysis for epigenetic memory and stability."""

    @staticmethod
    def compute_lyapunov_exponents(jacobian):
        jacobian = jacobian.detach().cpu().float()
        eigenvalues = torch.linalg.eigvals(jacobian)
        real_parts = eigenvalues.real
        sorted_idx = torch.argsort(real_parts, descending=True)
        eigenvalues_sorted = eigenvalues[sorted_idx]
        max_real = real_parts.max().item()
        is_stable = max_real < 0
        return {
            "eigenvalues": eigenvalues_sorted,
            "is_stable": bool(is_stable),
            "stability_margin": float(max_real),
            "real_parts": eigenvalues_sorted.real,
            "imag_parts": eigenvalues_sorted.imag,
        }

    @staticmethod
    def basin_depth(jacobian):
        jacobian = jacobian.detach().cpu().float()
        eigenvalues = torch.linalg.eigvals(jacobian)
        max_real = eigenvalues.real.max().item()
        depth = -max_real
        return max(0, float(depth))

    @staticmethod
    def transition_rate_kramers(
        basin_depth: float,
        noise_level: float,
        barrier_height: float = 1.0,
        temperature: float = 1.0,
        barrier_curvature: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute Kramers escape rate from bistable potential.

        Kramers Escape Rate:
            k(T) = (omega_a * omega_b) / (2*pi) * exp(-DeltaV / kT)

        where:
            omega_a = sqrt(V''(x_min))  -- frequency at potential minimum
            omega_b = sqrt(|V''(x_barrier)|) -- frequency at saddle point
            DeltaV = barrier_height
            kT ~ noise_level^2 (fluctuation-dissipation theorem)

        Args:
            basin_depth: Lyapunov exponent magnitude (stability rate).
            noise_level: Stochastic fluctuation magnitude (sigma).
            barrier_height: Energy barrier between states (default 1.0).
            temperature: Effective temperature scaling (default 1.0).
            barrier_curvature: |V''(x_barrier)| at the saddle point. If None,
                estimated as 2 * barrier_height (parabolic barrier approximation).

        Returns:
            Dict with escape_rate, mean_time_to_transition, barrier_exponent,
            frequency_factor, omega_a, omega_b.
        """
        # FIX #3: Compute omega_b properly as sqrt(|V''(x_barrier)|),
        # a property of the POTENTIAL LANDSCAPE, not of the noise.
        #
        # BEFORE (BUGGY):
        #   omega_a = max(basin_depth, 1e-6)
        #   omega_b = noise_level + 1e-6      <-- WRONG: conflates barrier
        #                                          curvature with noise amplitude
        #
        # In Kramers theory:
        #   omega_a = sqrt(V''(x_min)) -- curvature at the potential minimum
        #   omega_b = sqrt(|V''(x_barrier)|) -- curvature at the saddle point
        # Both are geometric properties of the energy landscape.
        # The noise only enters through the Boltzmann factor exp(-DV/kT).
        #
        # AFTER (FIXED):
        #   omega_a = sqrt(basin_depth) -- derived from eigenvalue magnitude
        #   omega_b = sqrt(barrier_curvature) -- from potential second derivative
        #   kT = temperature * noise_level^2 / 2 (fluctuation-dissipation)

        omega_a = max(np.sqrt(max(basin_depth, 1e-12)), 1e-6)

        if barrier_curvature is not None:
            omega_b = max(np.sqrt(max(barrier_curvature, 1e-12)), 1e-6)
        else:
            # Parabolic barrier approximation: |V''(x_b)| ~ 2 * DeltaV
            omega_b = max(np.sqrt(max(2.0 * barrier_height, 1e-12)), 1e-6)

        frequency_factor = (omega_a * omega_b) / (2 * np.pi)

        # kT from fluctuation-dissipation: sigma^2 = 2*kT/gamma, with gamma=1
        effective_temp = temperature * (noise_level ** 2 / 2.0) + 1e-10
        barrier_exponent = barrier_height / effective_temp

        escape_rate = frequency_factor * np.exp(-min(barrier_exponent, 500.0))
        mean_transition_time = 1.0 / escape_rate if escape_rate > 1e-300 else np.inf

        return {
            "escape_rate": float(escape_rate),
            "mean_time_to_transition": float(mean_transition_time),
            "barrier_exponent": float(barrier_exponent),
            "frequency_factor": float(frequency_factor),
            "omega_a": float(omega_a),
            "omega_b": float(omega_b),
        }

    @staticmethod
    def waddington_potential(
        states: torch.Tensor,
        velocities: torch.Tensor,
        bandwidth: Optional[float] = None,
    ) -> torch.Tensor:
        """Reconstruct Waddington quasi-potential from steady-state distribution.

        The quasi-potential is defined via Boltzmann inversion:
            V(x) = -log(P_ss(x))
        where P_ss(x) is the steady-state probability density estimated via KDE.

        Args:
            states: [n_samples, n_dims] trajectory of system states.
            velocities: [n_samples, n_dims] velocities (kept for API compat,
                not used -- potential comes from state density, not kinetic energy).
            bandwidth: Optional KDE bandwidth. If None, uses Scott's rule.

        Returns:
            Estimated quasi-potential V(x) at each state point.
        """
        states = states.detach().cpu().float()
        # FIX #4: Compute proper quasi-potential via KDE, not kinetic energy.
        #
        # BEFORE (BUGGY):
        #   dot_products = (velocities ** 2).sum(dim=1)  # = ||v||^2
        #   potential = -torch.cumsum(dot_products * dt, dim=0)
        #
        # This computed cumulative KINETIC ENERGY, which is highest at barrier
        # crossings (fast movement) and lowest at attractors (slow/stationary).
        # That is the OPPOSITE of the potential landscape.
        #
        # AFTER (FIXED):
        #   V(x) = -log(P_ss(x)) via Gaussian KDE of the state trajectory.
        #   P_ss is high at attractors (frequently visited) => V is low there.
        #   P_ss is low at barriers (rarely visited) => V is high there.

        n_samples, n_dims = states.shape
        states_np = states.numpy()

        if n_dims <= 3:
            try:
                kde = stats.gaussian_kde(
                    states_np.T,
                    bw_method=bandwidth if bandwidth is not None else 'scott'
                )
                log_density = kde.logpdf(states_np.T)
                potential = torch.tensor(-log_density, dtype=torch.float32)
            except (np.linalg.LinAlgError, ValueError):
                potential = _kde_fallback(states, bandwidth)
        else:
            potential = _high_dim_kde_potential(states, bandwidth)

        potential = potential - potential.min()
        return potential


def _kde_fallback(states, bandwidth=None):
    """Histogram-based fallback when scipy KDE fails."""
    n_samples, n_dims = states.shape
    log_density = torch.zeros(n_samples)
    for d in range(n_dims):
        x = states[:, d]
        n_bins = max(10, int(np.sqrt(n_samples)))
        hist, bin_edges = torch.histogram(x, bins=n_bins)
        hist = hist.float() + 1e-10
        hist = hist / hist.sum()
        bin_idx = torch.bucketize(x, bin_edges[1:-1])
        bin_idx = torch.clamp(bin_idx, 0, len(hist) - 1)
        log_density += torch.log(hist[bin_idx])
    return -log_density


def _high_dim_kde_potential(states, bandwidth=None):
    """KDE quasi-potential via PCA projection for high-dim states."""
    n_samples, n_dims = states.shape
    mean = states.mean(dim=0)
    centered = states - mean
    n_components = min(2, n_dims, n_samples)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vh[:n_components].T
    projected_np = projected.numpy()
    try:
        kde = stats.gaussian_kde(
            projected_np.T,
            bw_method=bandwidth if bandwidth is not None else 'scott'
        )
        log_density = kde.logpdf(projected_np.T)
        potential = torch.tensor(-log_density, dtype=torch.float32)
    except (np.linalg.LinAlgError, ValueError):
        potential = _kde_fallback(projected, bandwidth)
    return potential


class NetworkTopologyTheory:
    """Network topology analysis for STRING PPI and drug target selection."""

    @staticmethod
    def compute_degree_distribution(edge_index, n_nodes):
        edge_index = edge_index.detach().cpu().long()
        degrees = torch.zeros(n_nodes, dtype=torch.long)
        degrees.scatter_add_(0, edge_index[0], 1)
        degrees.scatter_add_(0, edge_index[1], 1)
        deg_counts = torch.bincount(degrees)
        degree_dist = deg_counts.float() / deg_counts.sum()
        k_vals = torch.arange(1, len(degree_dist))
        p_vals = degree_dist[1:]
        valid_idx = p_vals > 0
        if valid_idx.sum() > 2:
            log_k = torch.log(k_vals[valid_idx].float())
            log_p = torch.log(p_vals[valid_idx])
            mean_log_k = log_k.mean()
            mean_log_p = log_p.mean()
            numerator = ((log_k - mean_log_k) * (log_p - mean_log_p)).sum()
            denominator = ((log_k - mean_log_k) ** 2).sum()
            if denominator > 1e-10:
                slope = numerator / denominator
                gamma = -slope.item()
            else:
                gamma = 0.0
        else:
            gamma = 0.0
        return {
            "degrees": degrees,
            "degree_distribution": degree_dist,
            "power_law_exponent": float(gamma),
            "is_scale_free": 2.0 <= gamma <= 3.0,
            "average_degree": float(degrees.float().mean().item()),
            "max_degree": int(degrees.max().item()),
        }

    @staticmethod
    def compute_pagerank(edge_index, n_nodes, damping=0.85, n_iterations=30):
        edge_index = edge_index.detach().cpu().long()
        out_degrees = torch.zeros(n_nodes, dtype=torch.float)
        out_degrees.scatter_add_(0, edge_index[0], 1.0)
        out_degrees[out_degrees == 0] = 1.0
        pagerank = torch.ones(n_nodes) / n_nodes
        for _ in range(n_iterations):
            new_rank = (1 - damping) / n_nodes * torch.ones(n_nodes)
            for src, dst in edge_index.t():
                src, dst = src.item(), dst.item()
                new_rank[dst] += damping * pagerank[src] / out_degrees[src]
            pagerank = new_rank
        return pagerank

    @staticmethod
    def identify_network_motifs(edge_index, n_nodes, motif_type="feed_forward_loop"):
        edge_index = edge_index.detach().cpu().long()
        adjacency = {i: set() for i in range(n_nodes)}
        for src, dst in edge_index.t():
            adjacency[src.item()].add(dst.item())
        motifs = []
        if motif_type == "feed_forward_loop":
            for a in range(n_nodes):
                for b in adjacency[a]:
                    for c in adjacency[b]:
                        if c in adjacency[a] and a != c:
                            motifs.append((a, b, c))
        elif motif_type in ("feedback_loop", "bistable_switch"):
            for a in range(n_nodes):
                for b in adjacency[a]:
                    if a in adjacency[b]:
                        motifs.append((a, b))
        return motifs

    @staticmethod
    def target_druggability_score(pagerank, degree, betweenness=None):
        pagerank = pagerank.detach().cpu().float()
        degree = degree.detach().cpu().float()
        pr_norm = (pagerank - pagerank.min()) / (pagerank.max() - pagerank.min() + 1e-10)
        deg_norm = (degree - degree.min()) / (degree.max() - degree.min() + 1e-10)
        if betweenness is not None:
            betweenness = betweenness.detach().cpu().float()
            bc_norm = (betweenness - betweenness.min()) / (betweenness.max() - betweenness.min() + 1e-10)
        else:
            bc_norm = torch.zeros_like(pr_norm)
        druggability = 0.4 * pr_norm + 0.3 * deg_norm + 0.3 * bc_norm
        return druggability / druggability.max()


class CognitiveDecisionTheory:
    """Tversky's similarity and decision frameworks for target ranking."""

    @staticmethod
    def tversky_similarity(features_a, features_b, alpha=1.0, beta=1.0):
        features_a = features_a.detach().cpu().float()
        features_b = features_b.detach().cpu().float()
        a_binary = (features_a > 0.5).float()
        b_binary = (features_b > 0.5).float()
        intersection = (a_binary * b_binary).sum().item()
        a_only = (a_binary * (1 - b_binary)).sum().item()
        b_only = (b_binary * (1 - a_binary)).sum().item()
        denominator = intersection + alpha * a_only + beta * b_only
        if denominator < 1e-10:
            return 0.0
        return float(intersection / denominator)

    @staticmethod
    def elimination_by_aspects(candidates, feature_names, feature_importance, n_rounds=3, elimination_threshold=0.5):
        candidates = candidates.detach().cpu().float()
        feature_importance = feature_importance.detach().cpu().float()
        importance = feature_importance / feature_importance.sum()
        sorted_idx = torch.argsort(importance, descending=True)
        active = set(range(len(candidates)))
        scores = torch.zeros(len(candidates))
        for round_idx in range(min(n_rounds, len(sorted_idx))):
            feature_idx = sorted_idx[round_idx].item()
            feature_scores = candidates[:, feature_idx]
            active_scores = torch.tensor([feature_scores[i].item() for i in active])
            if len(active_scores) > 0:
                threshold = torch.quantile(active_scores, elimination_threshold)
                to_remove = {idx for idx in active if feature_scores[idx].item() < threshold.item()}
                active = active - to_remove
                for idx in active:
                    scores[idx] += importance[feature_idx] * feature_scores[idx]
        remaining = sorted([(idx, scores[idx].item()) for idx in active], key=lambda x: x[1], reverse=True)
        return remaining
