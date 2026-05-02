"""Module 3: Cell-State Plasticity Framework — Gap 4.2 Multi-Scale Integration.

Integrates transcriptomic noise (fast), epigenetic bistability (slow), and
genetic mutation accumulation (irreversible) into a unified resistance predictor.

This module implements the hierarchical Waddington landscape with three
scales of plasticity:

1. Scale 1 (fast, reversible): Transcriptomic noise via SDE
   - Represents stochastic gene expression fluctuations
   - Timescale: hours to days
   - Reversible by resetting GRN state

2. Scale 2 (slow, bistable): Epigenetic state via double-well potential
   - Driven by chromatin modifier (writer/eraser) abundances
   - Timescale: days to weeks
   - Can flip between attractor basins (semi-reversible)

3. Scale 3 (irreversible): Genetic mutations accumulating
   - Driven by mutation rate μ and fitness landscape
   - Timescale: months to years
   - Irreversible ratchet mechanism

The combined resistance probability is:
    P_resist = α·P_transcriptomic + β·P_epigenetic + γ·P_genetic

where α, β, γ are learnable softmax weights that adapt during training.

References:
    - Waddington, C.H. (1957) "Strategy of the Genes"
    - Klemm & Shivashankar (2019) Cell Systems 9(3):273-284
    - Sneppen & Ringrose (2019) Cell 176(4):728-739
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TranscriptomicNoiseModel(nn.Module):
    """Scale 1 (fast): Stochastic differential equation for gene expression.

    Implements the SDE:
        dX/dt = f(X, θ) + σ(X) · ξ(t)

    where:
        X: gene expression state (D-dimensional)
        f: mean-reverting drift (learned MLP)
        σ: state-dependent noise (learned MLP)
        ξ: white noise process

    The drift naturally pulls toward an attractor (mean-reverting),
    while noise allows transient excursions. This models the rapid,
    reversible fluctuations in gene expression due to stochastic
    transcription and translation.

    Args:
        state_dim: Dimension of gene expression vector (default 32).
        hidden_dim: Hidden layer size for drift/diffusion networks (default 16).
        noise_scale: Global noise amplitude (default 0.1).
        mean_reversion_strength: How strongly X reverts to mean (default 0.5).
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 16,
        noise_scale: float = 0.1,
        mean_reversion_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.noise_scale = noise_scale
        self.mean_reversion_strength = mean_reversion_strength

        # Drift network: f(X) → ℝ^D (mean-reverting force)
        self.drift_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Diffusion network: σ(X) → ℝ^D (positive state-dependent noise)
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),  # Ensure σ > 0
        )

        # Learnable equilibrium state (mean to revert toward)
        self.equilibrium = nn.Parameter(torch.zeros(state_dim))

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean-reverting drift f(X) = -β·(X - X*) + learned_term.

        Args:
            x: (B, D) gene expression state.

        Returns:
            (B, D) drift vector.
        """
        learned_drift = self.drift_net(x)
        mean_reversion = -self.mean_reversion_strength * (x - self.equilibrium)
        return learned_drift + mean_reversion

    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-dependent noise σ(X).

        Args:
            x: (B, D) gene expression state.

        Returns:
            (B, D) diagonal diffusion coefficients.
        """
        return self.noise_scale * self.diffusion_net(x)

    def forward(
        self,
        x0: torch.Tensor,
        t_span: tuple[float, float] = (0.0, 1.0),
        n_steps: int = 50,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Integrate SDE via Euler-Maruyama scheme.

        Args:
            x0: (B, D) initial gene expression state.
            t_span: Time interval (t0, tf).
            n_steps: Number of integration steps.
            return_trajectory: If True, return full trajectory; else final state.

        Returns:
            Final state (B, D) or trajectory (T, B, D).
        """
        batch_size, device = x0.shape[0], x0.device
        dt = (t_span[1] - t_span[0]) / n_steps
        x = x0.clone()
        trajectory = [x0]

        for step in range(n_steps):
            # Drift term
            drift_term = self.drift(x) * dt

            # Diffusion term (Wiener increment)
            diffusion_term = self.diffusion(x) * torch.randn_like(x) * (dt ** 0.5)

            # Euler-Maruyama step
            x = x + drift_term + diffusion_term

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return x

    def compute_transition_probability(
        self, x_start: torch.Tensor, x_target: torch.Tensor, horizon: int = 50
    ) -> torch.Tensor:
        """Estimate probability that SDE reaches target state from start state.

        Uses Monte Carlo sampling: run many trajectories and compute fraction
        that reach within L2 distance threshold of target.

        Args:
            x_start: (B, D) starting states.
            x_target: (B, D) target states.
            horizon: Number of integration steps (MC samples).

        Returns:
            (B,) transition probabilities in [0, 1].
        """
        n_samples = 100
        threshold = 0.5  # L2 distance threshold
        x_start_expanded = x_start.unsqueeze(1).expand(-1, n_samples, -1)
        x_start_expanded = x_start_expanded.reshape(-1, x_start.shape[1])

        x_final = self.forward(x_start_expanded, n_steps=horizon)

        x_final = x_final.reshape(x_start.shape[0], n_samples, -1)
        x_target_expanded = x_target.unsqueeze(1)

        distances = torch.norm(x_final - x_target_expanded, dim=2)  # (B, n_samples)
        reached = (distances < threshold).float()
        prob = reached.mean(dim=1)

        return prob


class EpigeneticBistabilityModel(nn.Module):
    """Scale 2 (slow): Epigenetic state governed by double-well potential.

    Implements the ODE:
        dE/dt = -dV_epi/dE + η

    where:
        E: epigenetic state (e.g., chromatin compaction level)
        V_epi(E) = a·E⁴ - b·E²: quartic double-well potential
        a, b: depend on chromatin modifier (writer/eraser) abundances
        η: thermal noise (weak compared to potential gradient)

    The parameters a and b are functions of chromatin modifiers (EZH2, TET1, etc).
    When a is large relative to b, the potential becomes highly bistable,
    and the system can be trapped in one basin.

    Transition rates between basins follow Kramers escape rate theory:
        k_escape ≈ (ω_a / 2π) · exp(-ΔE_barrier / k_B·T)

    Args:
        equilibrium_dim: Dimension of epigenetic state (default 8).
        hidden_dim: Hidden layer size for potential networks (default 16).
        noise_strength: Thermal noise amplitude (default 0.01).
    """

    def __init__(
        self,
        equilibrium_dim: int = 8,
        hidden_dim: int = 16,
        noise_strength: float = 0.01,
    ) -> None:
        super().__init__()
        self.equilibrium_dim = equilibrium_dim
        self.noise_strength = noise_strength

        # Network to predict potential parameters from modifier abundances
        # Input: chromatin modifiers (e.g., EZH2, DNMT1, TET1 levels)
        # Output: [a, b] for V(E) = a·E⁴ - b·E²
        self.potential_net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 4 chromatin modifiers
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [a, b]
            nn.Softplus(),  # Ensure a, b > 0
        )

        # Learnable thermal temperature (affects noise strength and escape rates)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def potential(self, e: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute double-well potential V(E) = a·E⁴ - b·E².

        Args:
            e: (B, D) epigenetic states.
            a: (B, 1) quartic coefficient.
            b: (B, 1) quadratic coefficient.

        Returns:
            (B, D) potential values.
        """
        return a * (e ** 4) - b * (e ** 2)

    def potential_gradient(
        self, e: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient dV/dE = 4·a·E³ - 2·b·E.

        Args:
            e: (B, D) epigenetic states.
            a: (B, 1) quartic coefficient.
            b: (B, 1) quadratic coefficient.

        Returns:
            (B, D) gradient.
        """
        return 4 * a * (e ** 3) - 2 * b * e

    def forward(
        self,
        e0: torch.Tensor,
        chromatin_modifiers: torch.Tensor,
        t_span: tuple[float, float] = (0.0, 1.0),
        n_steps: int = 50,
    ) -> torch.Tensor:
        """Integrate epigenetic ODE via Euler scheme.

        Args:
            e0: (B, D) initial epigenetic state.
            chromatin_modifiers: (B, 4) abundances of EZH2, DNMT1, TET1, etc.
            t_span: Time interval (t0, tf).
            n_steps: Number of integration steps.

        Returns:
            (B, D) final epigenetic state.
        """
        batch_size, device = e0.shape[0], e0.device
        dt = (t_span[1] - t_span[0]) / n_steps

        # Predict potential parameters
        potential_params = self.potential_net(chromatin_modifiers)  # (B, 2)
        a = potential_params[:, 0:1]
        b = potential_params[:, 1:2]

        e = e0.clone()
        temperature = torch.exp(self.log_temperature)

        for step in range(n_steps):
            # Deterministic part: -dV/dE
            grad_v = self.potential_gradient(e, a, b)
            drift = -grad_v

            # Stochastic part (weak thermal noise)
            noise = self.noise_strength * temperature * torch.randn_like(e)

            # Euler step
            e = e + (drift + noise) * dt

        return e

    def kramers_escape_rate(
        self, chromatin_modifiers: torch.Tensor
    ) -> torch.Tensor:
        """Compute Kramers escape rate for transitions between bistable states.

        Kramers rate: k ≈ ω₀ · exp(-ΔE_barrier / k_B·T)

        where ΔE_barrier ∝ b²/(4a) for V = a·E⁴ - b·E².

        Args:
            chromatin_modifiers: (B, 4) modifier abundances.

        Returns:
            (B,) escape rates (transitions per unit time).
        """
        potential_params = self.potential_net(chromatin_modifiers)
        a = potential_params[:, 0:1]
        b = potential_params[:, 1:2]

        # Barrier height: ΔE_barrier = b² / (4a)
        barrier = (b ** 2) / (4 * a + 1e-8)

        # Temperature affects escape
        temperature = torch.exp(self.log_temperature)

        # Escape rate (exponential dependence on barrier)
        # Prefactor ω₀ ≈ 1.0 (characteristic frequency)
        omega0 = 1.0
        escape_rate = omega0 * torch.exp(-barrier / (temperature + 1e-8))

        return escape_rate.squeeze(-1)

    def transition_probability(
        self, chromatin_modifiers: torch.Tensor, time_horizon: float = 100.0
    ) -> torch.Tensor:
        """Compute probability of epigenetic state transition over time_horizon.

        Uses Kramers rate: P_trans = 1 - exp(-k_escape · t_horizon).

        Args:
            chromatin_modifiers: (B, 4) modifier abundances.
            time_horizon: Duration (in arbitrary time units) for transition.

        Returns:
            (B,) transition probabilities in [0, 1].
        """
        escape_rate = self.kramers_escape_rate(chromatin_modifiers)
        prob = 1.0 - torch.exp(-escape_rate * time_horizon)
        return torch.clamp(prob, 0.0, 1.0)


class GeneticMutationAccumulator(nn.Module):
    """Scale 3 (irreversible): Accumulation of driver mutations.

    Implements:
        dC/dt = μ · h(C)

    where:
        C: cumulative mutation count (integer-valued, treated as continuous)
        μ: per-cell mutation rate (driver-specific, from sequencing data)
        h(C): fitness landscape function (typically sigmoidal saturation)

    This is a ratchet mechanism: mutations only increase, never revert.
    The fitness landscape can accelerate accumulation if early mutations
    create selective pressure for subsequent drivers.

    Args:
        n_drivers: Number of driver mutation loci to track (default 5).
        hidden_dim: Hidden layer size (default 16).
    """

    def __init__(self, n_drivers: int = 5, hidden_dim: int = 16) -> None:
        super().__init__()
        self.n_drivers = n_drivers

        # Network to predict fitness landscape h(C) from mutation counts
        # This allows early mutations to accelerate subsequent accumulation
        self.fitness_net = nn.Sequential(
            nn.Linear(n_drivers, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_drivers),
            nn.Sigmoid(),  # Output in [0, 1] (fitness acceleration factor)
        )

        # Learnable per-driver mutation rates (log-scale for numerical stability)
        self.log_mutation_rates = nn.Parameter(torch.ones(n_drivers) * (-3.0))

    def forward(
        self,
        c0: torch.Tensor,
        base_mutation_rates: torch.Tensor | None = None,
        t_span: tuple[float, float] = (0.0, 1.0),
        n_steps: int = 50,
    ) -> torch.Tensor:
        """Integrate mutation accumulation ODE.

        Args:
            c0: (B, n_drivers) initial cumulative mutation counts.
            base_mutation_rates: (B, n_drivers) optional empirical mutation rates.
            t_span: Time interval (t0, tf).
            n_steps: Number of integration steps.

        Returns:
            (B, n_drivers) final mutation counts (clipped to [0, ∞)).
        """
        batch_size, device = c0.shape[0], c0.device
        dt = (t_span[1] - t_span[0]) / n_steps

        c = c0.clone()
        learned_rates = torch.exp(self.log_mutation_rates).unsqueeze(0)

        # Use provided rates or learned rates
        if base_mutation_rates is not None:
            mutation_rates = base_mutation_rates
        else:
            mutation_rates = learned_rates.expand(batch_size, -1)

        for step in range(n_steps):
            # Fitness landscape: how much does current mutation state accelerate accumulation?
            fitness_factor = self.fitness_net(c)  # (B, n_drivers) in [0, 1]

            # Rate equation: dc/dt = μ · h(C)
            # h(C) = fitness_factor represents epistasis and selective pressure
            dc_dt = mutation_rates * fitness_factor

            # Euler step with clipping (mutations only increase)
            c = torch.clamp(c + dc_dt * dt, min=0.0)

        return c

    def expected_accumulated_mutations(
        self, base_rates: torch.Tensor, time_horizon: float
    ) -> torch.Tensor:
        """Estimate expected mutation count over time_horizon.

        Simple linear model: E[C(t)] = μ · h(0) · t.
        More accurate if rates don't change much over horizon.

        Args:
            base_rates: (B, n_drivers) per-driver mutation rates.
            time_horizon: Time duration.

        Returns:
            (B, n_drivers) expected cumulative mutation counts.
        """
        # Initial fitness landscape (no mutations yet)
        c0 = torch.zeros_like(base_rates)
        fitness_at_origin = self.fitness_net(c0)

        expected_count = base_rates * fitness_at_origin * time_horizon
        return expected_count


class WaddingtonLandscape(nn.Module):
    """Computes the Waddington landscape potential from GRN topology.

    Given a gene regulatory network (GRN) as a weight matrix W,
    computes an implicit potential φ(E) that captures the dynamics:
        dE/dt = -∇φ(E) + noise

    The landscape is reconstructed from:
        1. GRN adjacency/weight matrix (learned or data-derived)
        2. Gene expression levels
        3. Energy-based model: φ(E) = -½ E^T W E + λ·||E||²

    Finds attractor states via gradient descent on the potential.

    Args:
        gene_dim: Dimension of gene expression space (default 32).
        hidden_dim: Hidden layer size (default 16).
    """

    def __init__(self, gene_dim: int = 32, hidden_dim: int = 16) -> None:
        super().__init__()
        self.gene_dim = gene_dim

        # Learnable GRN weight matrix (with regularization for sparsity)
        self.grn_weights = nn.Parameter(torch.randn(gene_dim, gene_dim) * 0.1)

        # Regularization coefficients (learnable)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    def potential(self, e: torch.Tensor) -> torch.Tensor:
        """Compute Waddington landscape potential φ(E).

        φ(E) = -½ E^T W E + λ·||E||²

        The first term (negative quadratic form) encodes GRN interactions.
        The second term is L2 regularization (prevents unbounded growth).

        Args:
            e: (B, D) gene expression states.

        Returns:
            (B,) potential values.
        """
        # GRN interaction term: -½ E^T W E
        grn_term = -0.5 * torch.einsum("bd,dd,bd->b", e, self.grn_weights, e)

        # Regularization term: λ·||E||²
        lamb = torch.exp(self.log_lambda)
        reg_term = lamb * (e ** 2).sum(dim=1)

        return grn_term + reg_term

    def potential_gradient(self, e: torch.Tensor) -> torch.Tensor:
        """Compute gradient ∇φ(E) = -W·E + 2λ·E.

        Args:
            e: (B, D) gene expression states.

        Returns:
            (B, D) gradient.
        """
        lamb = torch.exp(self.log_lambda)
        w_e = torch.matmul(e, self.grn_weights.T)
        return -w_e + 2 * lamb * e

    def find_attractors(
        self, e0: torch.Tensor, n_iterations: int = 100, learning_rate: float = 0.01
    ) -> torch.Tensor:
        """Find attractor states via gradient descent on potential.

        Args:
            e0: (B, D) initial gene expression states.
            n_iterations: Number of gradient descent steps.
            learning_rate: Step size for descent.

        Returns:
            (B, D) attractor states.
        """
        e = e0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([e], lr=learning_rate)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            phi = self.potential(e).sum()
            phi.backward()
            optimizer.step()

        return e.detach()

    def compute_landscape_distance(
        self, e1: torch.Tensor, e2: torch.Tensor
    ) -> torch.Tensor:
        """Compute path-length distance over Waddington landscape.

        Uses geodesic approximation: shortest path along descent trajectory.

        Args:
            e1: (B, D) start state.
            e2: (B, D) end state.

        Returns:
            (B,) distances (higher = more separated in landscape).
        """
        phi1 = self.potential(e1)
        phi2 = self.potential(e2)
        delta_phi = (phi2 - phi1).abs()

        # Geodesic ~ sqrt(ΔE²) + regularization term
        delta_e = torch.norm(e2 - e1, dim=1)
        distance = delta_e + torch.sqrt(delta_phi + 1e-8)

        return distance


# =============================================================================
# FIX #2: ResistanceSignatureBank — replaces random noise target
# =============================================================================

class ResistanceSignatureBank(nn.Module):
    """Bank of drug-class-specific resistance expression signatures.

    Stores learnable resistance phenotype targets for each drug class.
    These signatures represent the characteristic gene expression pattern
    of a resistant cell state for a given drug mechanism of action.

    Biologically, resistance signatures capture:
    - EMT marker upregulation (e.g., VIM, SNAI1/2, ZEB1)
    - ABC transporter overexpression (ABCB1/MDR1, ABCG2)
    - Anti-apoptotic gene activation (BCL2, MCL1)
    - Drug target pathway rewiring

    Each signature is a learnable vector in gene expression space that is
    refined during training to match observed resistant phenotypes.

    Args:
        state_dim: Dimension of gene expression space (must match
            TranscriptomicNoiseModel.state_dim).
        n_drug_classes: Number of distinct drug classes/mechanisms to model.
            Default 8 covers major categories: alkylating agents, antimetabolites,
            topoisomerase inhibitors, mitotic inhibitors, targeted therapy (kinase),
            targeted therapy (other), immunotherapy, hormonal therapy.
    """

    def __init__(self, state_dim: int = 32, n_drug_classes: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_drug_classes = n_drug_classes

        # Learnable resistance signatures per drug class
        # Initialized with small positive values to represent mild upregulation
        # of resistance genes as a biologically reasonable prior.
        self.signatures = nn.Parameter(
            torch.randn(n_drug_classes, state_dim) * 0.1 + 0.5
        )

        # Default (drug-agnostic) resistance signature — a weighted average
        # learned from data when no drug class is specified.
        self.default_signature = nn.Parameter(
            torch.randn(state_dim) * 0.1 + 0.5
        )

        # Network to refine signature based on current cell state context.
        # This allows the resistance target to be context-dependent: the
        # exact resistant phenotype depends on where the cell currently is
        # in expression space.
        self.context_refinement = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

    def get_signature(
        self,
        expression_state: torch.Tensor,
        drug_class_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the resistance expression signature for the given context.

        Args:
            expression_state: (B, state_dim) current gene expression state.
                Used for context-dependent refinement of the signature.
            drug_class_idx: (B,) optional integer indices into drug class bank.
                If None, uses the default (drug-agnostic) learned signature.

        Returns:
            (B, state_dim) resistance target expression signatures.
        """
        batch_size = expression_state.shape[0]

        if drug_class_idx is not None:
            # Look up drug-specific signatures
            base_signature = self.signatures[drug_class_idx]  # (B, state_dim)
        else:
            # Use default signature, expanded to batch
            base_signature = self.default_signature.unsqueeze(0).expand(
                batch_size, -1
            )

        # Context-dependent refinement: the resistant phenotype depends on
        # the current cell state (e.g., lineage, differentiation stage)
        context_input = torch.cat([expression_state, base_signature], dim=-1)
        refinement = self.context_refinement(context_input)

        # Residual connection: base signature + small refinement
        return base_signature + 0.1 * refinement


class MultiScaleResistancePredictor(nn.Module):
    """Integrates three-scale plasticity into unified resistance predictor.

    Combines:
        P_resist = softmax(w) · [P_transcriptomic, P_epigenetic, P_genetic]

    where w are learnable weights that adapt based on tumor context.

    The softmax weighting automatically discovers which plasticity scale
    dominates resistance in a given patient/tumor type.

    Args:
        state_dim: Gene expression dimension (default 32).
        n_drivers: Number of driver mutations (default 5).
        hidden_dim: Hidden layer size (default 16).
        n_drug_classes: Number of drug classes for resistance bank (default 8).
    """

    def __init__(
        self,
        state_dim: int = 32,
        n_drivers: int = 5,
        hidden_dim: int = 16,
        n_drug_classes: int = 8,
    ) -> None:
        super().__init__()
        self.transcriptomic_model = TranscriptomicNoiseModel(
            state_dim=state_dim, hidden_dim=hidden_dim
        )
        self.epigenetic_model = EpigeneticBistabilityModel(
            equilibrium_dim=8, hidden_dim=hidden_dim
        )
        self.genetic_model = GeneticMutationAccumulator(n_drivers=n_drivers, hidden_dim=hidden_dim)
        self.landscape = WaddingtonLandscape(gene_dim=state_dim, hidden_dim=hidden_dim)

        # FIX #2: Replace random target with a learnable ResistanceSignatureBank.
        #
        # BEFORE (BUGGY):
        #   expression_target = torch.randn_like(expression_state)
        #
        # This used random Gaussian noise as the "resistant phenotype" target,
        # meaning the model was training toward RANDOM targets on every forward
        # pass — the transition probability P_transcriptomic was meaningless
        # noise, and gradient signal was non-stationary.
        #
        # AFTER (FIXED):
        #   The ResistanceSignatureBank stores drug-class-specific learned
        #   resistance signatures that are refined during training. This gives
        #   a biologically meaningful, deterministic target that represents
        #   the characteristic gene expression profile of resistant cells.
        self.resistance_bank = ResistanceSignatureBank(
            state_dim=state_dim, n_drug_classes=n_drug_classes
        )

        # Learnable softmax weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(
        self,
        expression_state: torch.Tensor,
        epigenetic_state: torch.Tensor,
        mutation_counts: torch.Tensor,
        chromatin_modifiers: torch.Tensor,
        time_horizon: float = 100.0,
        resistance_signature: Optional[torch.Tensor] = None,
        drug_class_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict multi-scale resistance probability.

        Args:
            expression_state: (B, state_dim) gene expression.
            epigenetic_state: (B, 8) epigenetic configuration.
            mutation_counts: (B, n_drivers) cumulative mutations.
            chromatin_modifiers: (B, 4) chromatin modifier abundances.
            time_horizon: Duration over which to assess resistance.
            resistance_signature: (B, state_dim) optional externally provided
                resistance expression target. If provided, overrides the
                learned signature bank. Useful when differential expression
                signatures are available from experiments.
            drug_class_idx: (B,) optional drug class indices for looking up
                drug-specific resistance signatures from the bank.

        Returns:
            (B,) resistance probabilities in [0, 1].
        """
        # FIX #2: Use biologically meaningful resistance signature as target.
        #
        # Priority order:
        # 1. Externally provided resistance_signature (from experiments)
        # 2. Drug-class-specific signature from the bank
        # 3. Default learned signature from the bank
        if resistance_signature is not None:
            expression_target = resistance_signature
        else:
            expression_target = self.resistance_bank.get_signature(
                expression_state, drug_class_idx=drug_class_idx
            )

        # Scale 1: Transcriptomic noise (can fluctuate to resistant state?)
        p_transcriptomic = self.transcriptomic_model.compute_transition_probability(
            expression_state, expression_target, horizon=50
        )

        # Scale 2: Epigenetic bistability (can flip to other basin?)
        p_epigenetic = self.epigenetic_model.transition_probability(
            chromatin_modifiers, time_horizon=time_horizon
        )

        # Scale 3: Genetic mutations (irreversible accumulation)
        # Resistance ~ probability that mutations reach critical threshold
        base_rates = torch.abs(torch.randn(mutation_counts.shape[0], mutation_counts.shape[1], device=mutation_counts.device)) * 1e-2
        expected_mutations = self.genetic_model.expected_accumulated_mutations(
            base_rates, time_horizon
        )
        total_mutations = (mutation_counts + expected_mutations).sum(dim=1)
        p_genetic = torch.sigmoid(total_mutations - 2.0)  # Threshold at ~2 mutations

        # Combine via learnable softmax weighting
        weights = F.softmax(self.scale_weights, dim=0)
        p_resistance = (
            weights[0] * p_transcriptomic
            + weights[1] * p_epigenetic
            + weights[2] * p_genetic
        )

        return torch.clamp(p_resistance, 0.0, 1.0)


class PlasticityDiagnostics:
    """Compute diagnostic metrics for cell-state plasticity.

    Metrics:
        TNS (Transcriptomic Noise Sensitivity): How much does noise-induced
            fluctuation allow escaping current state?
        ESI (Epigenetic Stability Index): Basin depth / barrier height for
            bistable epigenetic switches.
        GL (Generalized Lyapunov): Global stability of the combined landscape
            (eigenvalues of Jacobian averaged across states).
    """

    @staticmethod
    def transcriptomic_noise_sensitivity(
        model: TranscriptomicNoiseModel, x_samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute TNS: average displacement from noise over fixed time.

        High TNS = more plastic (noise-driven exploration).
        Low TNS = more stable (noise-resistant steady state).

        Args:
            model: TranscriptomicNoiseModel instance.
            x_samples: (B, D) expression states to evaluate.

        Returns:
            Tuple of (tns_values, displacements):
                tns_values: (B,) sensitivity scores in [0, ∞).
                displacements: (B,) average L2 displacement.
        """
        x_original = x_samples.clone()

        # Integrate with high noise (stochastic)
        x_perturbed = model.forward(x_original, t_span=(0.0, 1.0), n_steps=50)

        displacements = torch.norm(x_perturbed - x_original, dim=1)
        return displacements, displacements

    @staticmethod
    def epigenetic_stability_index(
        model: EpigeneticBistabilityModel, chromatin_modifiers: torch.Tensor
    ) -> torch.Tensor:
        """Compute ESI: (barrier height) / (basin width).

        High ESI = stable bistable switch (high barrier relative to basin size).
        Low ESI = unstable, easily flipped.

        Args:
            model: EpigeneticBistabilityModel instance.
            chromatin_modifiers: (B, 4) modifier abundances.

        Returns:
            (B,) ESI values in [0, 1].
        """
        potential_params = model.potential_net(chromatin_modifiers)
        a = potential_params[:, 0:1]
        b = potential_params[:, 1:2]

        # Barrier height: ΔE_barrier = b² / (4a)
        barrier = (b ** 2) / (4 * a + 1e-8)

        # Basin width (distance between minima): w ~ sqrt(b/a)
        basin_width = torch.sqrt(b / (a + 1e-8))

        # ESI = barrier / basin_width (normalized)
        esi = torch.sigmoid(barrier / (basin_width + 1e-8))

        return esi.squeeze(-1)

    @staticmethod
    def generalized_lyapunov(
        landscape: WaddingtonLandscape, e_samples: torch.Tensor, eps: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GL: dominant eigenvalue of potential Hessian (stability).

        Negative eigenvalue = attractor (stable).
        Positive eigenvalue = repellor (unstable).
        Magnitude ~ convergence rate.

        Args:
            landscape: WaddingtonLandscape instance.
            e_samples: (B, D) gene expression samples.
            eps: Finite difference step size.

        Returns:
            Tuple of (dominant_eigenvalues, gl_scores):
                dominant_eigenvalues: (B,) max eigenvalue at each sample.
                gl_scores: (B,) normalized stability in [0, 1].
        """
        batch_size, dim = e_samples.shape
        device = e_samples.device

        # Numerical Hessian via finite differences (expensive, only use for diagnostics)
        grad0 = landscape.potential_gradient(e_samples)  # (B, D)

        hessians = torch.zeros(batch_size, dim, dim, device=device)

        for j in range(min(dim, 5)):  # Compute Hessian of first 5 dims only for efficiency
            e_perturbed = e_samples.clone()
            e_perturbed[:, j] = e_perturbed[:, j] + eps
            grad_plus = landscape.potential_gradient(e_perturbed)
            hessians[:, :, j] = (grad_plus - grad0) / eps

        # Eigenvalues of symmetric part of Hessian
        hessians_sym = (hessians + hessians.transpose(-1, -2)) / 2.0

        try:
            eigenvalues = torch.linalg.eigvalsh(hessians_sym)  # (B, D)
            dominant_eigs = eigenvalues[:, -1]  # Most positive (unstable) or least negative (stable)
        except RuntimeError:
            # Fallback if eigendecomposition fails
            dominant_eigs = torch.zeros(batch_size, device=device)

        # Stability score: negative eigs are stable, positive are unstable
        # GL in [0, 1]: 1 = very stable (large negative), 0 = unstable (positive)
        gl_scores = torch.sigmoid(-dominant_eigs)

        return dominant_eigs, gl_scores
