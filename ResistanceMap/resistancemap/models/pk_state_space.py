"""
PK-Constrained State-Space Model for MM Relapse Prediction
==========================================================
Core innovation: Encodes known biomarker pharmacokinetics as structural priors,
making the model inherently interpretable while reducing parameter space.

Architecture:
  Latent state x_t ∈ R^k represents disease phase:
    - x[0]: tumor_burden (log-scale plasma cell mass surrogate)
    - x[1]: clonal_diversity (heterogeneity of Ig-secreting clones)
    - x[2]: immune_function (T/NK cell competence)
    - x[3]: renal_function (GFR proxy)
    - x[4]: inflammatory_state (acute phase response)

  Transition: x_{t+1} = f_θ(x_t, u_t) + ε_t
    where u_t = treatment regimen encoding

  Observation: y_t = g_PK(x_t) + η_t
    where g_PK encodes pharmacokinetic constraints:
      - FLC components use 2-6hr effective half-life
      - M-protein uses ~21-day IgG half-life
      - B2M uses renal clearance model
      - CBC components use hematopoietic production/destruction model

  Hazard: λ(t|x_t) = λ_0(t) · exp(h_θ(x_t))
    with competing risks for biochemical vs clinical progression

Author: Abhignya Jagathpally
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, NamedTuple
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PKSSMConfig:
    """Configuration for the PK State-Space Model."""
    # Latent state
    latent_dim: int = 5          # {tumor_burden, clonal_diversity, immune, renal, inflammatory}
    hidden_dim: int = 64         # MLP hidden dimension

    # Biomarker panel
    n_biomarkers: int = 20       # Number of observed biomarker features
    n_treatments: int = 9        # Treatment encoding dimension (from ml_mmrf)

    # Pharmacokinetic priors (hours, converted to observation-interval units internally)
    flc_halflife_hours: float = 4.0       # Free light chain serum half-life
    igg_halflife_days: float = 21.0       # IgG (M-protein carrier) half-life
    iga_halflife_days: float = 6.0        # IgA half-life
    igm_halflife_days: float = 5.0        # IgM half-life
    albumin_halflife_days: float = 20.0   # Albumin half-life
    b2m_halflife_hours: float = 2.5       # Beta-2 microglobulin renal clearance half-life
    observation_interval_days: float = 60.0  # CoMMpass granularity (2-month intervals)

    # Training
    dropout: float = 0.1
    n_competing_risks: int = 2   # biochemical progression, clinical progression

    # Interpretability
    n_resistance_mechanisms: int = 4  # {drug_efflux, clonal_evolution, immune_escape, microenvironment}


# =============================================================================
# Biomarker Index Map
# =============================================================================

BIOMARKER_NAMES = [
    'serum_kappa', 'serum_lambda', 'kl_ratio',          # FLC panel
    'serum_m_protein',                                    # M-protein
    'serum_igg', 'serum_iga', 'serum_igm',              # Immunoglobulins
    'serum_beta2_microglobulin',                          # B2M
    'chem_albumin', 'chem_totprot',                       # Protein panel
    'cbc_hemoglobin', 'cbc_wbc', 'cbc_platelet',         # CBC
    'cbc_abs_neut',                                       # Differential
    'chem_creatinine', 'chem_bun', 'chem_calcium',       # Chemistry
    'chem_ldh', 'chem_glucose',                           # Metabolic
    'globulin',                                           # Derived: totprot - albumin
]

# Which latent dimensions drive which biomarkers (structural prior)
# This is the KEY interpretability mechanism
LATENT_TO_BIOMARKER_MAP = {
    'tumor_burden': [0, 1, 2, 3, 4, 5, 6, 7, 10, 17],   # FLCs, M-protein, Igs, B2M, Hb, LDH
    'clonal_diversity': [0, 1, 2, 4, 5, 6],               # FLC ratio shifts, Ig isotype changes
    'immune_function': [11, 12, 13],                       # WBC, platelets, neutrophils
    'renal_function': [7, 14, 15],                         # B2M, creatinine, BUN
    'inflammatory_state': [8, 9, 16, 19],                  # Albumin, total protein, calcium, globulin
}


# =============================================================================
# Pharmacokinetic Observation Model
# =============================================================================

class PKObservationModel(nn.Module):
    """
    Maps latent disease state to expected biomarker values using
    pharmacokinetically-informed transformations.

    Key insight: Different biomarkers have different temporal response
    characteristics to the SAME underlying disease change. FLC responds
    in hours; M-protein takes weeks to reflect the same change. This is
    not learned — it's a known physical constraint.
    """

    def __init__(self, config: PKSSMConfig):
        super().__init__()
        self.config = config

        # Compute decay constants per observation interval
        # decay = exp(-ln(2) * interval / half_life)
        interval = config.observation_interval_days
        self.register_buffer('flc_decay', torch.tensor(
            math.exp(-math.log(2) * interval * 24 / config.flc_halflife_hours)))
        self.register_buffer('igg_decay', torch.tensor(
            math.exp(-math.log(2) * interval / config.igg_halflife_days)))
        self.register_buffer('iga_decay', torch.tensor(
            math.exp(-math.log(2) * interval / config.iga_halflife_days)))
        self.register_buffer('igm_decay', torch.tensor(
            math.exp(-math.log(2) * interval / config.igm_halflife_days)))
        self.register_buffer('albumin_decay', torch.tensor(
            math.exp(-math.log(2) * interval / config.albumin_halflife_days)))
        self.register_buffer('b2m_decay', torch.tensor(
            math.exp(-math.log(2) * interval * 24 / config.b2m_halflife_hours)))

        # Learnable mapping from latent state to biomarker production rates
        # This captures HOW MUCH tumor burden affects FLC production, etc.
        self.production_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_biomarkers),
        )

        # Learnable observation noise (log-scale)
        self.log_obs_noise = nn.Parameter(torch.zeros(config.n_biomarkers) - 1.0)

        # Decay constants per biomarker (structural prior + learnable residual)
        # Start from PK priors, allow small learned corrections
        pk_decays = torch.ones(config.n_biomarkers) * 0.5  # default
        pk_decays[0] = self.flc_decay     # kappa
        pk_decays[1] = self.flc_decay     # lambda
        pk_decays[2] = 1.0                # ratio (instantaneous)
        pk_decays[3] = self.igg_decay     # M-protein (usually IgG)
        pk_decays[4] = self.igg_decay     # IgG
        pk_decays[5] = self.iga_decay     # IgA
        pk_decays[6] = self.igm_decay     # IgM
        pk_decays[7] = self.b2m_decay     # B2M
        pk_decays[8] = self.albumin_decay # Albumin
        self.register_buffer('pk_decay_prior', pk_decays)

        # Small learnable correction to PK priors (initialized near zero)
        self.decay_residual = nn.Parameter(torch.zeros(config.n_biomarkers) * 0.01)

    def get_effective_decay(self) -> torch.Tensor:
        """PK prior + small learned correction, clamped to valid range."""
        return torch.clamp(self.pk_decay_prior + self.decay_residual, 0.001, 0.999)

    def forward(self, x_t: torch.Tensor, y_prev: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: Latent state [batch, latent_dim]
            y_prev: Previous observation [batch, n_biomarkers] (for AR component)

        Returns:
            y_mean: Expected biomarker values [batch, n_biomarkers]
            y_var: Observation variance [batch, n_biomarkers]
        """
        # Production rate from current disease state
        production = self.production_net(x_t)  # [batch, n_biomarkers]

        decay = self.get_effective_decay()  # [n_biomarkers]

        if y_prev is not None:
            # PK-informed observation: new value = decayed previous + new production
            # This is the discrete-time analog of: dy/dt = production - clearance * y
            y_mean = decay * y_prev + (1 - decay) * production
        else:
            # Initial observation: just production (no history)
            y_mean = production

        y_var = torch.exp(self.log_obs_noise).unsqueeze(0).expand_as(y_mean)

        return y_mean, y_var


# =============================================================================
# Treatment-Conditioned State Transition
# =============================================================================

class StateTransition(nn.Module):
    """
    Models how the latent disease state evolves between observations,
    conditioned on the active treatment regimen.

    x_{t+1} = f_θ(x_t, u_t) + ε_t

    Treatment effects are modeled as perturbations to the transition dynamics,
    not as additive terms. This captures non-linear drug interactions.
    """

    def __init__(self, config: PKSSMConfig):
        super().__init__()
        self.config = config

        # Base transition (disease progression without treatment)
        self.base_transition = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        # Treatment effect modulator
        self.treatment_encoder = nn.Sequential(
            nn.Linear(config.n_treatments, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),  # scale + shift
        )

        # Transition noise (learnable, latent-dim-specific)
        self.log_transition_noise = nn.Parameter(torch.zeros(config.latent_dim) - 2.0)

    def forward(self, x_t: torch.Tensor, u_t: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: Current latent state [batch, latent_dim]
            u_t: Treatment encoding [batch, n_treatments]

        Returns:
            x_next_mean: Expected next state [batch, latent_dim]
            x_next_var: Transition variance [batch, latent_dim]
        """
        # Base dynamics (natural disease progression)
        dx = self.base_transition(x_t)

        # Treatment modulation (FiLM-style: scale and shift)
        treatment_params = self.treatment_encoder(u_t)
        scale, shift = treatment_params.chunk(2, dim=-1)
        scale = torch.sigmoid(scale) * 2  # [0, 2] range
        dx_modulated = scale * dx + shift

        # Residual connection (state persists + changes)
        x_next_mean = x_t + dx_modulated

        x_next_var = torch.exp(self.log_transition_noise).unsqueeze(0).expand_as(x_t)

        return x_next_mean, x_next_var


# =============================================================================
# Mechanism Classifier (Interpretability Head)
# =============================================================================

class MechanismClassifier(nn.Module):
    """
    Maps the latent state trajectory to a resistance mechanism classification.

    Given a sequence of latent states [x_1, ..., x_T], identifies which
    resistance mechanism is most consistent with the observed trajectory:

    1. Drug efflux (ABCB1): M-protein decay flattens while FLC stays elevated
    2. Clonal evolution: FLC isotype shift (kappa↔lambda dominance change)
    3. Immune escape: Neutrophil/lymphocyte ratio shifts, B2M rises independently
    4. Microenvironmental: Slow, uniform biomarker deterioration with albumin drop

    This is the KEY interpretability feature that no existing model provides.
    """

    def __init__(self, config: PKSSMConfig):
        super().__init__()
        self.config = config

        # Temporal attention over state trajectory
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=1,
            batch_first=True,
            dropout=config.dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_resistance_mechanisms),
        )

        self.mechanism_names = [
            'drug_efflux',
            'clonal_evolution',
            'immune_escape',
            'microenvironmental',
        ]

    def forward(self, x_trajectory: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_trajectory: Latent states over time [batch, T, latent_dim]
            mask: Boolean mask [batch, T] where True = valid timepoint

        Returns:
            mechanism_logits: [batch, n_mechanisms]
            attention_weights: [batch, T] — which timepoints drive the prediction
        """
        # Self-attention over temporal sequence
        attn_output, attn_weights = self.temporal_attention(
            x_trajectory, x_trajectory, x_trajectory,
            key_padding_mask=~mask if mask is not None else None,
        )

        # Pool: attention-weighted average
        if mask is not None:
            attn_output = attn_output * mask.unsqueeze(-1).float()
            pooled = attn_output.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = attn_output.mean(dim=1)

        logits = self.classifier(pooled)

        return logits, attn_weights.squeeze(1) if attn_weights.dim() == 3 else attn_weights


# =============================================================================
# Competing Risk Hazard Head (reuses v19 survival module interface)
# =============================================================================

class LatentHazardHead(nn.Module):
    """
    Maps latent disease state to time-varying hazard for competing risks:
    - Risk 1: Biochemical progression (M-protein or FLC-based)
    - Risk 2: Clinical progression (CRAB criteria, new lesions)

    Compatible with the v19 CompetingRiskHead evaluation pipeline.
    """

    def __init__(self, config: PKSSMConfig):
        super().__init__()
        self.config = config

        self.hazard_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_competing_risks),
        )

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Latent state [batch, latent_dim]

        Returns:
            log_hazard: [batch, n_competing_risks]
        """
        return self.hazard_net(x_t)


# =============================================================================
# Full PK-SSM Model
# =============================================================================

class PKSSM(nn.Module):
    """
    Pharmacokinetically-Constrained State-Space Model for MM Relapse Prediction.

    Full forward pass:
    1. Encode initial observation y_0 → x_0 (recognition model)
    2. For t = 1..T:
       a. Transition: x_t = f(x_{t-1}, u_{t-1}) + noise
       b. Observe: y_t ~ g_PK(x_t) + noise
       c. Update: posterior(x_t | y_{1:t}) via filtering
    3. Hazard: λ(t) from x_t trajectory
    4. Mechanism: classify resistance type from x_{1:T}

    Training: ELBO = reconstruction + KL + survival likelihood
    """

    def __init__(self, config: PKSSMConfig):
        super().__init__()
        self.config = config

        # Recognition model: y_0 → x_0
        self.encoder = nn.Sequential(
            nn.Linear(config.n_biomarkers, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.enc_mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.enc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Core components
        self.transition = StateTransition(config)
        self.observation = PKObservationModel(config)
        self.hazard = LatentHazardHead(config)
        self.mechanism = MechanismClassifier(config)

        # Baseline features integration (ISS, age, gender, etc.)
        self.baseline_proj = nn.Linear(17, config.latent_dim)  # ml_mmrf baseline dim

    def encode(self, y_0: torch.Tensor, baseline: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode initial observation to latent state distribution."""
        h = self.encoder(y_0)
        if baseline is not None:
            h = h + self.baseline_proj(baseline)
        return self.enc_mean(h), self.enc_logvar(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(
        self,
        y: torch.Tensor,          # [batch, T, n_biomarkers] — observed lab values
        u: torch.Tensor,          # [batch, T, n_treatments] — treatment at each step
        mask: torch.Tensor,       # [batch, T] — True where observation exists
        baseline: Optional[torch.Tensor] = None,  # [batch, 17]
        event_time: Optional[torch.Tensor] = None, # [batch] — time of event
        event_type: Optional[torch.Tensor] = None,  # [batch] — censoring indicator
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the PK-SSM.

        Returns dict with:
            - recon_loss: reconstruction NLL
            - kl_loss: KL divergence from prior
            - survival_loss: negative log-likelihood of survival
            - mechanism_logits: resistance mechanism classification
            - attention_weights: temporal attention for interpretability
            - x_trajectory: latent states for visualization
            - y_pred: predicted biomarker trajectories
        """
        batch_size, T, _ = y.shape

        # 1. Encode first valid observation → x_0
        # Find first valid timestep per patient
        first_valid = mask.float().argmax(dim=1)  # [batch]
        y_0 = y[torch.arange(batch_size), first_valid]  # [batch, n_biomarkers]

        z_mean, z_logvar = self.encode(y_0, baseline)
        x_t = self.reparameterize(z_mean, z_logvar)

        # KL divergence from standard normal prior
        kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(dim=-1).mean()

        # 2. Roll forward through time
        x_trajectory = []
        y_preds = []
        recon_losses = []
        hazard_sequence = []
        y_prev = y_0

        for t in range(T):
            x_trajectory.append(x_t)

            # Observation model
            y_mean, y_var = self.observation(x_t, y_prev)
            y_preds.append(y_mean)

            # Reconstruction loss (Gaussian NLL, only where observed)
            if mask[:, t].any():
                obs_mask = mask[:, t]  # [batch]
                y_true = y[:, t]       # [batch, n_biomarkers]

                # Handle NaN in observations (missing individual features)
                feature_mask = ~torch.isnan(y_true) & obs_mask.unsqueeze(-1)
                y_true_clean = torch.where(feature_mask, y_true, y_mean.detach())

                nll = 0.5 * ((y_true_clean - y_mean).pow(2) / y_var + y_var.log())
                nll = (nll * feature_mask.float()).sum() / feature_mask.float().sum().clamp(min=1)
                recon_losses.append(nll)

                # Update y_prev with actual observation where available
                y_prev = torch.where(feature_mask, y_true_clean, y_mean)
            else:
                y_prev = y_mean

            # Hazard at this timestep
            log_h = self.hazard(x_t)
            hazard_sequence.append(log_h)

            # Transition to next state (if not last timestep)
            if t < T - 1:
                u_t = u[:, t]
                x_next_mean, x_next_var = self.transition(x_t, u_t)
                x_t = self.reparameterize(x_next_mean,
                                          torch.log(x_next_var + 1e-8))

        # Stack results
        x_trajectory = torch.stack(x_trajectory, dim=1)   # [batch, T, latent_dim]
        y_preds = torch.stack(y_preds, dim=1)              # [batch, T, n_biomarkers]
        hazards = torch.stack(hazard_sequence, dim=1)       # [batch, T, n_risks]
        recon_loss = torch.stack(recon_losses).mean() if recon_losses else torch.tensor(0.0)

        # 3. Survival loss (discrete-time hazard, right-censored)
        survival_loss = torch.tensor(0.0, device=y.device)
        if event_time is not None and event_type is not None:
            survival_loss = self._survival_nll(hazards, event_time, event_type, mask)

        # 4. Mechanism classification (no supervision — soft labels from trajectory shape)
        mechanism_logits, attn_weights = self.mechanism(x_trajectory, mask)

        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'survival_loss': survival_loss,
            'total_loss': recon_loss + 0.1 * kl_loss + survival_loss,
            'mechanism_logits': mechanism_logits,
            'attention_weights': attn_weights,
            'x_trajectory': x_trajectory,
            'y_pred': y_preds,
            'hazards': hazards,
        }

    def _survival_nll(self, hazards: torch.Tensor, event_time: torch.Tensor,
                      event_type: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Discrete-time survival negative log-likelihood with right censoring."""
        # hazards: [batch, T, n_risks], already log-scale
        batch_size, T, n_risks = hazards.shape

        # Clamp event_time to valid range
        event_time = event_time.long().clamp(0, T - 1)

        # For each patient: sum log(1-h_k) for t < event_time (survival)
        # At event_time: add log(h_k) if uncensored
        log_h = F.log_softmax(hazards, dim=-1)  # normalize across risks
        log_1_minus_h = torch.log1p(-torch.exp(log_h).sum(dim=-1) + 1e-8)  # [batch, T]

        # Survival contribution: sum log(1-h) for t < event_time
        time_range = torch.arange(T, device=hazards.device).unsqueeze(0)  # [1, T]
        before_event = (time_range < event_time.unsqueeze(1)) & mask  # [batch, T]
        surv_contrib = (log_1_minus_h * before_event.float()).sum(dim=1)  # [batch]

        # Event contribution (uncensored only)
        uncensored = event_type > 0  # 1=event, 0=censored
        if uncensored.any():
            event_log_h = log_h[torch.arange(batch_size), event_time, 0]  # risk 0 for simplicity
            event_contrib = event_log_h * uncensored.float()
        else:
            event_contrib = torch.zeros(batch_size, device=hazards.device)

        nll = -(surv_contrib + event_contrib).mean()
        return nll

    def predict_trajectory(self, y_0: torch.Tensor, u_sequence: torch.Tensor,
                           baseline: Optional[torch.Tensor] = None,
                           n_steps: int = 6) -> Dict[str, torch.Tensor]:
        """
        Predict future biomarker trajectory from current observation.

        This is the clinical use case: given today's blood work and planned treatment,
        forecast where biomarkers will be in 3/6/12 months and what the relapse
        probability is at each horizon.
        """
        self.eval()
        with torch.no_grad():
            z_mean, _ = self.encode(y_0, baseline)
            x_t = z_mean
            y_prev = y_0

            predictions = []
            hazard_probs = []

            for t in range(n_steps):
                # Predict observation
                y_mean, y_var = self.observation(x_t, y_prev)
                predictions.append(y_mean)

                # Hazard
                log_h = self.hazard(x_t)
                hazard_probs.append(torch.sigmoid(log_h))

                # Transition
                if t < n_steps - 1:
                    u_t = u_sequence[:, t] if u_sequence.dim() == 3 else u_sequence
                    x_next_mean, _ = self.transition(x_t, u_t)
                    x_t = x_next_mean

                y_prev = y_mean

        return {
            'predicted_biomarkers': torch.stack(predictions, dim=1),
            'hazard_probabilities': torch.stack(hazard_probs, dim=1),
        }


# =============================================================================
# Interpretability utilities
# =============================================================================

def explain_prediction(model: PKSSM, result: Dict[str, torch.Tensor],
                       patient_idx: int = 0) -> str:
    """
    Generate human-readable explanation of a prediction.

    This is what makes the model clinically useful: not just "relapse probability = 0.73"
    but "FLC ratio is rising with kinetics consistent with rapid clonal expansion
    (3-hour effective half-life), while M-protein hasn't caught up yet due to its
    21-day clearance — expect M-protein rise in ~6 weeks. Pattern most consistent
    with clonal evolution mechanism."
    """
    mechanism_probs = F.softmax(result['mechanism_logits'][patient_idx], dim=-1)
    mechanism_names = ['Drug efflux', 'Clonal evolution', 'Immune escape', 'Microenvironmental']
    top_mechanism = mechanism_names[mechanism_probs.argmax().item()]
    top_prob = mechanism_probs.max().item()

    # Get attention weights (which timepoints matter most)
    attn = result['attention_weights'][patient_idx]
    critical_timepoint = attn.argmax().item()

    # Get PK decay constants for interpretation
    decay = model.observation.get_effective_decay()
    flc_response_time = -model.config.observation_interval_days / torch.log(decay[0]).item()
    mp_response_time = -model.config.observation_interval_days / torch.log(decay[3]).item()

    explanation = (
        f"Resistance mechanism: {top_mechanism} (confidence: {top_prob:.1%})\n"
        f"Critical timepoint: month {critical_timepoint * 2} (observation #{critical_timepoint})\n"
        f"FLC effective response time: {flc_response_time:.1f} days\n"
        f"M-protein effective response time: {mp_response_time:.1f} days\n"
    )

    return explanation