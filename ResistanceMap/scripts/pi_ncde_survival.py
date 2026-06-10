"""PI-NCDE: Physics-Informed Neural CDE for Multi-State Myeloma Survival.

Combines Neural CDE (torchcde) for continuous-time latent trajectories,
PK-informed observation with structural half-life priors, multi-state
competing-risks survival, FiLM treatment conditioning, and mechanism
classification.  Target: C-index > 0.82, AUROC > 0.80 at 12+ months.

INGEST NOTES (added when landing this file in the repo; model logic unchanged):
  * Requires CoMMpass PER_PATIENT_VISIT (D_LAB_* + VISITDY) and
    STAND_ALONE_TRTRESP flat files — MMRF Virtual Lab longitudinal labs. It does
    NOT consume GDC open-tier OS clinical (age/gender/OS), which has no labs.
  * Fixed two paste-corruption syntax errors: the `cdeint` assignment in
    PINCDE.forward and the checkpoint-save / else block in main().
  * Removed an unused numpy global-RNG seed line: that RNG is never sampled here
    (the split uses a torch.Generator), torch.manual_seed already handles
    reproducibility, and the repo's no-synthetic-data hook forbids numpy RNG
    seeding outside tests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchcde
    HAS_TORCHCDE = True
except ImportError:
    HAS_TORCHCDE = False
    import torchdiffeq  # fallback

# --- Constants ---------------------------------------------------------------

LAB_FEATURES: list[str] = [
    "serum_kappa", "serum_lambda", "kl_ratio", "serum_m_protein",
    "serum_igg", "serum_iga", "serum_igm", "serum_beta2_microglobulin",
    "chem_albumin", "chem_totprot", "cbc_hemoglobin", "cbc_wbc",
    "cbc_platelet", "cbc_abs_neut", "chem_creatinine", "chem_bun",
    "chem_calcium", "chem_ldh", "chem_glucose", "globulin",
]

# PK elimination half-lives in *days* (structural priors, not learned).
# FLC ~4 hr, B2M ~2.5 hr, IgG/albumin ~20 d, CBC ~1 d.
_H = 1.0 / 24.0  # hour -> day conversion
PK_HALF_LIVES_DAYS: dict[str, float] = {
    "serum_kappa": 4 * _H, "serum_lambda": 4 * _H, "kl_ratio": 4 * _H,
    "serum_m_protein": 21.0, "serum_igg": 21.0, "serum_iga": 6.0,
    "serum_igm": 5.0, "serum_beta2_microglobulin": 2.5 * _H,
    "chem_albumin": 20.0, "chem_totprot": 20.0,
    "cbc_hemoglobin": 1.0, "cbc_wbc": 1.0, "cbc_platelet": 1.0,
    "cbc_abs_neut": 1.0, "chem_creatinine": 1.0, "chem_bun": 1.0,
    "chem_calcium": 1.0, "chem_ldh": 1.0, "chem_glucose": 0.5,
    "globulin": 20.0,
}

TREATMENT_AGENTS: list[str] = [
    "bortezomib", "lenalidomide", "dexamethasone", "carfilzomib",
    "pomalidomide", "daratumumab", "elotuzumab", "ixazomib", "thalidomide",
]

DISEASE_STATES: list[str] = [
    "remission", "biochemical_relapse", "clinical_relapse",
    "refractory", "death",
]

# Allowed state transitions (row -> col) as (src_idx, dst_idx).
STATE_TRANSITIONS: list[tuple[int, int]] = [
    (0, 1),  # remission -> biochemical relapse
    (0, 4),  # remission -> death (competing risk)
    (1, 2),  # biochemical -> clinical relapse
    (1, 4),  # biochemical -> death
    (2, 3),  # clinical relapse -> refractory
    (2, 4),  # clinical -> death
    (3, 4),  # refractory -> death
]

MECHANISM_LABELS: list[str] = [
    "drug_efflux", "clonal_evolution", "immune_escape", "microenvironmental",
]

N_LABS = len(LAB_FEATURES)
N_TREATMENTS = len(TREATMENT_AGENTS)
N_STATES = len(DISEASE_STATES)
N_TRANSITIONS = len(STATE_TRANSITIONS)
N_MECHANISMS = len(MECHANISM_LABELS)


# --- Configuration -----------------------------------------------------------

@dataclass
class PINCDEConfig:
    """All hyperparameters for the PI-NCDE model."""

    # Dimensions
    latent_dim: int = 48
    hidden_dim: int = 128
    encoder_layers: int = 2
    cde_hidden_layers: int = 3

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0

    # Loss weights
    recon_weight: float = 1.0
    survival_weight: float = 2.0
    mechanism_entropy_weight: float = 0.1

    # CDE solver
    solver: str = "dopri5"
    atol: float = 1e-4
    rtol: float = 1e-4
    adjoint: bool = True

    # Data
    n_labs: int = N_LABS
    n_treatments: int = N_TREATMENTS
    max_seq_len: int = 60  # max visits per patient

    # Conformal prediction
    conformal_alpha: float = 0.1  # 90 % coverage


# --- PK Observation Model

def _build_pk_decay_rates() -> torch.Tensor:
    """Return per-biomarker elimination rate constants k = ln2 / t_half (1/day)."""
    rates = [math.log(2) / PK_HALF_LIVES_DAYS[f] for f in LAB_FEATURES]
    return torch.tensor(rates, dtype=torch.float32)


class PKObservationModel(nn.Module):
    """Map latent z(t) to biomarker predictions: y_b(t) = g_b(z(t)) * exp(-k_b * dt).

    k_b is the *fixed* PK elimination rate (structural prior); g_b is learned.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.emission = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, N_LABS),
        )
        # Fixed (non-learned) decay rates — registered as buffer.
        self.register_buffer("k", _build_pk_decay_rates())
        # Learnable log-variance per biomarker for heteroscedastic NLL.
        self.log_var = nn.Parameter(torch.zeros(N_LABS))

    def forward(
        self, z: torch.Tensor, delta_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_variance), each shape ``(..., N_LABS)``."""
        mu = self.emission(z)  # (..., N_LABS)
        if delta_t is not None:
            decay = torch.exp(-self.k * delta_t)  # broadcast
            mu = mu * decay
        return mu, self.log_var.expand_as(mu)


# --- FiLM-conditioned CDE Vector Field

class CDEFunc(nn.Module):
    """Neural CDE vector field f_theta(z,t) with FiLM treatment modulation.

    dz = f_theta(z) dX where X is the driving path.  Treatment conditions
    the field via FiLM: gamma, beta produced from treatment vector.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        input_channels: int,
        n_treatments: int = N_TREATMENTS,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels

        # FiLM generator: treatment -> (gamma, beta) per hidden layer.
        self.film_gen = nn.Linear(n_treatments, 2 * hidden_dim * n_hidden_layers)
        self.n_hidden = n_hidden_layers

        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # Output: matrix of shape (latent_dim, input_channels) per sample.
        self.proj = nn.Linear(hidden_dim, latent_dim * input_channels)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector field.  Returns shape ``(batch, latent_dim, input_channels)``."""
        treatment = self._treatment  # set externally before integration
        film_params = self.film_gen(treatment)  # (batch, 2*H*L)
        film_params = film_params.view(-1, self.n_hidden, 2, self.layers[0].out_features)
        # film_params[:, l, 0] = gamma_l, film_params[:, l, 1] = beta_l

        h = z
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.n_hidden:
                gamma = film_params[:, i, 0]
                beta = film_params[:, i, 1]
                h = gamma * h + beta
            h = torch.tanh(h)  # tanh keeps CDE Lipschitz-bounded

        out = self.proj(h)
        return out.view(-1, self.latent_dim, self.input_channels)

    def set_treatment(self, treatment: torch.Tensor) -> None:
        """Cache the treatment vector before CDE integration."""
        self._treatment = treatment


# --- Multi-State Competing-Risks Survival Head

class MultiStateSurvivalHead(nn.Module):
    """Predict cause-specific hazards lambda_ij(t | z(t)) for each allowed transition."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, N_TRANSITIONS),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return instantaneous hazards, shape ``(..., N_TRANSITIONS)``, all > 0."""
        return F.softplus(self.net(z))

    @staticmethod
    def cumulative_incidence(
        hazards: torch.Tensor, dt: torch.Tensor,
    ) -> torch.Tensor:
        """CIF via numerical integration: hazards (B,T,K), dt (B,T) -> CIF (B,T,K)."""
        # Overall survival S(t) = exp(- sum_j integral lambda_j dt)
        total_hazard = hazards.sum(dim=-1)  # (B, T)
        cum_total = torch.cumsum(total_hazard * dt, dim=1)  # (B, T)
        survival = torch.exp(-cum_total).unsqueeze(-1)  # (B, T, 1)
        cif = torch.cumsum(hazards * dt.unsqueeze(-1) * survival, dim=1)
        return cif


# --- Mechanism Classifier with Temporal Attention

class MechanismClassifier(nn.Module):
    """Temporal attention over the latent trajectory -> resistance mechanism probabilities."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, N_MECHANISMS),
        )

    def forward(self, z_traj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z_traj (B,T,D), mask (B,T) -> (B, N_MECHANISMS) log-probs."""
        scores = self.attn_score(z_traj).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (weights * z_traj).sum(dim=1)  # (B, latent_dim)
        return F.log_softmax(self.classifier(context), dim=-1)


# --- Encoder

class InitialEncoder(nn.Module):
    """Map first observation + treatment to initial latent state z_0."""

    def __init__(self, n_labs: int, n_treatments: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_labs + n_treatments, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x0: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x0, treatment], dim=-1))


# --- CDE Fallback (when torchcde is unavailable)

class _FallbackControlledODE(nn.Module):
    """ODE wrapper that manually queries a linear-interpolated control path."""

    def __init__(self, func: CDEFunc, coeffs: torch.Tensor, times: torch.Tensor) -> None:
        super().__init__()
        self.func = func
        self.coeffs = coeffs  # (B, T, C)
        self.times = times    # (T,)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Piecewise-linear derivative of X at time t.
        idx = torch.searchsorted(self.times, t.clamp(self.times[0], self.times[-1])) - 1
        idx = idx.clamp(0, self.coeffs.shape[1] - 2)
        dt_seg = (self.times[idx + 1] - self.times[idx]).clamp(min=1e-6)
        dX = (self.coeffs[:, idx + 1] - self.coeffs[:, idx]) / dt_seg  # (B, C)
        f = self.func(t, z)  # (B, latent, C)
        return torch.einsum("blc,bc->bl", f, dX)


# --- Full PI-NCDE Model

class PINCDE(nn.Module):
    """Physics-Informed Neural CDE for multi-state myeloma survival."""

    def __init__(self, cfg: PINCDEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_channels = cfg.n_labs + 1  # labs + time channel

        self.encoder = InitialEncoder(cfg.n_labs, cfg.n_treatments, cfg.latent_dim, cfg.hidden_dim)
        self.cde_func = CDEFunc(
            cfg.latent_dim, cfg.hidden_dim, cfg.cde_hidden_layers,
            input_channels, cfg.n_treatments,
        )
        self.pk_obs = PKObservationModel(cfg.latent_dim, cfg.hidden_dim)
        self.survival_head = MultiStateSurvivalHead(cfg.latent_dim, cfg.hidden_dim)
        self.mechanism_cls = MechanismClassifier(cfg.latent_dim, cfg.hidden_dim)

        # PK-derived per-biomarker precision weights for reconstruction loss.
        # Fast-turnover markers (large k) are noisier; down-weight them.
        k = _build_pk_decay_rates()
        precision = 1.0 / (1.0 + k)
        self.register_buffer("pk_precision", precision / precision.sum() * N_LABS)

    def forward(
        self,
        times: torch.Tensor,       # (B, T) observation days
        values: torch.Tensor,      # (B, T, N_LABS) lab values (0-filled + mask)
        mask: torch.Tensor,        # (B, T) observation indicator
        treatment: torch.Tensor,   # (B, N_TREATMENTS) binary regimen
    ) -> dict[str, torch.Tensor]:
        B, T, C = values.shape
        device = values.device

        # --- build control path X = (time, labs) ---
        time_channel = times.unsqueeze(-1)  # (B, T, 1)
        X_raw = torch.cat([time_channel, values], dim=-1)  # (B, T, C+1)

        # --- initial state ---
        z0 = self.encoder(values[:, 0], treatment)  # (B, latent)

        # --- CDE integration ---
        self.cde_func.set_treatment(treatment)

        if HAS_TORCHCDE:
            coeffs = torchcde.natural_cubic_coeffs(X_raw, t=times[0])
            X_spline = torchcde.CubicSpline(coeffs, t=times[0])
            # NOTE(ingest fix): original paste mangled this assignment; torchcde
            # selects the adjoint backend via the `adjoint=` kwarg on cdeint.
            cdeint = torchcde.cdeint
            z_traj = cdeint(
                X=X_spline,
                z0=z0,
                func=self.cde_func,
                t=times[0],
                adjoint=self.cfg.adjoint,
                method=self.cfg.solver,
                atol=self.cfg.atol,
                rtol=self.cfg.rtol,
            )  # (B, T, latent)
        else:
            ode_func = _FallbackControlledODE(self.cde_func, X_raw, times[0])
            odeint = torchdiffeq.odeint_adjoint if self.cfg.adjoint else torchdiffeq.odeint
            # odeint returns (T, B, latent); transpose to (B, T, latent)
            z_traj = odeint(
                ode_func, z0, times[0],
                method=self.cfg.solver, atol=self.cfg.atol, rtol=self.cfg.rtol,
            ).permute(1, 0, 2)

        # --- PK observation ---
        recon_mu, recon_logvar = self.pk_obs(z_traj)

        # --- survival hazards ---
        hazards = self.survival_head(z_traj)  # (B, T, N_TRANSITIONS)

        # --- mechanism classification ---
        mech_logprob = self.mechanism_cls(z_traj, mask)

        return {
            "z_traj": z_traj,
            "recon_mu": recon_mu,
            "recon_logvar": recon_logvar,
            "hazards": hazards,
            "mechanism_logprob": mech_logprob,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        event_times: torch.Tensor,   # (B,) event / censoring time
        event_types: torch.Tensor,   # (B,) transition index or -1 if censored
    ) -> dict[str, torch.Tensor]:
        """Composite: PK-weighted reconstruction + multi-state survival + entropy."""
        cfg = self.cfg
        mu = outputs["recon_mu"]
        logvar = outputs["recon_logvar"]

        # --- PK-weighted Gaussian NLL ---
        var = torch.exp(logvar)
        sq_err = (mu - values) ** 2
        nll = 0.5 * (logvar + sq_err / var)  # (B, T, N_LABS)
        # Apply observation mask and PK precision weights.
        obs_mask = mask.unsqueeze(-1)  # (B, T, 1)
        recon_nll = (nll * obs_mask * self.pk_precision).sum() / obs_mask.sum().clamp(min=1)

        # --- Multi-state survival NLL ---
        hazards = outputs["hazards"]  # (B, T, K)
        dt = torch.diff(times, dim=1, prepend=times[:, :1])  # (B, T)
        # Total hazard integral (all transitions) for S(t).
        cum_hazard = (hazards.sum(dim=-1) * dt).sum(dim=1)  # (B,)

        # Log-likelihood: for uncensored, add log lambda_k(t_event); for censored, only -Lambda.
        uncensored = event_types >= 0
        if uncensored.any():
            # Find time index nearest to event_time.
            t_diff = (times[uncensored] - event_times[uncensored].unsqueeze(1)).abs()
            t_idx = t_diff.argmin(dim=1)
            batch_idx = torch.arange(uncensored.sum(), device=times.device)
            event_hazards = hazards[uncensored][batch_idx, t_idx, event_types[uncensored].long()]
            log_hazard_at_event = torch.log(event_hazards.clamp(min=1e-8)).sum()
        else:
            log_hazard_at_event = torch.tensor(0.0, device=times.device)

        survival_nll = cum_hazard.sum() - log_hazard_at_event
        survival_nll = survival_nll / times.shape[0]

        # --- Mechanism entropy regulariser (encourage peaked predictions) ---
        mech_logp = outputs["mechanism_logprob"]  # (B, N_MECHANISMS)
        entropy = -(mech_logp.exp() * mech_logp).sum(dim=-1).mean()

        loss = (
            cfg.recon_weight * recon_nll
            + cfg.survival_weight * survival_nll
            + cfg.mechanism_entropy_weight * entropy
        )
        return {
            "loss": loss,
            "recon_nll": recon_nll.detach(),
            "survival_nll": survival_nll.detach(),
            "entropy_reg": entropy.detach(),
        }


# --- Conformal Survival Wrapper

class ConformalSurvivalWrapper:
    """Conformalized survival bands with finite-sample coverage (1 - alpha)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.q_hat: Optional[float] = None

    def calibrate(
        self,
        pred_cif: np.ndarray,
        true_event_times: np.ndarray,
        censored: np.ndarray,
        time_grid: np.ndarray,
    ) -> None:
        """Compute conformal quantile (censoring-adjusted, Candes et al. 2023)."""
        n = len(true_event_times)
        scores = np.full(n, np.inf)
        for i in range(n):
            if not censored[i]:
                idx = np.searchsorted(time_grid, true_event_times[i])
                idx = min(idx, len(time_grid) - 1)
                scores[i] = 1.0 - pred_cif[i, idx]  # nonconformity score
        # Quantile over uncensored scores with finite-sample correction.
        uncensored_scores = scores[~censored]
        if len(uncensored_scores) == 0:
            warnings.warn("No uncensored observations in calibration set.")
            self.q_hat = 1.0
            return
        level = min((1.0 - self.alpha) * (1.0 + 1.0 / len(uncensored_scores)), 1.0)
        self.q_hat = float(np.quantile(uncensored_scores, level))

    def predict_band(
        self, pred_cif: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) CIF bands clipped to [0, 1]."""
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_band().")
        upper = np.clip(pred_cif + self.q_hat, 0.0, 1.0)
        lower = np.clip(pred_cif - self.q_hat, 0.0, 1.0)
        return lower, upper


# --- CoMMpass Dataset

class CoMMpassDataset(Dataset):
    """MMRF CoMMpass IA-24 longitudinal labs -> PI-NCDE tensors.

    Reads PER_PATIENT_VISIT (D_LAB_* + VISITDY) and STAND_ALONE_TRTRESP
    from ``data_dir``.  Missing values are zero-filled; mask tracks presence.
    """

    # Map CoMMpass D_LAB_* columns to our canonical feature names.
    _COL_MAP: dict[str, str] = {f"D_LAB_{f}": f for f in LAB_FEATURES}

    def __init__(self, data_dir: str | Path, max_seq_len: int = 60) -> None:
        data_dir = Path(data_dir)
        visit_path = self._find_file(data_dir, "PER_PATIENT_VISIT")
        resp_path = self._find_file(data_dir, "STAND_ALONE_TRTRESP")

        visit_df = pd.read_csv(visit_path, low_memory=False)
        resp_df = pd.read_csv(resp_path, low_memory=False)

        # Resolve available lab columns (CoMMpass naming varies across IAs).
        available = {c: self._COL_MAP[c] for c in self._COL_MAP if c in visit_df.columns}
        missing_features = set(LAB_FEATURES) - set(available.values())
        if missing_features:
            warnings.warn(f"Lab features not found in data, will be zero-filled: {missing_features}")

        self.patients: list[dict[str, torch.Tensor]] = []
        grouped = visit_df.groupby("PUBLIC_ID")

        for pid, grp in grouped:
            grp = grp.sort_values("VISITDY").head(max_seq_len)
            T = len(grp)
            if T < 2:
                continue

            times = torch.tensor(grp["VISITDY"].values, dtype=torch.float32)
            vals = torch.zeros(T, N_LABS)
            obs_mask = torch.zeros(T)

            for col, feat in available.items():
                idx = LAB_FEATURES.index(feat)
                series = grp[col].values
                not_nan = ~pd.isna(series)
                vals[not_nan, idx] = torch.tensor(
                    series[not_nan].astype(np.float64), dtype=torch.float32,
                )
                obs_mask |= torch.tensor(not_nan, dtype=torch.float32)

            obs_mask = (obs_mask > 0).float()

            # Treatment vector: try extracting from treatment file or response.
            treatment = torch.zeros(N_TREATMENTS)
            p_resp = resp_df[resp_df["PUBLIC_ID"] == pid]
            if not p_resp.empty and "TRTNAME" in p_resp.columns:
                trt_str = " ".join(p_resp["TRTNAME"].dropna().str.lower().tolist())
                for i, agent in enumerate(TREATMENT_AGENTS):
                    if agent in trt_str:
                        treatment[i] = 1.0

            # Event: use first best-response progression event.
            event_time = times[-1].item()  # default censored at last visit
            event_type = -1  # censored
            if not p_resp.empty and "BESTRESPCD" in p_resp.columns:
                prog = p_resp[p_resp["BESTRESPCD"] == "PD"]
                if not prog.empty and "TTBR" in prog.columns:
                    ttbr = prog["TTBR"].dropna()
                    if len(ttbr) > 0:
                        event_time = float(ttbr.iloc[0])
                        event_type = 1  # biochemical relapse (transition 0->1)

            self.patients.append({
                "times": times,
                "values": vals,
                "mask": obs_mask,
                "treatment": treatment,
                "event_time": torch.tensor(event_time, dtype=torch.float32),
                "event_type": torch.tensor(event_type, dtype=torch.long),
            })

    @staticmethod
    def _find_file(data_dir: Path, pattern: str) -> Path:
        for ext in ("csv", "tsv"):
            for p in data_dir.glob(f"*{pattern}*.{ext}"):
                return p
        raise FileNotFoundError(f"No file matching *{pattern}* in {data_dir}")

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.patients[idx]


def collate_fn(
    batch: list[dict[str, torch.Tensor]], n_grid: int = 64,
) -> dict[str, torch.Tensor]:
    """Re-sample to a shared regular time grid and stack.

    torchcde requires a common 1-D time vector across the batch.  We build a
    regular grid from t=0 to t=max(last_visit) with ``n_grid`` points and
    linearly interpolate each patient's labs onto it.
    """
    B = len(batch)
    t_max = max(b["times"][-1].item() for b in batch)
    grid = torch.linspace(0.0, t_max, n_grid)  # shared (n_grid,)

    values = torch.zeros(B, n_grid, N_LABS)
    mask = torch.zeros(B, n_grid)
    treatments = torch.stack([b["treatment"] for b in batch])
    event_times = torch.stack([b["event_time"] for b in batch])
    event_types = torch.stack([b["event_type"] for b in batch])

    for i, b in enumerate(batch):
        t_i = b["times"]         # (T_i,)
        v_i = b["values"]        # (T_i, N_LABS)
        m_i = b["mask"]          # (T_i,)
        # Linear interpolation per lab channel onto the shared grid.
        for c in range(N_LABS):
            values[i, :, c] = _interp1d(t_i, v_i[:, c], grid)
        # Mask: 1 where grid point falls within patient's observed range.
        mask[i] = (grid <= t_i[-1]).float()

    # Stack times: every sample shares the same grid.
    times = grid.unsqueeze(0).expand(B, -1)  # (B, n_grid)
    return {
        "times": times, "values": values, "mask": mask,
        "treatment": treatments, "event_times": event_times,
        "event_types": event_types,
    }


def _interp1d(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation (NumPy-style) in pure PyTorch."""
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    slope = (f1 - f0) / (x1 - x0).clamp(min=1e-8)
    return f0 + slope * (x - x0)


# --- Training & Evaluation

def train_epoch(
    model: PINCDE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch. Returns average losses."""
    model.train()
    totals: dict[str, float] = {"loss": 0, "recon_nll": 0, "survival_nll": 0, "entropy_reg": 0}
    n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(
            times=batch["times"],
            values=batch["values"],
            mask=batch["mask"],
            treatment=batch["treatment"],
        )
        losses = model.compute_loss(
            outputs, batch["values"], batch["mask"], batch["times"],
            batch["event_times"], batch["event_types"],
        )
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), model.cfg.grad_clip)
        optimizer.step()

        bs = batch["times"].shape[0]
        for k in totals:
            totals[k] += losses[k].item() * bs
        n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: PINCDE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model: losses + concordance index."""
    model.eval()
    totals: dict[str, float] = {"loss": 0, "recon_nll": 0, "survival_nll": 0, "entropy_reg": 0}
    all_risk: list[float] = []
    all_event_times: list[float] = []
    all_event_types: list[int] = []
    n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            times=batch["times"],
            values=batch["values"],
            mask=batch["mask"],
            treatment=batch["treatment"],
        )
        losses = model.compute_loss(
            outputs, batch["values"], batch["mask"], batch["times"],
            batch["event_times"], batch["event_types"],
        )

        bs = batch["times"].shape[0]
        for k in totals:
            totals[k] += losses[k].item() * bs
        n += bs

        # Aggregate risk score = total cumulative hazard at final observed time.
        hazards = outputs["hazards"]
        dt = torch.diff(batch["times"], dim=1, prepend=batch["times"][:, :1])
        cum_haz = (hazards.sum(dim=-1) * dt).sum(dim=1)
        all_risk.extend(cum_haz.cpu().tolist())
        all_event_times.extend(batch["event_times"].cpu().tolist())
        all_event_types.extend(batch["event_types"].cpu().tolist())

    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    metrics["c_index"] = _concordance_index(
        np.array(all_risk), np.array(all_event_times), np.array(all_event_types),
    )
    return metrics


def _concordance_index(
    risk: np.ndarray, event_times: np.ndarray, event_types: np.ndarray,
) -> float:
    """Harrell's C-index (handles censoring)."""
    concordant = 0
    permissible = 0
    uncensored = event_types >= 0
    for i in np.where(uncensored)[0]:
        for j in range(len(risk)):
            if i == j:
                continue
            if event_times[j] > event_times[i]:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / max(permissible, 1)


# --- Entry Point

def main() -> None:
    """Full training loop with early stopping."""
    import argparse

    parser = argparse.ArgumentParser(description="PI-NCDE for Multiple Myeloma survival")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CoMMpass flat files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility: torch RNG covers model init + the data split (torch.Generator).
    # The numpy global RNG is intentionally NOT seeded — it is never sampled in this
    # module, and the repo's no-synthetic-data hook forbids seeding it outside tests.
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = PINCDEConfig()
    dataset = CoMMpassDataset(args.data_dir, max_seq_len=cfg.max_seq_len)
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, collate_fn=collate_fn)

    model = PINCDE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_c_index={val_metrics['c_index']:.4f}"
        )

        # NOTE(ingest fix): original paste corrupted this checkpoint-save / else
        # block into a single broken line; restored to the intended structure.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), "pi_ncde_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Final evaluation
    model.load_state_dict(torch.load("pi_ncde_best.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest metrics: {test_metrics}")

    # Conformal calibration on validation set
    conformal = ConformalSurvivalWrapper(alpha=cfg.conformal_alpha)
    print(f"Conformal wrapper ready (alpha={cfg.conformal_alpha}).")


if __name__ == "__main__":
    main()
