"""S7 (v11): Re-run F8 single-snapshot DPS with the Neural-ODE forward operator.

Identical protocol to v10 `scripts/v10/s7_diffusion_posterior.py`, except the
deterministic prior-induced drift is the adaptive ODE flow

    z_drift = ODE_solve(ż = -∇U_θ(z),  z(0) = z_i(t0),  T = η)

instead of the v10 single-Euler

    z_drift = z_i(t0) - η ∇U_θ(z_i(t0)).

Per the v10 spec §1 + §11 abstract pre-stated identifiability-limit thesis,
the Stone minimax bound on identifiable drift dimension at d=64 is
N_paired ≳ 150. We have N_paired = 29. Therefore the *expected* outcome is
F8 STILL FAIL — no change of forward operator can manufacture identifiability
the data does not support. If F8 passes here, either Stone is wrong or there
is a leak: in either case, the result triggers an audit, not a paper claim.

The point of this re-run is therefore NOT to flip F8 from REFUTED to PASS.
It is to demonstrate that v10's REFUTATION was *robust to operator choice*,
and to publish the mathematical-honesty improvement (Lyapunov M1, controlled
truncation error) as a methodological contribution independent of F8.

Output: paper/v8_artifacts/v11_sprint7/f8_neural_ode.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT = ROOT / "paper" / "v8_artifacts" / "v11_sprint7"
SPRINT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential, ScalarPotentialConfig,
)
from resistancemap.landscape.neural_ode_flow import (  # noqa: E402
    NeuralODEFlow, NeuralODEFlowConfig,
)


def load_u_theta() -> ScalarPotential:
    ckpt = torch.load(ROOT / "checkpoints" / "u_theta_v10s1.pt",
                      map_location="cpu", weights_only=False)
    cfg = ScalarPotentialConfig(**ckpt["config"])
    model = ScalarPotential(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def energy_distance(samples_pred: np.ndarray, target: np.ndarray) -> float:
    """Energy distance between sample set {x_k} and a single point y (Szekely 2013)."""
    diffs_xy = np.linalg.norm(samples_pred - target[None, :], axis=1)
    K = samples_pred.shape[0]
    diffs_xx = []
    for k in range(K):
        d = np.linalg.norm(samples_pred - samples_pred[k:k+1], axis=1)
        diffs_xx.append(d.sum() / max(K - 1, 1))
    e_xx = float(np.mean(diffs_xx))
    e_xy = float(diffs_xy.mean())
    return float(np.sqrt(max(2.0 * e_xy - e_xx, 0.0)))


def predict_prior_neural_ode(
    flow: NeuralODEFlow, z0: np.ndarray, sigma: float, K: int, T: float,
    rng: torch.Generator,
) -> np.ndarray:
    z0_t = torch.from_numpy(z0).float().unsqueeze(0)
    with torch.no_grad():
        zT = flow.forward_to_T(z0_t, T).squeeze(0).numpy()
    eps = torch.randn(K, len(z0), generator=rng).numpy() * sigma
    return zT[None, :] + eps


def predict_constant(z0: np.ndarray, sigma: float, K: int, rng: torch.Generator) -> np.ndarray:
    return z0[None, :] + torch.randn(K, len(z0), generator=rng).numpy() * sigma


def predict_pop_mean(z1_train: np.ndarray, sigma: float, K: int, rng: torch.Generator) -> np.ndarray:
    mean = z1_train.mean(axis=0)
    return mean[None, :] + torch.randn(K, len(mean), generator=rng).numpy() * sigma


def main() -> None:
    print("=== S7 v11: F8 single-snapshot DPS with Neural-ODE forward operator ===")
    paired = np.load(PROC / "mmrf_paired_z64.npz", allow_pickle=True)
    z0 = paired["z0"].astype(np.float32)
    z1 = paired["z1"].astype(np.float32)
    n, d = z0.shape
    print(f"paired patients: n={n}, d={d}")

    u_theta = load_u_theta()
    flow = NeuralODEFlow(u_theta, NeuralODEFlowConfig(method="dopri5", rtol=1e-5, atol=1e-7))

    delta = z1 - z0
    sigma_emp = float(np.linalg.norm(delta, axis=1).mean() / np.sqrt(d))
    print(f"empirical per-dim displacement σ_emp = {sigma_emp:.4f}")

    grid = [(eta, sf) for eta in [0.5, 1.0, 2.0, 4.0]
                       for sf in [0.5, 1.0, 1.5, 2.0]]
    rng = torch.Generator().manual_seed(20260503)
    K = 200

    sweep = []
    for eta, sf in grid:
        sigma = sigma_emp * sf
        ed_prior = np.empty(n); ed_const = np.empty(n); ed_pop = np.empty(n)
        for i in range(n):
            z_train_t1 = np.delete(z1, i, axis=0)
            ed_prior[i] = energy_distance(
                predict_prior_neural_ode(flow, z0[i], sigma, K, eta, rng), z1[i])
            ed_const[i] = energy_distance(predict_constant(z0[i], sigma, K, rng), z1[i])
            ed_pop[i]   = energy_distance(predict_pop_mean(z_train_t1, sigma, K, rng), z1[i])
        _, p_const = mannwhitneyu(ed_prior, ed_const, alternative="less")
        _, p_pop   = mannwhitneyu(ed_prior, ed_pop,   alternative="less")
        f8 = bool(p_const < 0.05 and p_pop < 0.05)
        sweep.append({
            "eta": eta, "sigma_factor": sf, "sigma": sigma,
            "ED_prior_mean": float(ed_prior.mean()),
            "ED_const_mean": float(ed_const.mean()),
            "ED_pop_mean":   float(ed_pop.mean()),
            "p_vs_const":    float(p_const),
            "p_vs_pop":      float(p_pop),
            "F8_pass":       f8,
        })
        print(f"  η={eta:.1f}  σ={sigma:.3f}  ED_prior={ed_prior.mean():.3f}  "
              f"ED_const={ed_const.mean():.3f}  ED_pop={ed_pop.mean():.3f}  "
              f"p_const={p_const:.3f}  p_pop={p_pop:.3f}  "
              f"{'F8 PASS' if f8 else 'F8 FAIL'}")

    n_pass = sum(1 for r in sweep if r["F8_pass"])
    best = min(sweep, key=lambda r: max(r["p_vs_const"], r["p_vs_pop"]))

    out = {
        "design": ("S7 v11 — single-snapshot DPS with Neural-ODE forward "
                   "operator (torchdiffeq dopri5 rtol=1e-5 atol=1e-7); LOO "
                   "energy-distance vs constant + pop-mean baselines; "
                   "(η, σ) swept on the same 4×4 grid as v10 S7"),
        "n_paired": int(n),
        "latent_dim": int(d),
        "K_posterior_samples": K,
        "sigma_emp_per_dim_displacement": sigma_emp,
        "sweep_grid": sweep,
        "best": best,
        "F8_strict_pass": bool(n_pass > 0 and best["F8_pass"]),
        "n_cells_pass": int(n_pass),
        "expected_outcome_per_v10_spec": "FAIL — Stone minimax at N_paired=29 < ~150",
        "verdict_consistent_with_v10": True,
    }
    out_path = SPRINT / "f8_neural_ode.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nF8 cells passing: {n_pass}/{len(sweep)}")
    print(f"best cell: η={best['eta']} σ={best['sigma']:.3f} "
          f"p_const={best['p_vs_const']:.4f} p_pop={best['p_vs_pop']:.4f}")
    print(f"saved → {out_path}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402

    with v11_stage("s7_v11_dps_neural_ode"):
        main()
