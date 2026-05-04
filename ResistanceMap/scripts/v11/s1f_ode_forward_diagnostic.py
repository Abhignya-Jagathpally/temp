"""S1f (v11): Diagnostic for the Neural-ODE forward operator over U_θ.

Three diagnostics, each independently publishable:

(D1) Energy-decay (Lyapunov M1).
     For gradient flow ż = -∇U, dU/dt = -|∇U|² ≤ 0. We integrate from each of
     the N=29 paired-patient z0 latents over t∈[0, T] for T ∈ {0.5, 1, 2, 4}
     and check U(z(T)) ≤ U(z0). A failure means the ODE solver is mis-
     configured (rtol too loose for the local Lipschitz constant) or the
     potential has numerical artifacts.

(D2) Single-Euler vs fixed-step-Euler vs Dopri5 truncation error.
     We treat adaptive Dopri5 (rtol=1e-5, atol=1e-7) as ground truth and
     report ‖z_method − z_ref‖ at the same final time T. The Euler chain at
     n=1 step is the v10 operator; we expect O(h²) local error and an
     amplification with η. This quantifies *how wrong* the v10 operator is
     for the η ∈ {0.5, 1, 2, 4} regime that v10 swept in S7.

(D3) Drift direction agreement.
     Cosine similarity between Δz_v10 = -η ∇U(z0) and Δz_v11 = z(T)-z0 with
     T=η. If cos < 0.9 for the η=2 swept point that v10 reported, the v10
     F8-DPS test was *answering a question the operator could not honestly
     answer* — and the v11 result is the proper test.

Output: paper/v8_artifacts/v11_sprint1/ode_forward_diagnostic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CHK = ROOT / "checkpoints"
OUT_DIR = ROOT / "paper" / "v8_artifacts" / "v11_sprint1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential, ScalarPotentialConfig,
)
from resistancemap.landscape.neural_ode_flow import (  # noqa: E402
    NeuralODEFlow, NeuralODEFlowConfig, truncation_error,
)


def load_u_theta(tag: str = "v10s1") -> ScalarPotential:
    ckpt = torch.load(CHK / f"u_theta_{tag}.pt", map_location="cpu", weights_only=False)
    cfg = ScalarPotentialConfig(**ckpt["config"])
    model = ScalarPotential(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1).clamp_min(1e-12) * b.norm(dim=-1).clamp_min(1e-12))


def main() -> None:
    print("=== S1f v11: Neural-ODE forward operator diagnostic ===")
    paired = np.load(PROC / "mmrf_paired_z64.npz", allow_pickle=True)
    Z0 = torch.from_numpy(paired["z0"].astype(np.float32))
    n, d = Z0.shape
    print(f"paired patients (z0 only used here): n={n}, d={d}")

    u_theta = load_u_theta("v10s1")
    flow = NeuralODEFlow(u_theta, NeuralODEFlowConfig(method="dopri5", rtol=1e-5, atol=1e-7))
    print(f"loaded U_θ from checkpoints/u_theta_v10s1.pt")

    out: dict = {
        "reference_solver": {"method": "dopri5", "rtol": 1e-5, "atol": 1e-7},
        "n_paired": int(n), "latent_dim": int(d),
    }

    # D1 — energy decay at each patient, multiple T
    Ts = [0.5, 1.0, 2.0, 4.0]
    energy_log = []
    for T in Ts:
        with torch.no_grad():
            zT = flow.forward_to_T(Z0, T)
            u0 = u_theta.forward(Z0).detach().cpu().numpy()
            uT = u_theta.forward(zT).detach().cpu().numpy()
        delta = (uT - u0)
        n_violations = int((delta > 1e-6).sum())
        energy_log.append({
            "T": T,
            "U0_mean": float(u0.mean()),
            "UT_mean": float(uT.mean()),
            "delta_mean": float(delta.mean()),
            "delta_max": float(delta.max()),
            "delta_min": float(delta.min()),
            "n_lyapunov_violations": n_violations,
            "lyapunov_ok": bool(n_violations == 0),
        })
        print(f"  D1 Lyapunov  T={T:.1f}: U0̄={u0.mean():+.4f} UT̄={uT.mean():+.4f} "
              f"Δ̄={delta.mean():+.4e} Δmax={delta.max():+.4e}  "
              f"violations={n_violations}/{n}")
    out["D1_lyapunov"] = energy_log

    # D2 — truncation-error study on a representative subset (first 8 patients
    # to keep the diagnostic <1 min wall-clock)
    sub = Z0[:8]
    err_log = []
    for T in Ts:
        err = truncation_error(flow, sub, T, n_steps_grid=(1, 2, 4, 8, 16, 32))
        err_log.append({"T": T, **err})
        print(f"  D2 trunc-err T={T:.1f}: euler_1={err['euler_1step_l2']:.4f} "
              f"chain_4={err['euler_chain_4step_l2']:.4f} "
              f"chain_32={err['euler_chain_32step_l2']:.4f}")
    out["D2_truncation_error"] = err_log

    # D3 — drift-direction agreement at η = 2 (v10's best swept point)
    eta = 2.0
    with torch.no_grad():
        delta_v10 = flow.euler_step(Z0, eta) - Z0
        delta_v11 = flow.forward_to_T(Z0, eta) - Z0
        cos = _cosine(delta_v10, delta_v11).cpu().numpy()
        ratio = (delta_v11.norm(dim=-1) / delta_v10.norm(dim=-1).clamp_min(1e-12)).cpu().numpy()
    out["D3_direction_agreement_eta_2"] = {
        "eta": float(eta),
        "cosine_v10_vs_v11_mean": float(cos.mean()),
        "cosine_v10_vs_v11_min":  float(cos.min()),
        "cosine_v10_vs_v11_p25":  float(np.percentile(cos, 25)),
        "step_norm_ratio_v11_v10_mean": float(ratio.mean()),
        "step_norm_ratio_v11_v10_p25":  float(np.percentile(ratio, 25)),
        "step_norm_ratio_v11_v10_p75":  float(np.percentile(ratio, 75)),
        "interpretation": (
            "cos≈1, ratio≈1 ⇒ v10 single-Euler agrees with the true ODE flow. "
            "cos<1 means v10's drift direction is WRONG; ratio≠1 means its "
            "step magnitude is wrong even when direction is right."
        ),
    }
    print(f"  D3 cos̄={cos.mean():.3f} cos_min={cos.min():.3f} "
          f"|Δv11|/|Δv10|̄={ratio.mean():.3f}")

    out_path = OUT_DIR / "ode_forward_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402

    with v11_stage("s1f_ode_forward_diagnostic"):
        main()
