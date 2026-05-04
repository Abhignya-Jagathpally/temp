"""S2d: Re-run F5 (median-r identifiability) on REAL paired (z0, z1) pairs.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 row F5:
    median ‖ẑ - z₁‖ / ‖z₀ - z₁‖ ≥ 0.9 → energy formulation rejected.

Where ẑ is the result of K Euler steps of ż = -∇U_θ starting from z₀, and
z₁ is the *real* second-timepoint latent for the same patient.

This is the spec's intended F5 — Sprint 1's substitute (random in-distribution
targets) was strictly harder and uninformative.

Important interpretation note: a *snapshot-only* DSM-trained U_θ has no
explicit temporal supervision. Under DSM, both z₀ and z₁ are samples from the
same population data distribution, so both lie near local minima of U_θ
(score ≈ 0 on the data manifold). The F5 test against real pairs therefore
measures whether the *stationary energy structure alone* incidentally encodes
the baseline→relapse direction. If F5 fails, the implication is that paired
temporal supervision is required (Sprint 3 trajectory likelihood). If F5
passes, that is evidence the cohort has a non-trivial gradient-aligned drift
even without temporal training — a stronger result than the spec assumed.

Outputs:
    paper/v8_artifacts/v10_sprint2/f5_paired.json
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
SPRINT2 = ROOT / "paper" / "v8_artifacts" / "v10_sprint2"
SPRINT2.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential,
    ScalarPotentialConfig,
)


def euler_flow(model: ScalarPotential, z0: torch.Tensor, n_steps: int, step_size: float) -> torch.Tensor:
    z = z0.clone()
    for _ in range(n_steps):
        z_in = z.detach().clone().requires_grad_(True)
        u = model.forward(z_in).sum()
        (g,) = torch.autograd.grad(u, z_in)
        z = z - step_size * g.detach()
    return z


def median_r(z_hat: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> dict:
    num = (z_hat - z1).norm(dim=-1)
    den = (z0 - z1).norm(dim=-1).clamp_min(1e-9)
    r = (num / den).cpu().numpy()
    return {
        "n": int(len(r)),
        "median_r": float(np.median(r)),
        "mean_r": float(np.mean(r)),
        "p25_r": float(np.percentile(r, 25)),
        "p75_r": float(np.percentile(r, 75)),
        "min_r": float(np.min(r)),
        "max_r": float(np.max(r)),
        "frac_below_1": float((r < 1.0).mean()),
        "frac_below_0.9": float((r < 0.9).mean()),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v10s2",
        help="Sprint tag — selects checkpoints/u_theta_{tag}.pt and writes "
             "paper/v8_artifacts/v10_sprint2/f5_paired_{tag}.json.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paired = np.load(PROC / "mmrf_paired_z64.npz", allow_pickle=True)
    Z0 = torch.from_numpy(paired["z0"]).to(device)
    Z1 = torch.from_numpy(paired["z1"]).to(device)
    has_2nd_line = paired["has_2nd_line"]
    print(f"[S2d] paired latents: z0={tuple(Z0.shape)}, z1={tuple(Z1.shape)}")

    ckpt_path = CHK / f"u_theta_{args.tag}.pt"
    if not ckpt_path.exists():
        sys.exit(f"missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ScalarPotentialConfig(**ckpt["config"])
    model = ScalarPotential(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[S2d] loaded U_θ checkpoint: {ckpt_path}")

    results = {}
    grid = [
        (50, 5e-3),
        (50, 5e-2),
        (200, 5e-3),
        (200, 5e-2),
    ]
    for n_steps, step_size in grid:
        Zh = euler_flow(model, Z0, n_steps, step_size)
        all_r = median_r(Zh, Z0, Z1)
        # Strict-paired subset
        strict_mask = torch.from_numpy(has_2nd_line.astype(bool))
        if strict_mask.sum() >= 5:
            sZh = Zh[strict_mask]; sZ0 = Z0[strict_mask]; sZ1 = Z1[strict_mask]
            strict_r = median_r(sZh, sZ0, sZ1)
        else:
            strict_r = None
        key = f"steps={n_steps}_step={step_size:.0e}"
        results[key] = {
            "all_paired": all_r,
            "strict_paired_with_2nd_line": strict_r,
            "n_steps": n_steps,
            "step_size": step_size,
        }
        if strict_r is not None:
            print(
                f"[S2d] {key} ALL median_r={all_r['median_r']:.3f} "
                f"frac<1={all_r['frac_below_1']:.2f}  "
                f"STRICT median_r={strict_r['median_r']:.3f}"
            )
        else:
            print(
                f"[S2d] {key} ALL median_r={all_r['median_r']:.3f} "
                f"frac<1={all_r['frac_below_1']:.2f}"
            )

    # Identifiability "passes" if ANY config beats the 0.9 threshold (better than the substitute)
    best_median = min(r["all_paired"]["median_r"] for r in results.values())
    refuted = best_median >= 0.9
    summary = {
        "tag": args.tag,
        "ckpt_path": str(ckpt_path),
        "configs": results,
        "best_median_r": float(best_median),
        "refutation_threshold": 0.9,
        "refuted": bool(refuted),
        "note": (
            "Snapshot-only DSM-trained U_θ has no temporal supervision; "
            "F5 against real pairs tests whether the stationary energy "
            "alone encodes baseline→relapse direction. Failure here is "
            "expected at this stage and motivates Sprint 3 trajectory "
            "likelihood + Bellman-Harris branching prior."
        ),
    }
    out_path = SPRINT2 / f"f5_paired_{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[S2d] wrote {out_path}")


if __name__ == "__main__":
    main()
