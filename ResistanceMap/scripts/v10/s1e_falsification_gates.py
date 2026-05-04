"""S1e: Run F4 (CurlFraction) and F5 (median-r identifiability) falsification gates.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6:

F4 — CurlFraction = E‖v_θ‖² / E‖F_θ‖² via natural Helmholtz–Hodge decomposition
on a k-NN simplicial complex of the baselines (here: actual 787 baselines).
The gradient field of U_θ alone defines F_θ = -∇U_θ. CurlFraction tests how
much of the empirical "data flow" (estimated via local-velocity field from
k-NN density gradients) is *not* captured by -∇U_θ.

F5 — Median-r identifiability test (per WADDINGTON_NEURAL_ODE_REVIEW.md §2.3):
hold out a fraction of patients, seed the gradient flow ż = -∇U_θ from each
held-out z₀ to a target z*, and report median ‖ẑ - z*‖ / ‖z₀ - z*‖. If ≥0.9
the flow is making essentially no progress and the energy formulation is
rejected.

Outputs:
    paper/v8_artifacts/v10_sprint1/falsification_gates.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CHK = ROOT / "checkpoints"

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential,
    ScalarPotentialConfig,
)


def f4_curl_fraction(
    model: ScalarPotential,
    Z: torch.Tensor,
    k: int = 15,
) -> dict:
    """Discrete Helmholtz–Hodge curl-fraction of the data drift's residual after
    removing the model's gradient field.

    Spec interpretation (V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 row F4):
        F_θ = -∇U_θ is the model's gradient drift. The empirical drift estimated
        from the k-NN graph (mean-shift vector at each node) defines f_emp on
        each edge. We compare:

            f_emp_ij  = ⟨0.5(drift_i + drift_j), z_j - z_i⟩       (data flow)
            f_model_ij= ⟨-0.5(∇U_i + ∇U_j),     z_j - z_i⟩        (model flow)
            residual  = f_emp - f_model
            curl(residual) = residual - lstsq(B u, residual).gradient_part

        CurlFraction = ‖curl(residual)‖² / ‖f_emp‖².

    Threshold: > 0.30 → gradient-only U_θ cannot capture the empirical flow,
    a divergence-free flux head v_θ is needed (Sprint 2 fallback per spec).
    """
    device = Z.device
    Zn = Z.detach().cpu().numpy()
    n = Zn.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Zn)
    _, knn_idx = nbrs.kneighbors(Zn)
    edges_list: list[tuple[int, int]] = []
    for i in range(n):
        for j in knn_idx[i, 1:]:
            if i < j:
                edges_list.append((i, int(j)))
    edges = np.array(edges_list, dtype=np.int64)
    n_e = len(edges)

    # Empirical drift: mean-shift vector at each node (a standard density-
    # gradient proxy on a k-NN graph)
    knn_mean = Zn[knn_idx[:, 1:]].mean(axis=1)
    drift_emp = knn_mean - Zn  # (n, d)

    # Model drift: F_θ = -∇U_θ, evaluated at every node in batches
    grads = []
    bs = 256
    for s in range(0, n, bs):
        zb = Z[s:s+bs].detach().clone().requires_grad_(True)
        u = model.forward(zb).sum()
        (g,) = torch.autograd.grad(u, zb)
        grads.append(g.detach().cpu().numpy())
    grad_U = np.concatenate(grads, axis=0)
    drift_model = -grad_U  # (n, d)

    z_diff = Zn[edges[:, 1]] - Zn[edges[:, 0]]
    edge_emp = (0.5 * (drift_emp[edges[:, 0]] + drift_emp[edges[:, 1]]) * z_diff).sum(axis=1)
    edge_model = (0.5 * (drift_model[edges[:, 0]] + drift_model[edges[:, 1]]) * z_diff).sum(axis=1)
    residual_edge = edge_emp - edge_model

    # Hodge decomposition of the residual: solve B u = residual_edge in
    # least-squares; the lstsq residual is the curl component
    rows = np.concatenate([np.arange(n_e), np.arange(n_e)])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([np.ones(n_e), -np.ones(n_e)])
    B = sp.coo_matrix((data, (rows, cols)), shape=(n_e, n)).tocsr()
    BtB = (B.T @ B).tocsc() + sp.identity(n) * 1e-6

    # Decompose BOTH the empirical edge-flow and the residual-after-model
    Btf_emp = B.T @ edge_emp
    u_emp = spla.spsolve(BtB, Btf_emp)
    curl_emp = edge_emp - (B @ u_emp)
    cf_emp = float((curl_emp ** 2).sum() / max((edge_emp ** 2).sum(), 1e-12))

    Btf_res = B.T @ residual_edge
    u_res = spla.spsolve(BtB, Btf_res)
    curl_res = residual_edge - (B @ u_res)
    cf_res = float((curl_res ** 2).sum() / max((edge_emp ** 2).sum(), 1e-12))

    return {
        "curl_fraction_residual": cf_res,
        "curl_fraction_empirical": cf_emp,
        "curl_fraction": cf_res,
        "n_edges": int(n_e),
        "k_nn": int(k),
        "edge_emp_l2_sq": float((edge_emp ** 2).sum()),
        "edge_model_l2_sq": float((edge_model ** 2).sum()),
        "residual_l2_sq": float((residual_edge ** 2).sum()),
        "method": "discrete_hodge_kNN_lstsq_residual_of_emp_minus_model_drift",
    }


def f5_median_r(
    model: ScalarPotential,
    Z: torch.Tensor,
    held_out_frac: float = 0.20,
    n_steps: int = 50,
    step_size: float = 5e-3,
    seed: int = 0,
) -> dict:
    """Median-r identifiability: simulate -∇U flow from random initial points
    toward random target points; report median ‖ẑ-z*‖ / ‖z₀-z*‖. ≥0.9 ⇒ refute.
    """
    g_cpu = torch.Generator().manual_seed(seed)
    n = Z.shape[0]
    n_test = max(int(round(held_out_frac * n)), 8)
    perm = torch.randperm(n, generator=g_cpu)
    test_idx = perm[:n_test]
    other_idx = perm[n_test:]
    target_perm = torch.randperm(other_idx.numel(), generator=g_cpu)[:n_test]
    target_idx = other_idx[target_perm]

    z0 = Z[test_idx].clone()
    z_star = Z[target_idx].clone()
    z = z0.clone()
    for _ in range(n_steps):
        z_in = z.detach().clone().requires_grad_(True)
        u = model.forward(z_in).sum()
        (g,) = torch.autograd.grad(u, z_in)
        z = z - step_size * g.detach()
    num = (z - z_star).norm(dim=-1)
    den = (z0 - z_star).norm(dim=-1).clamp_min(1e-9)
    r = (num / den).cpu().numpy()
    return {
        "n_test": int(n_test),
        "median_r": float(np.median(r)),
        "mean_r": float(np.mean(r)),
        "p25_r": float(np.percentile(r, 25)),
        "p75_r": float(np.percentile(r, 75)),
        "n_steps": int(n_steps),
        "step_size": float(step_size),
        "refutation_threshold": 0.9,
        "refuted": bool(float(np.median(r)) >= 0.9),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v10s1",
        help="Sprint tag — selects checkpoints/u_theta_{tag}.pt and writes "
             "paper/v8_artifacts/{sprint_dir}/falsification_gates_{tag}.json.")
    parser.add_argument("--sprint-dir", type=str, default="v10_sprint1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z = torch.from_numpy(np.load(PROC / "mmrf_z64.npy")).to(device)
    print(f"[S1e] latents: {tuple(Z.shape)} on {device}")

    ckpt_path = CHK / f"u_theta_{args.tag}.pt"
    if not ckpt_path.exists():
        sys.exit(f"missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ScalarPotentialConfig(**ckpt["config"])
    model = ScalarPotential(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[S1e] loaded U_θ checkpoint: {ckpt_path}")

    f4 = f4_curl_fraction(model, Z, k=15)
    print(f"[S1e] F4 CurlFraction = {f4['curl_fraction']:.4f} "
          f"(threshold 0.30 → flux head needed if exceeded)")

    f5 = f5_median_r(model, Z, held_out_frac=0.20, n_steps=50, step_size=5e-3)
    print(f"[S1e] F5 median_r = {f5['median_r']:.4f} (refute ≥0.9: {f5['refuted']})")

    out = {
        "tag": args.tag,
        "ckpt_path": str(ckpt_path),
        "F4_curl_fraction": f4,
        "F5_median_r": f5,
        "verdict": {
            "F4_passes": bool(f4["curl_fraction"] <= 0.30),
            "F5_passes": bool(not f5["refuted"]),
        },
    }
    out_dir = ROOT / "paper" / "v8_artifacts" / args.sprint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"falsification_gates_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[S1e] wrote {out_path}")


if __name__ == "__main__":
    main()
