#!/usr/bin/env python3
"""
scripts/mortfm_train_lens_resistance.py
=======================================
Phase A.2 — train MORT-FM-LENS on the MMRF baseline -> followup paired
latents (n=29) with the TT2L resistance endpoint.

Pipeline
--------
1. Load ``mmrf_paired_z64.npz`` (29 × 64 baseline z0, 29 × 64 followup z1).
2. Join with TT2L from ``mmrf_outcomes_treatment.tsv`` (29/29 patients
   matched, 20 observed events, median TT2L = 449d).
3. For each LOO fold, train GraphEnergyResistanceSDE + CompetingRiskHead
   on n-1 patients; predict the held-out patient's discrete-time hazard
   on the time grid [0, 1L+1L_max_days].
4. Reduce hazards to a single risk score per patient (negative survival
   at the median grid time) and compute Harrell C-index over all LOO
   predictions.
5. Refit a single-parameter SurvivalCalibration on the LOO predictions
   to report calibrated NLL.

Honest constraints
------------------
* n=29 is small. LOO is unbiased w.r.t. patient identity but variance is
  high; we report bootstrap CIs on the LOO C-index from REAL observations.
* The graph_emb input is a zero-tensor in this run (Block-A graph
  embeddings are on the cell-line side and not yet wired through to
  MMRF patient latents). The SDE still trains because f_theta(z, 0, d, c)
  is a valid drift function. We log this honestly.
* The resistance_emergence claim is granted only if C-index lower CI > 0.5.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.survival import (
    CompetingRiskHead, SurvivalCalibration,
    concordance_index_bootstrap_ci, discrete_time_grid,
    integrated_brier_score, km_strata_separation,
)
from resistancemap.mortfm.survival.competing_risk_head import (
    nll_competing_risk,
)
from resistancemap.mortfm.trajectory import GraphEnergyResistanceSDE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_train_lens_resistance")


def _load_pairs() -> dict:
    z = np.load("data/processed/mmrf_paired_z64.npz", allow_pickle=True)
    pids = [str(p) for p in z["patient_id"]]
    outcomes = pd.read_csv("data/processed/mmrf_outcomes_treatment.tsv", sep="\t")
    out_map = outcomes.set_index("submitter_id").to_dict("index")
    rows = []
    for i, pid in enumerate(pids):
        rec = out_map.get(pid)
        if rec is None:
            continue
        tt2l = rec.get("tt2L_days")
        had_2l = rec.get("had_2L")
        if pd.isna(tt2l) or pd.isna(had_2l):
            continue
        rows.append({
            "i": i, "patient_id": pid,
            "z0": z["z0"][i], "z1": z["z1"][i],
            "tt2l_days": float(tt2l),
            "event_observed": int(had_2l),
            "bort_1L": int(rec.get("bort_1L", 0) or 0),
        })
    logger.info("Loaded %d MMRF paired patients with TT2L; %d observed events",
                len(rows), sum(r["event_observed"] for r in rows))
    return {"rows": rows}


def _build_drug_context(row: dict, d_drug: int) -> torch.Tensor:
    """One-hot-ish: index 0 = bortezomib-1L flag; remaining dims zero."""
    v = torch.zeros(d_drug)
    if row["bort_1L"]:
        v[0] = 1.0
    return v


def _build_batch(rows: list[dict], d_latent: int, d_drug: int, d_graph: int, d_clin: int):
    z0 = torch.tensor(np.stack([r["z0"] for r in rows]), dtype=torch.float32)
    z1 = torch.tensor(np.stack([r["z1"] for r in rows]), dtype=torch.float32)
    drug = torch.stack([_build_drug_context(r, d_drug) for r in rows])
    graph_emb = torch.zeros(len(rows), d_graph)
    clinical = torch.zeros(len(rows), d_clin)
    return z0, z1, drug, graph_emb, clinical


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--t-max-days", type=float, default=1600.0)
    ap.add_argument("--out", default="logs/mortfm/lens_resistance_summary.json")
    args = ap.parse_args()

    data = _load_pairs()
    rows = data["rows"]
    if len(rows) < 10:
        logger.error("Too few paired patients (%d) — aborting", len(rows))
        return 1

    d_latent = 64
    d_drug, d_graph, d_clin = 8, 16, 4
    t_grid = discrete_time_grid(args.t_max_days, args.n_bins)

    # Per-patient event-bin assignment.
    bin_edges = t_grid.numpy()
    for r in rows:
        bin_idx = int(np.searchsorted(bin_edges[1:], r["tt2l_days"], side="right"))
        r["event_bin"] = min(bin_idx, args.n_bins - 1)

    logger.info("Event-bin distribution: %s",
                pd.Series([r["event_bin"] for r in rows]).value_counts().to_dict())

    # ---- LOO CV ----------------------------------------------------------
    risk_scores: list[float] = []
    pred_survivals: list[np.ndarray] = []
    truth_times: list[float] = []
    truth_events: list[int] = []
    all_hazards: list[torch.Tensor] = []
    all_bins: list[int] = []
    all_obs: list[int] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    median_grid_idx = args.n_bins // 2

    for fold_i, held in enumerate(rows):
        train = [r for r in rows if r["patient_id"] != held["patient_id"]]
        z0_tr, z1_tr, drug_tr, graph_tr, clin_tr = _build_batch(train, d_latent, d_drug, d_graph, d_clin)
        z0_tr = z0_tr.to(device); drug_tr = drug_tr.to(device)
        graph_tr = graph_tr.to(device); clin_tr = clin_tr.to(device)
        event_bins_tr = torch.tensor([r["event_bin"] for r in train], dtype=torch.long, device=device)
        event_obs_tr = torch.tensor([r["event_observed"] for r in train], dtype=torch.float32, device=device)

        sde = GraphEnergyResistanceSDE(
            d_latent=d_latent, d_graph=d_graph, d_drug=d_drug, d_clinical=d_clin,
            n_basins=5, integration_time=1.0, n_time_grid=args.n_bins,
            n_mc_samples=0,  # deterministic training
        ).to(device)
        head = CompetingRiskHead(d_latent=d_latent, n_bins=args.n_bins,
                                  event_names=["progression"]).to(device)

        opt = torch.optim.Adam(list(sde.parameters()) + list(head.parameters()), lr=args.lr)
        for ep in range(args.epochs):
            opt.zero_grad()
            out = sde(z0_tr, graph_tr, drug_tr, clin_tr, return_samples=False)
            z_final = out["z_traj"][:, -1, :]
            h_out = head(z_final)
            loss = nll_competing_risk(
                h_out["hazard"], h_out["survival_curve"],
                event_bins_tr, event_obs_tr,
            )
            loss.backward()
            opt.step()
        # Predict on held-out
        sde.eval(); head.eval()
        with torch.no_grad():
            z0_h = torch.tensor(held["z0"], dtype=torch.float32, device=device).unsqueeze(0)
            d_h = _build_drug_context(held, d_drug).to(device).unsqueeze(0)
            g_h = torch.zeros(1, d_graph, device=device)
            c_h = torch.zeros(1, d_clin, device=device)
            out_h = sde(z0_h, g_h, d_h, c_h, return_samples=False)
            head_h = head(out_h["z_traj"][:, -1, :])
            S_curve = head_h["survival_curve"][0].cpu().numpy()      # (K,)
            h_curve = head_h["hazard"][0, :, 0].cpu().numpy()        # (K,) progression event
        pred_survivals.append(S_curve)
        risk_scores.append(1.0 - float(S_curve[median_grid_idx]))     # higher = riskier
        truth_times.append(held["tt2l_days"])
        truth_events.append(held["event_observed"])
        all_hazards.append(torch.tensor(head_h["hazard"].cpu().numpy()))
        all_bins.append(held["event_bin"])
        all_obs.append(held["event_observed"])
        if (fold_i + 1) % 5 == 0:
            logger.info("LOO fold %d/%d done", fold_i + 1, len(rows))

    # ---- Metrics ---------------------------------------------------------
    cidx, c_lo, c_hi = concordance_index_bootstrap_ci(truth_times, truth_events, risk_scores, n_boot=1000)
    surv_pred = np.stack(pred_survivals, axis=0)            # (n, K)
    # t_grid has K+1 boundary points; the predicted survival has K bins —
    # use the right edges as the evaluation timepoints.
    eval_times = t_grid.numpy()[1:]
    ibs = integrated_brier_score(surv_pred, eval_times, truth_times, truth_events)
    km_report = km_strata_separation(truth_times, truth_events, risk_scores)

    # Calibration on LOO hazards
    hazard_all = torch.cat(all_hazards, dim=0)              # (n, K, R)
    bins_all = torch.tensor(all_bins, dtype=torch.long)
    obs_all = torch.tensor(all_obs, dtype=torch.float32)
    pre_S = torch.cumprod(torch.prod(1 - hazard_all, dim=-1), dim=1)
    pre_nll = float(nll_competing_risk(hazard_all, pre_S, bins_all, obs_all).item())
    calib = SurvivalCalibration()
    calib.fit(hazard_all, bins_all, obs_all)
    post_hazard = calib(hazard_all)
    post_S = torch.cumprod(torch.prod(1 - post_hazard, dim=-1), dim=1)
    post_nll = float(nll_competing_risk(post_hazard, post_S, bins_all, obs_all).item())

    # ---- Gate verdict ----------------------------------------------------
    RESISTANCE_EMERGENCE_PASS = (c_lo > 0.55)  # require lower CI > 0.55, not just 0.5
    SURVIVAL_PREDICTION_PASS = (c_lo > 0.50)

    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-lens-resistance-mmrf",
        "cohort": "MMRF paired (baseline -> followup)",
        "n_patients": int(len(rows)),
        "n_observed_events": int(sum(truth_events)),
        "endpoint": "time_to_next_treatment (TT2L)",
        "t_grid_days": [float(x) for x in t_grid.tolist()],
        "loo_cindex": cidx,
        "loo_cindex_95ci": [c_lo, c_hi],
        "integrated_brier_score": ibs,
        "nll_uncalibrated": pre_nll,
        "nll_temperature_scaled": post_nll,
        "calibration_log_temperature": float(calib.log_temperature.detach().item()),
        "km_strata_separation": km_report,
        "graph_emb_zero_tensor": True,
        "graph_emb_note": (
            "graph_emb was a zero tensor — Block A's biological graph embedding "
            "is not yet wired through to MMRF patient latents. f_theta(z, 0, d, c) "
            "is still a valid drift function; integrating the graph remains a "
            "scope-extension after this run."
        ),
        "resistance_emergence_gate_threshold_cindex_lower_ci": 0.55,
        "resistance_emergence_gate_pass": bool(RESISTANCE_EMERGENCE_PASS),
        "survival_prediction_gate_threshold_cindex_lower_ci": 0.50,
        "survival_prediction_gate_pass": bool(SURVIVAL_PREDICTION_PASS),
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("LOO C-index = %.3f [95%% CI %.3f, %.3f]; IBS=%.3f; pre_nll=%.3f post_nll=%.3f",
                cidx, c_lo, c_hi, ibs, pre_nll, post_nll)
    logger.info("Resistance-emergence gate pass: %s; survival-prediction gate pass: %s",
                RESISTANCE_EMERGENCE_PASS, SURVIVAL_PREDICTION_PASS)
    return 0 if SURVIVAL_PREDICTION_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
