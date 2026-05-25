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
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.longitudinal.clinical_features import encode_for_lens
from resistancemap.mortfm.survival import (
    CompetingRiskHead, SurvivalCalibration,
    concordance_index_bootstrap_ci, discrete_time_grid,
    integrated_brier_score, km_strata_separation,
)
from resistancemap.mortfm.survival.competing_risk_head import (
    nll_competing_risk,
)
from resistancemap.mortfm.trajectory import (
    GraphEnergyResistanceSDE, LatentToGraphProjector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_train_lens_resistance")


def _load_from_confirmed(confirmed_path: str, split_manifest_path: str = None) -> dict:
    """Load training data from the Spark lakehouse confirmed dataset.

    Joins with pre-computed z64 latents (PCA-encoded baseline expression from
    scripts/v10/s1c_encode_mmrf_z64.py) to provide real molecular features.
    """
    import json
    df = pd.read_parquet(confirmed_path)
    logger.info("Loaded confirmed dataset: %d rows, %d columns", len(df), len(df.columns))

    # Filter to rows with valid survival supervision
    if "allowed_loss_survival" in df.columns:
        df = df[df["allowed_loss_survival"] == True]
    elif "has_valid_survival_supervision" in df.columns:
        df = df[df["has_valid_survival_supervision"] == True]

    # Load split manifest for train/test assignment
    if split_manifest_path and Path(split_manifest_path).exists():
        manifest = json.loads(Path(split_manifest_path).read_text())
        logger.info("Split manifest: %s", {k: v for k, v in manifest.items() if k != "splits"})

    # Load pre-computed z64 latents (real PCA-encoded baseline expression)
    z64_path = Path("data/processed/mmrf_z64.npy")
    z64_ids_path = Path("data/processed/mmrf_z64_sample_ids.json")
    z64_map = {}
    if z64_path.exists() and z64_ids_path.exists():
        z64_arr = np.load(str(z64_path))
        z64_ids = json.loads(z64_ids_path.read_text())
        z64_map = {pid: z64_arr[i] for i, pid in enumerate(z64_ids)}
        logger.info("Loaded z64 latents for %d patients (real PCA features)", len(z64_map))
    else:
        logger.warning("No z64 latents found — using zero vectors (will underperform)")

    # Build rows, matching submitter_id to z64 latents
    rows = []
    n_matched = 0
    for _, r in df.iterrows():
        et = r.get("event_time_days")
        eo = r.get("event_observed")
        if pd.isna(et) or et is None:
            continue
        pid = r.get("submitter_id", "")
        # Try to match patient ID to z64 latent (strip suffix for matching)
        pid_short = pid.split("_")[0] + "_" + pid.split("_")[1] if "_" in pid else pid
        z0 = z64_map.get(pid, z64_map.get(pid_short, np.zeros(64)))
        if not np.all(z0 == 0):
            n_matched += 1
        rows.append({
            "patient_id": pid,
            "z0": z0,
            "z1": np.zeros(64),  # no follow-up for survival-only
            "tt2l_days": float(et),
            "event_observed": int(bool(eo)),
            "bort_1L": 0,
        })

    logger.info("Loaded %d patients from confirmed dataset; %d observed events; "
                "%d matched to z64 latents",
                len(rows), sum(r["event_observed"] for r in rows), n_matched)
    return {"rows": rows}


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


def _build_batch(rows: list[dict], d_latent: int, d_drug: int, d_graph: int, d_clin: int,
                  clinical_feats: Optional[np.ndarray] = None,
                  clinical_index: Optional[dict[str, int]] = None):
    z0 = torch.tensor(np.stack([r["z0"] for r in rows]), dtype=torch.float32)
    z1 = torch.tensor(np.stack([r["z1"] for r in rows]), dtype=torch.float32)
    drug = torch.stack([_build_drug_context(r, d_drug) for r in rows])
    # graph_emb is now computed by the model's LatentToGraphProjector;
    # we still pass a zero tensor placeholder for the data side.
    graph_emb = torch.zeros(len(rows), d_graph)
    if clinical_feats is not None and clinical_index is not None:
        idx = [clinical_index[r["patient_id"]] for r in rows]
        clinical = torch.tensor(clinical_feats[idx], dtype=torch.float32)
        # Right-pad / truncate to d_clin.
        if clinical.shape[1] < d_clin:
            pad = torch.zeros(clinical.shape[0], d_clin - clinical.shape[1])
            clinical = torch.cat([clinical, pad], dim=1)
        elif clinical.shape[1] > d_clin:
            clinical = clinical[:, :d_clin]
    else:
        clinical = torch.zeros(len(rows), d_clin)
    return z0, z1, drug, graph_emb, clinical


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--t-max-days", type=float, default=1600.0)
    ap.add_argument("--use-clinical", action="store_true",
                    help="Encode ISS+age+gender+bort_1L+n_treatments as clinical context.")
    ap.add_argument("--use-graph-projector", action="store_true",
                    help="Learn graph_emb from z0 via LatentToGraphProjector "
                         "instead of feeding a zero tensor.")
    ap.add_argument("--confirmed-dataset", type=str, default=None,
                    help="Path to confirmed Parquet dataset from Spark lakehouse. "
                         "If provided, loads training data from lakehouse instead of "
                         "legacy checkpoints.")
    ap.add_argument("--split-manifest", type=str, default=None,
                    help="Path to split_manifest.json from Spark lakehouse.")
    ap.add_argument("--graph-embedding-table", type=str, default=None,
                    help="Path to graph embedding Parquet (biological_edges).")
    ap.add_argument("--out", default="logs/mortfm/lens_resistance_summary.json")
    args = ap.parse_args()

    if args.confirmed_dataset:
        data = _load_from_confirmed(args.confirmed_dataset, args.split_manifest)
    else:
        data = _load_pairs()
    rows = data["rows"]
    if len(rows) < 10:
        logger.error("Too few patients (%d) — aborting", len(rows))
        return 1

    d_latent = 64
    d_drug, d_graph = 8, 16
    d_clin = 5 if args.use_clinical else 4
    t_grid = discrete_time_grid(args.t_max_days, args.n_bins)

    # Optional clinical features (deterministic, no imputation).
    clinical_feats = None
    clinical_index = None
    if args.use_clinical:
        enc = encode_for_lens(
            "data/processed/mmrf_outcomes_treatment.tsv",
            patient_ids=[r["patient_id"] for r in rows],
        )
        clinical_feats = enc.features
        clinical_index = {pid: i for i, pid in enumerate(enc.patient_ids)}
        logger.info("Encoded clinical features: shape=%s (mean ISS=%.2f, %% bort_1L=%.1f%%)",
                    clinical_feats.shape, clinical_feats[:, 0].mean(),
                    100 * clinical_feats[:, 3].mean())

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
        z0_tr, z1_tr, drug_tr, graph_tr, clin_tr = _build_batch(
            train, d_latent, d_drug, d_graph, d_clin,
            clinical_feats=clinical_feats, clinical_index=clinical_index,
        )
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
        projector = LatentToGraphProjector(
            d_latent=d_latent, d_graph=d_graph,
        ).to(device) if args.use_graph_projector else None

        params = list(sde.parameters()) + list(head.parameters())
        if projector is not None:
            params += list(projector.parameters())
        opt = torch.optim.Adam(params, lr=args.lr)
        for ep in range(args.epochs):
            opt.zero_grad()
            g_in = projector(z0_tr) if projector is not None else graph_tr
            out = sde(z0_tr, g_in, drug_tr, clin_tr, return_samples=False)
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
        if projector is not None:
            projector.eval()
        with torch.no_grad():
            z0_h = torch.tensor(held["z0"], dtype=torch.float32, device=device).unsqueeze(0)
            d_h = _build_drug_context(held, d_drug).to(device).unsqueeze(0)
            if clinical_feats is not None and clinical_index is not None:
                c_idx = clinical_index[held["patient_id"]]
                c_h = torch.tensor(clinical_feats[c_idx], dtype=torch.float32, device=device).unsqueeze(0)
                if c_h.shape[1] < d_clin:
                    c_h = torch.cat([c_h, torch.zeros(1, d_clin - c_h.shape[1], device=device)], dim=1)
                elif c_h.shape[1] > d_clin:
                    c_h = c_h[:, :d_clin]
            else:
                c_h = torch.zeros(1, d_clin, device=device)
            g_h = projector(z0_h) if projector is not None else torch.zeros(1, d_graph, device=device)
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
        "use_clinical": bool(args.use_clinical),
        "use_graph_projector": bool(args.use_graph_projector),
        "clinical_feature_names": (
            ["iss_stage_ordinal", "age_at_dx_zscore", "gender_is_male", "bort_1L", "n_treatments_norm"]
            if args.use_clinical else []
        ),
        "graph_emb_zero_tensor": not args.use_graph_projector,
        "graph_emb_note": (
            "graph_emb learned from z0 via LatentToGraphProjector"
            if args.use_graph_projector else
            "graph_emb was a zero tensor"
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
