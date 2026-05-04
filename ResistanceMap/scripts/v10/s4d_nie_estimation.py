"""S4d: Lange-Hansen-style hazard-scale NIE for proteasome → TT2L on Bort cohort.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §9 row 4: single-mediator NIE
(proteasome → time-to-2nd-line Bortezomib) with EIMOC disclaimer at Pearl tier
L2 (interventional, under documented assumptions).

Causal model (Pearl SCM):

      C ──┬──→ A ──┬──→ Y
          │   │    │
          ↓   ↓    ↓
          M ──┴───→
              (mediator)

Where:
    A = bort_1L              (binary treatment: Bort 1L vs not)
    M = M_proteasome         (continuous mediator: proteasome propagation score)
    Y = (tt2L_days, had_2L)  (right-censored survival outcome)
    C = {iss_stage, age, gender, cyto_*}

Mediation decomposition on the log-hazard scale (Lange & Hansen 2011):

    log h(t | A=a, M=m, C) = β_0(t) + β_A·a + β_M·m + β_AM·a·m + β_C^T·c

    Total effect (TE)        ≈ β_A + β_AM·E[M|A=1]
    Natural Direct (NDE)     ≈ β_A + β_AM·E[M|A=0]
    Natural Indirect (NIE)   ≈ β_AM·(E[M|A=1] − E[M|A=0]) + β_M·(E[M|A=1] − E[M|A=0])
                              = (β_M + β_AM)·ΔM       (interaction-aware)

Identification assumptions (CRITICAL — listed in EIMOC disclaimer):
    A1. No unmeasured confounding A→Y given C
    A2. No unmeasured confounding A→M given C
    A3. No unmeasured confounding M→Y given (A, C)
    A4. No exposure-induced mediator-outcome confounder
    A5. Cox proportional hazards holds for the modeled covariates

Bootstrap-resampled 95% CI for NDE/NIE/TE (B=1000 patient-level resamples).

Sensitivity:
    E-value (VanderWeele & Ding 2017) for the unmeasured confounder strength
    that would nullify the observed (point estimate, lower CI bound).

Output: paper/v8_artifacts/v10_sprint4/nie_proteasome_tt2L.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT4 = ROOT / "paper" / "v8_artifacts" / "v10_sprint4"
SPRINT4.mkdir(parents=True, exist_ok=True)

CONFOUNDERS = [
    "iss_stage_ord", "age_at_dx_years", "gender_male",
    "cyto_del17p", "cyto_chr1q21_gain", "cyto_del13q",
    "cyto_t_4_14", "cyto_t_11_14",
]


def fit_cox_with_interaction(df: pd.DataFrame, mediator_col: str = "M_proteasome") -> dict:
    """Fit Cox PH with A, M, A*M, and confounders. Return coef dict."""
    work = df.copy()
    work["A_x_M"] = work["bort_1L"] * work[mediator_col]
    cols = ["bort_1L", mediator_col, "A_x_M"] + CONFOUNDERS
    cph = CoxPHFitter(penalizer=0.001)  # tiny ridge for numerical stability
    cph.fit(
        work[cols + ["tt2L_days", "had_2L"]],
        duration_col="tt2L_days", event_col="had_2L",
        show_progress=False, fit_options={"step_size": 0.5},
    )
    return {
        "beta_A": float(cph.params_["bort_1L"]),
        "beta_M": float(cph.params_[mediator_col]),
        "beta_AM": float(cph.params_["A_x_M"]),
        "concordance": float(cph.concordance_index_),
        "ll": float(cph.log_likelihood_),
        "params": cph.params_.to_dict(),
        "p_values": cph.summary["p"].to_dict(),
    }


def decompose_effects(coefs: dict, df: pd.DataFrame, mediator_col: str) -> dict:
    """Decompose total effect on the log-hazard scale into NDE + NIE.

    Convention: log-HR for A=1 vs A=0 evaluated at population mean M values.
    """
    M_under_A1 = float(df.loc[df["bort_1L"] == 1, mediator_col].mean())
    M_under_A0 = float(df.loc[df["bort_1L"] == 0, mediator_col].mean())
    delta_M = M_under_A1 - M_under_A0

    beta_A = coefs["beta_A"]
    beta_M = coefs["beta_M"]
    beta_AM = coefs["beta_AM"]

    # Natural Direct Effect: A=1 vs A=0 holding M at M(A=0)
    NDE_log = beta_A + beta_AM * M_under_A0
    # Natural Indirect Effect: change in Y from setting M(A=1) instead of M(A=0)
    NIE_log = (beta_M + beta_AM) * delta_M
    TE_log = NDE_log + NIE_log
    return {
        "NDE_log_hr": NDE_log,
        "NDE_hr": float(np.exp(NDE_log)),
        "NIE_log_hr": NIE_log,
        "NIE_hr": float(np.exp(NIE_log)),
        "TE_log_hr": TE_log,
        "TE_hr": float(np.exp(TE_log)),
        "delta_M_A1_minus_A0": delta_M,
        "M_mean_A1": M_under_A1,
        "M_mean_A0": M_under_A0,
        "proportion_mediated": float(NIE_log / TE_log) if abs(TE_log) > 1e-9 else None,
    }


def bootstrap_decomposition(
    df: pd.DataFrame, B: int, rng_seed: int, mediator_col: str = "M_proteasome",
) -> dict:
    g = torch.Generator().manual_seed(rng_seed)
    n = len(df)
    NDE = np.empty(B); NIE = np.empty(B); TE = np.empty(B); BAM = np.empty(B); BM = np.empty(B)
    n_ok = 0
    for b in range(B):
        idx = torch.randint(0, n, (n,), generator=g).numpy()
        sub = df.iloc[idx].reset_index(drop=True)
        try:
            coefs = fit_cox_with_interaction(sub, mediator_col=mediator_col)
            d = decompose_effects(coefs, sub, mediator_col)
            NDE[n_ok] = d["NDE_log_hr"]
            NIE[n_ok] = d["NIE_log_hr"]
            TE[n_ok]  = d["TE_log_hr"]
            BAM[n_ok] = coefs["beta_AM"]
            BM[n_ok]  = coefs["beta_M"]
            n_ok += 1
        except Exception as e:
            continue  # Cox didn't converge on this resample; skip
        if (b + 1) % 100 == 0:
            print(f"  boot {b+1}/{B}  n_ok={n_ok}")
    NDE = NDE[:n_ok]; NIE = NIE[:n_ok]; TE = TE[:n_ok]
    BAM = BAM[:n_ok]; BM = BM[:n_ok]

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    return {
        "n_ok": int(n_ok),
        "n_attempts": int(B),
        "NDE_log_hr": {"mean": float(NDE.mean()), "ci_95": ci(NDE)},
        "NIE_log_hr": {"mean": float(NIE.mean()), "ci_95": ci(NIE)},
        "TE_log_hr":  {"mean": float(TE.mean()),  "ci_95": ci(TE)},
        "NDE_hr":     {"mean": float(np.exp(NDE.mean())),
                       "ci_95": (float(np.exp(np.percentile(NDE, 2.5))),
                                  float(np.exp(np.percentile(NDE, 97.5))))},
        "NIE_hr":     {"mean": float(np.exp(NIE.mean())),
                       "ci_95": (float(np.exp(np.percentile(NIE, 2.5))),
                                  float(np.exp(np.percentile(NIE, 97.5))))},
        "TE_hr":      {"mean": float(np.exp(TE.mean())),
                       "ci_95": (float(np.exp(np.percentile(TE, 2.5))),
                                  float(np.exp(np.percentile(TE, 97.5))))},
        "beta_AM": {"mean": float(BAM.mean()), "ci_95": ci(BAM)},
        "beta_M":  {"mean": float(BM.mean()),  "ci_95": ci(BM)},
    }


def evalue(rr: float) -> float:
    """VanderWeele-Ding E-value for an HR / RR (point estimate or CI bound).

    For RR > 1: E = RR + sqrt(RR*(RR-1))
    For RR < 1: E = (1/RR) + sqrt((1/RR)*((1/RR)-1))
    """
    rr = float(rr)
    if rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def main() -> None:
    print("=== S4d: Cox NIE estimation ===")
    df = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df = df.dropna(subset=["M_proteasome", "tt2L_days", "had_2L"]).reset_index(drop=True)
    print(f"analysis N: {len(df)}")

    # Point estimate
    print("\n[Cox] fitting full Cox PH model on point estimate...")
    coefs = fit_cox_with_interaction(df, mediator_col="M_proteasome")
    print(f"  β_A    = {coefs['beta_A']:+.4f}  (p={coefs['p_values']['bort_1L']:.4g})")
    print(f"  β_M    = {coefs['beta_M']:+.4f}  (p={coefs['p_values']['M_proteasome']:.4g})")
    print(f"  β_A·M  = {coefs['beta_AM']:+.4f}  (p={coefs['p_values']['A_x_M']:.4g})")
    print(f"  concordance = {coefs['concordance']:.3f}  ll = {coefs['ll']:.1f}")

    decomp = decompose_effects(coefs, df, "M_proteasome")
    print(f"\n[Decomposition]")
    print(f"  NDE  HR = {decomp['NDE_hr']:.3f}  (log = {decomp['NDE_log_hr']:+.4f})")
    print(f"  NIE  HR = {decomp['NIE_hr']:.3f}  (log = {decomp['NIE_log_hr']:+.4f})")
    print(f"  TE   HR = {decomp['TE_hr']:.3f}  (log = {decomp['TE_log_hr']:+.4f})")
    if decomp["proportion_mediated"] is not None:
        print(f"  proportion mediated by M_proteasome: {decomp['proportion_mediated']:.2%}")

    # Bootstrap CI
    print("\n[Bootstrap] B=1000 patient-level resamples...")
    t0 = time.time()
    boot = bootstrap_decomposition(df, B=1000, rng_seed=20260503)
    print(f"\n[Bootstrap] done in {time.time()-t0:.1f}s; ok={boot['n_ok']}/1000")
    print(f"  NDE log-HR: {boot['NDE_log_hr']['mean']:+.4f}  95%CI {boot['NDE_log_hr']['ci_95']}")
    print(f"  NIE log-HR: {boot['NIE_log_hr']['mean']:+.4f}  95%CI {boot['NIE_log_hr']['ci_95']}")
    print(f"  TE  log-HR: {boot['TE_log_hr']['mean']:+.4f}  95%CI {boot['TE_log_hr']['ci_95']}")
    print(f"  NIE  HR: {boot['NIE_hr']['mean']:.3f}    95%CI [{boot['NIE_hr']['ci_95'][0]:.3f}, "
          f"{boot['NIE_hr']['ci_95'][1]:.3f}]")

    # E-value sensitivity
    nie_lower = boot["NIE_hr"]["ci_95"][0]
    nde_lower = boot["NDE_hr"]["ci_95"][0]
    e_nie = evalue(boot["NIE_hr"]["mean"])
    e_nie_ci = evalue(nie_lower if nie_lower < 1 else boot["NIE_hr"]["ci_95"][1])
    e_nde = evalue(boot["NDE_hr"]["mean"])
    print(f"\n[E-value sensitivity (VanderWeele-Ding)]")
    print(f"  NIE point E-value: {e_nie:.2f}  (smaller = more vulnerable to unmeasured confounding)")
    print(f"  NIE CI-bound E-value: {e_nie_ci:.2f}")
    print(f"  NDE point E-value: {e_nde:.2f}")

    # Falsification gates (pre-registered)
    nie_excludes_zero = (
        boot["NIE_log_hr"]["ci_95"][0] > 0 or boot["NIE_log_hr"]["ci_95"][1] < 0
    )
    f8_pass = bool(nie_excludes_zero)
    f10_pass = bool(e_nie_ci >= 1.5)
    print(f"\n[Falsification gates]")
    print(f"  F8 (NIE 95% CI excludes zero): {'PASS' if f8_pass else 'FAIL'}  "
          f"(CI = {boot['NIE_log_hr']['ci_95']})")
    print(f"  F10 (E-value ≥ 1.5): {'PASS' if f10_pass else 'FAIL'}  "
          f"(E-value at CI bound = {e_nie_ci:.2f})")

    out = {
        "design": {
            "treatment_A": "bort_1L  (Bortezomib 1st-line vs not)",
            "mediator_M": "M_proteasome  (z-weighted MMRF baseline expression of Sprint-3 Bortezomib RWR top-50)",
            "outcome_Y": "(tt2L_days, had_2L)  right-censored time-to-2nd-line therapy",
            "confounders_C": CONFOUNDERS,
            "model": "Cox PH with A + M + A:M + C",
            "scale": "log-hazard (Lange & Hansen 2011 mediation)",
            "Pearl_tier": "L2 (interventional under documented assumptions)",
            "EIMOC_assumptions": [
                "A1. No unmeasured confounding A→Y given C",
                "A2. No unmeasured confounding A→M given C",
                "A3. No unmeasured confounding M→Y given (A, C)",
                "A4. No exposure-induced mediator-outcome confounder",
                "A5. Cox proportional hazards holds for modeled covariates",
            ],
        },
        "cohort": {
            "N": int(len(df)),
            "n_bort_1L": int((df["bort_1L"] == 1).sum()),
            "n_non_bort_1L": int((df["bort_1L"] == 0).sum()),
            "n_observed_2L": int((df["had_2L"] == 1).sum()),
            "n_censored": int((df["had_2L"] == 0).sum()),
        },
        "point_estimate": {
            "coefs": coefs,
            "decomposition": decomp,
        },
        "bootstrap_B1000": boot,
        "evalue_sensitivity": {
            "NIE_hr_point": e_nie,
            "NIE_hr_ci_bound": e_nie_ci,
            "NDE_hr_point": e_nde,
            "interpretation": "Minimum strength of unmeasured confounding (on RR scale) that would explain away the observed effect if no other confounding is present.",
        },
        "falsification": {
            "F8_NIE_CI_excludes_zero": f8_pass,
            "F10_E_value_ci_bound_ge_1p5": f10_pass,
        },
    }
    out_path = SPRINT4 / "nie_proteasome_tt2L.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
