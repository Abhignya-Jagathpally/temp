"""Apply mmSYGNAL subtype-aware routing and compute head-to-head C-index vs v11 Cox.

Routing rules (from baliga-lab tutorial / paper):
  - For each patient, compute risk score from each subtype model the patient is positive for.
    Cyto subtypes (B-grade): amp(1q), del(13), del(1p)
    Cyto subtype (A-grade): t(4;14)
    RNAseq subtype (A-grade): FGFR3 (we do not have FGFR3 calls -> skip)
    Agnostic (C-grade): always available, fallback when no subtype.
  - Per tutorial step 5: keep highest-grade among applicable; if multiple at same grade, take mean.
  - Final risk score is per-patient probability of "high" risk.

Evaluate per-stratum + marginal Harrell C-index using lifelines.concordance_index.
Strata = same 5-stratum partition used in v11 (S1..S5 in cox_discrimination.json).
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from lifelines.utils import concordance_index

ROOT = Path("/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap")
SP5 = ROOT / "paper/v8_artifacts/v11_sprint5"

# Load mmSYGNAL per-model probabilities
risk = pd.read_csv("/tmp/our_mmsygnal_per_model.csv")
risk = risk.set_index("patient_id")
print("mmSYGNAL per-model:", risk.shape)

# Load outcome + cyto strata
clin = pd.read_csv(ROOT / "data/processed/mmrf_sprint4_analysis.tsv", sep="\t")
clin = clin.set_index("submitter_id")
print("clinical:", clin.shape)

assert risk.index.equals(clin.index) or set(risk.index) == set(clin.index), "id mismatch"
risk = risk.reindex(clin.index)

# Subtype routing
# Available cyto flags: cyto_del17p, cyto_chr1q21_gain, cyto_del13q, cyto_t_4_14, cyto_t_11_14
# Map to mmSYGNAL models:
#   amp1q  <- cyto_chr1q21_gain   (B grade)
#   del13  <- cyto_del13q          (B grade)
#   del1p  <- (NOT in our cyto cols, this MMRF panel doesn't include 1p deletion) -- skip
#   t4_14  <- cyto_t_4_14          (A grade)
#   FGFR3  <- (no RNAseq subtype call available) -- skip
#   agnostic always -> C grade fallback
# Note: del17p / t_11_14 are NOT mmSYGNAL subtypes; they don't route to a model.

def route_risk(row, cyto):
    A_scores = []
    B_scores = []
    if cyto["cyto_t_4_14"] == 1 and not pd.isna(row.get("t4_14")):
        A_scores.append(row["t4_14"])
    if cyto["cyto_chr1q21_gain"] == 1 and not pd.isna(row.get("amp1q")):
        B_scores.append(row["amp1q"])
    if cyto["cyto_del13q"] == 1 and not pd.isna(row.get("del13")):
        B_scores.append(row["del13"])
    if A_scores:
        return float(np.mean(A_scores)), "A"
    if B_scores:
        return float(np.mean(B_scores)), "B"
    return float(row["agnostic"]), "C"

routed = []
for pid, r in risk.iterrows():
    s, g = route_risk(r, clin.loc[pid])
    routed.append((pid, s, g))
routed_df = pd.DataFrame(routed, columns=["patient_id", "mmsygnal_risk", "grade"]).set_index("patient_id")
print("grade counts:", routed_df["grade"].value_counts().to_dict())

# Build evaluation dataframe
ev = clin.join(routed_df)
ev = ev.dropna(subset=["mmsygnal_risk", "tt2L_days", "had_2L"])
print("evaluable:", len(ev))

# Strata exactly as v11 cox uses (priority order matches cox_discrimination.json):
# S1_del17p, S2_t_4_14, S3_1q21, S4_t_11_14, S5_other (mutually exclusive by priority)
def stratum(row):
    if row["cyto_del17p"] == 1: return "S1_del17p"
    if row["cyto_t_4_14"] == 1: return "S2_t_4_14"
    if row["cyto_chr1q21_gain"] == 1: return "S3_1q21"
    if row["cyto_t_11_14"] == 1: return "S4_t_11_14"
    return "S5_other"

ev["stratum"] = ev.apply(stratum, axis=1)
print("stratum counts:", ev["stratum"].value_counts().to_dict())

# Compute Harrell C-index. Higher mmsygnal_risk => higher hazard => shorter time.
# concordance_index(event_times, predicted_scores, event_observed). Higher score => longer time predicted; we invert.
def cidx(df):
    return concordance_index(df["tt2L_days"].values, -df["mmsygnal_risk"].values, df["had_2L"].values)

per_stratum = {}
for s, sub in ev.groupby("stratum"):
    n_evt = int(sub["had_2L"].sum())
    per_stratum[s] = {"n": int(len(sub)), "n_events": n_evt, "c_index": float(cidx(sub))}
marginal = {
    "n": int(len(ev)),
    "n_events": int(ev["had_2L"].sum()),
    "c_index": float(cidx(ev)),
}

# Also compute for the agnostic-only model (no routing) to demonstrate effect of subtype routing
ev["agnostic_only"] = risk.reindex(ev.index)["agnostic"]
def cidx_agnostic(df):
    return concordance_index(df["tt2L_days"].values, -df["agnostic_only"].values, df["had_2L"].values)
agnostic_marginal = float(cidx_agnostic(ev))
agnostic_per_stratum = {}
for s, sub in ev.groupby("stratum"):
    agnostic_per_stratum[s] = {"n": int(len(sub)), "c_index": float(cidx_agnostic(sub))}

# Output
out = {
    "design": "mmSYGNAL risk models (Wall et al 2021 via Murie et al 2025) applied to MMRF 787-patient subset using published IA12 program activity (baliga-lab/mmSYGNAL-risk-prediction-models, GPL-3.0, file data/program_activity_IA12_py.csv). Subtype routing per tutorial: A>B>C grade, mean within grade. Available subtypes: t(4;14)[A], amp(1q)[B], del(13)[B], agnostic[C]. del(1p) and FGFR3 models NOT applied (no calls in our cyto panel). Outcome: tt2L_days / had_2L (TT2L). Same 5-stratum partition as v11 Cox.",
    "n_total": int(len(ev)),
    "n_events": int(ev["had_2L"].sum()),
    "subtype_routing_grade_counts": routed_df["grade"].value_counts().to_dict(),
    "models_applied": ["agnostic", "amp1q", "del13", "t4_14"],
    "models_skipped": {
        "del1p": "no del(1p) call in MMRF cyto panel (cyto_chr1p del column absent)",
        "FGFR3": "no FGFR3 RNAseq subtype call available",
    },
    "mmSYGNAL_routed": {
        "per_stratum": per_stratum,
        "marginal": {"c_index": marginal["c_index"]},
    },
    "mmSYGNAL_agnostic_only": {
        "per_stratum": agnostic_per_stratum,
        "marginal": {"c_index": agnostic_marginal},
    },
    "v11_cox_marginal_for_comparison": {
        "Cox_v10": 0.6526234680383602,
        "Cox_v11features": 0.6511865218613659,
        "Cox_v11_richer": 0.6537584472651166,
    },
}

# Save
out_path = SP5 / "mmsygnal" / "head_to_head_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", out_path)

print("\n=== mmSYGNAL routed ===")
print(f"Marginal C-index: {marginal['c_index']:.4f}  (n={marginal['n']}, events={marginal['n_events']})")
for s, v in per_stratum.items():
    print(f"  {s}: n={v['n']:3d} ev={v['n_events']:3d}  C-index={v['c_index']:.4f}")
print("\n=== mmSYGNAL agnostic-only (no routing) ===")
print(f"Marginal C-index: {agnostic_marginal:.4f}")
for s, v in agnostic_per_stratum.items():
    print(f"  {s}: n={v['n']:3d}  C-index={v['c_index']:.4f}")
print("\n=== v11 Cox (reference) ===")
print(f"Cox_v10:        marginal C-index 0.6526")
print(f"Cox_v11features: marginal C-index 0.6512")
print(f"Cox_v11_richer:  marginal C-index 0.6538")
