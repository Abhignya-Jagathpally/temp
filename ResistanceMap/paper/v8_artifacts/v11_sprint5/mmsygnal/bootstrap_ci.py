"""Bootstrap 95% CI for mmSYGNAL marginal C-index on the REAL 787-patient MMRF cohort.

The bootstrap RESAMPLES patient indices from the real cohort (n=787 with replacement)
to estimate the sampling distribution of Harrell C-index. This does NOT generate
synthetic data — every resampled patient is a real MMRF patient with real TT2L outcome
and real mmSYGNAL risk score.

Uses sklearn.utils.resample for deterministic, reproducible resampling.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from lifelines.utils import concordance_index
from sklearn.utils import resample

ROOT = Path("/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap")
SP5 = ROOT / "paper/v8_artifacts/v11_sprint5"

risk = pd.read_csv("/tmp/our_mmsygnal_per_model.csv").set_index("patient_id")
clin = pd.read_csv(ROOT / "data/processed/mmrf_sprint4_analysis.tsv", sep="\t").set_index("submitter_id")
risk = risk.reindex(clin.index)

def route_risk(row, cyto):
    A_scores = []
    B_scores = []
    if cyto["cyto_t_4_14"] == 1: A_scores.append(row["t4_14"])
    if cyto["cyto_chr1q21_gain"] == 1: B_scores.append(row["amp1q"])
    if cyto["cyto_del13q"] == 1: B_scores.append(row["del13"])
    if A_scores: return float(np.mean(A_scores))
    if B_scores: return float(np.mean(B_scores))
    return float(row["agnostic"])

routed = np.array([route_risk(risk.loc[p], clin.loc[p]) for p in clin.index])
T = clin["tt2L_days"].values
E = clin["had_2L"].values

def c_idx(scores, t, e):
    return concordance_index(t, -scores, e)

obs = c_idx(routed, T, E)
print(f"observed marginal C-index = {obs:.4f}")

# Bootstrap 1000 reps via sklearn.utils.resample (resample REAL patient indices)
n = len(routed)
indices = np.arange(n)
boots = []
for i in range(1000):
    idx = resample(indices, n_samples=n, random_state=i)
    try:
        boots.append(c_idx(routed[idx], T[idx], E[idx]))
    except Exception:
        pass
boots = np.array(boots)
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (mean over reps={boots.mean():.4f})")

# Per-stratum
def stratum(row):
    if row["cyto_del17p"] == 1: return "S1_del17p"
    if row["cyto_t_4_14"] == 1: return "S2_t_4_14"
    if row["cyto_chr1q21_gain"] == 1: return "S3_1q21"
    if row["cyto_t_11_14"] == 1: return "S4_t_11_14"
    return "S5_other"

clin = clin.copy()
clin["stratum"] = clin.apply(stratum, axis=1)
strat = clin["stratum"].values

per_strat = {}
for s in sorted(set(strat)):
    mask = strat == s
    obs_s = c_idx(routed[mask], T[mask], E[mask])
    sub_idx = np.arange(int(mask.sum()))
    rr = routed[mask]; tt = T[mask]; ee = E[mask]
    sb = []
    for i in range(1000):
        bidx = resample(sub_idx, n_samples=len(sub_idx), random_state=10000 + i)
        try: sb.append(c_idx(rr[bidx], tt[bidx], ee[bidx]))
        except: pass
    sb = np.array(sb)
    per_strat[s] = {
        "n": int(mask.sum()),
        "n_events": int(ee.sum()),
        "c_index": float(obs_s),
        "ci95": [float(np.percentile(sb, 2.5)), float(np.percentile(sb, 97.5))],
    }
    print(f"  {s}: C={obs_s:.4f}  95%CI=[{per_strat[s]['ci95'][0]:.4f}, {per_strat[s]['ci95'][1]:.4f}]  n={mask.sum()} ev={ee.sum()}")

result = {
    "marginal": {"c_index": float(obs), "ci95": [float(ci_lo), float(ci_hi)], "n": int(n), "n_events": int(E.sum())},
    "per_stratum": per_strat,
    "v11_cox_marginal": {
        "Cox_v10": 0.6526,
        "Cox_v11features": 0.6512,
        "Cox_v11_richer": 0.6538,
    },
    "delta_vs_best_v11": float(obs - 0.6538),
    "note": "Bootstrap CIs are unpaired (mmSYGNAL alone). A paired Δ vs v11-Cox requires per-patient v11-Cox log-hazards; cox_discrimination.json only stored aggregates.",
    "n_bootstraps": 1000,
    "method": "sklearn.utils.resample over real patient indices, no synthetic data",
}
with open(SP5 / "mmsygnal" / "bootstrap_ci.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nDelta vs best v11 Cox:", obs - 0.6538)
