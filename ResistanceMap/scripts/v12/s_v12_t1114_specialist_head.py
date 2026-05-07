"""v12 / V11_END_TO_END §4.5: t(11;14) specialist head.

The v11.5 chair verdict (`docs/V11_CHAIR_VERDICT.md` §3.7 + §6) and the
internal end-to-end eval (`docs/V11_END_TO_END_EVALUATION.md` §2.2 weakness 4)
flag the S4 t(11;14) stratum as the only stratum stuck near chance:
    Cox_v11_richer        S4 C = 0.521
    Cox_v11_routed_mmsygnal S4 C = 0.643 (best v11.5 to date)
    mmSYGNAL                S4 C = 0.693 (post unbiased 6-submodel routing)

t(11;14) MM is biologically distinguished by IGH-CCND1 fusion drive of
CCND1 over-expression and BCL2-family co-dependence (Kumar 2017,
Chesi 2008, Touzeau 2014). v11_richer / mmSYGNAL routed do not encode
these axes explicitly. This sprint adds a stratum-interacted BCL2-family
specialist head to the v11.5 ensemble and tests whether S4 C-index lifts
without disturbing the marginal F_S5 calibrated coverage.

Specialist features (4 main, 4 interactions):
    BCL2_z, MCL1_z, CCND1_z, BCL2L1_z   (baseline TPM z-scores)
    BCL2_z × cyto_t_11_14, MCL1_z × cyto_t_11_14,
    CCND1_z × cyto_t_11_14, BCL2L1_z × cyto_t_11_14

Cox PH variants tested (lifelines, penalizer=0.1, LOO):
    Cox_v11_routed_mmsygnal              -- v11.5 baseline (24 feats)
    Cox_v11_t1114_specialist             -- v11.5 + 4 main effects (28 feats)
    Cox_v11_t1114_interaction_specialist -- v11.5 + 4 main + 4 interactions (32 feats)

Paired bootstrap (sklearn.utils.resample on real patient indices) of
Δ_C(S4) specialist vs baseline, B=10000.

Falsification criteria:
    PASS at S4 stratum if Δ_C(S4) > +0.05 with 95% CI excluding 0.
    PASS at marginal if Δ_C(marginal) >= 0 with 95% CI excluding -0.02
        (specialist must not regress the marginal).
    FAIL otherwise.

Inputs:
    data/processed/mmrf_sprint4_analysis.tsv
    data/processed/mmrf_baseline_expression.parquet
    data/processed/mmrf_ensembl_to_symbol.tsv
    data/processed/mmrf_z64.npy
    data/processed/mmrf_mmsygnal_per_model_scores.csv

Outputs:
    paper/v8_artifacts/v12_sprint1/t1114_specialist_head.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=Warning)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "v8_artifacts" / "v12_sprint1"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.v11.s5_v11_conformal_waddington_features import (  # noqa: E402
    CONFOUNDERS_V10, assign_strata, compute_waddington_features,
)
from scripts.v11.s5e_cox_with_programs import route_mmsygnal  # noqa: E402

SPECIALIST_GENES = ["BCL2", "MCL1", "CCND1", "BCL2L1"]
N_BOOT = 10000
SEED = 20260504


def cox_loo(df_features, t_col, e_col, penalizer=0.1):
    n = len(df_features)
    feature_cols = [c for c in df_features.columns if c not in (t_col, e_col)]
    log_hr = np.empty(n, dtype=np.float64)
    n_failed = 0
    for i in range(n):
        train = df_features.drop(index=i)
        test = df_features.loc[[i]]
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(train, duration_col=t_col, event_col=e_col, show_progress=False)
            log_hr[i] = float(cph.predict_log_partial_hazard(test[feature_cols]).iloc[0])
        except Exception:
            log_hr[i] = 0.0
            n_failed += 1
        if (i + 1) % 100 == 0:
            print(f"    Cox LOO {i+1}/{n}  (failed: {n_failed})")
    return log_hr


def discrimination(risk, t_days, event, strata):
    out = {"per_stratum": {}, "marginal": {}}
    for s in sorted(np.unique(strata)):
        in_s = strata == s
        if in_s.sum() < 10:
            continue
        c_s = float(concordance_index(t_days[in_s], -risk[in_s], event[in_s]))
        out["per_stratum"][s] = {"n": int(in_s.sum()), "c_index": c_s}
    out["marginal"]["c_index"] = float(concordance_index(t_days, -risk, event))
    for label_h, horizon in [("AUROC_12mo", 365), ("AUROC_24mo", 730)]:
        usable = (event == 1) | (t_days >= horizon)
        if usable.sum() < 30 or (event[usable] == 1).sum() < 5:
            out["marginal"][label_h] = None
            continue
        y_bin = ((t_days <= horizon) & (event == 1)).astype(int)[usable]
        if len(np.unique(y_bin)) < 2:
            out["marginal"][label_h] = None
            continue
        out["marginal"][label_h] = float(roc_auc_score(y_bin, risk[usable]))
    return out


def paired_bootstrap_delta_c(
    risk_a, risk_b, t_days, event, strata, target_stratum=None,
    n_boot=N_BOOT, seed=SEED,
):
    """Paired Δ_C bootstrap. If target_stratum provided, restrict to that
    stratum index; else compute marginal Δ_C."""
    if target_stratum is not None:
        in_s = strata == target_stratum
        risk_a = risk_a[in_s]
        risk_b = risk_b[in_s]
        t_days = t_days[in_s]
        event = event[in_s]
    n = len(risk_a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = resample(
            np.arange(n), replace=True, n_samples=n, random_state=seed + b,
        )
        if event[idx].sum() < 2:
            deltas[b] = np.nan
            continue
        c_a = concordance_index(t_days[idx], -risk_a[idx], event[idx])
        c_b = concordance_index(t_days[idx], -risk_b[idx], event[idx])
        deltas[b] = c_a - c_b
    deltas_clean = deltas[~np.isnan(deltas)]
    return {
        "n_boot": int(n_boot),
        "n_valid": int(len(deltas_clean)),
        "delta_mean": float(np.mean(deltas_clean)),
        "delta_median": float(np.median(deltas_clean)),
        "ci_95_lo": float(np.percentile(deltas_clean, 2.5)),
        "ci_95_hi": float(np.percentile(deltas_clean, 97.5)),
        "ci_99_lo": float(np.percentile(deltas_clean, 0.5)),
        "ci_99_hi": float(np.percentile(deltas_clean, 99.5)),
        "p_two_sided": float(2 * min(
            (deltas_clean > 0).mean(), (deltas_clean < 0).mean()
        )),
        "p_a_better": float((deltas_clean > 0).mean()),
    }


def main():
    t_start = time.time()
    print("=== v12 S1: t(11;14) specialist head ===")

    df = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df = df.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)

    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))

    seed3_ens = [sym_to_ens[g] for g in ("PSMB5", "PSMB1", "PSMB2")]
    sub = expr[seed3_ens].copy()
    sub = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    M_seed3 = sub.mean(axis=1)
    df = df.merge(
        pd.DataFrame({"submitter_id": expr.index, "M_seed3": M_seed3.values}),
        on="submitter_id", how="inner",
    ).reset_index(drop=True)

    Z = np.load(PROC / "mmrf_z64.npy")
    ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    z_idx = np.array([id_to_idx[sid] for sid in df["submitter_id"]], dtype=np.int64)
    z0_pat = Z[z_idx]

    sc = pd.read_csv(PROC / "mmrf_mmsygnal_per_model_scores.csv")
    df_short_id = df["submitter_id"].str.replace(r"_\d+_\w+$", "", regex=True)
    df["mm_id"] = df_short_id
    sc = sc.set_index("patient_id")
    sc_aligned = sc.loc[df["mm_id"].values].reset_index(drop=True)
    print(f"mmSYGNAL alignment: per-model-scores {sc_aligned.shape}")

    strata = assign_strata(df)
    t_days = df["tt2L_days"].values.astype(np.float64) + 1e-3
    event = df["had_2L"].values.astype(np.int64)
    print(f"N={len(df)}; events={int(event.sum())} ({100*event.mean():.1f}%)")

    base_v10 = pd.DataFrame({
        "bort_1L": df["bort_1L"].astype(float),
        "M_seed3": df["M_seed3"].astype(float),
        "bort_x_M": (df["bort_1L"] * df["M_seed3"]).astype(float),
        **{c: df[c].astype(float) for c in CONFOUNDERS_V10},
    })
    X_wadd = compute_waddington_features(z0_pat, T=1.0)
    X_wadd_std = (X_wadd - X_wadd.mean(axis=0)) / (X_wadd.std(axis=0, ddof=0) + 1e-9)
    wadd = pd.DataFrame(X_wadd_std, columns=["U_z0", "U_zT", "grad_norm", "displacement"])
    Zc = z0_pat - z0_pat.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    z_pcs = (Zc @ Vt[:8].T)
    z_pcs = (z_pcs - z_pcs.mean(axis=0)) / (z_pcs.std(axis=0, ddof=0) + 1e-9)
    z_pc_df = pd.DataFrame(z_pcs, columns=[f"z_pc{k+1}" for k in range(8)])

    df_v11_richer = pd.concat([base_v10, wadd, z_pc_df], axis=1)

    routed = route_mmsygnal(sc_aligned, df)
    routed_std = (routed - routed.mean()) / (routed.std(ddof=0) + 1e-9)
    routed_feat = pd.DataFrame({"mmsygnal_routed": routed_std})

    spec_ens = []
    avail_genes = []
    for g in SPECIALIST_GENES:
        ens = sym_to_ens.get(g)
        if ens is None or ens not in expr.columns:
            print(f"  WARN: {g} not in expression matrix; skipping")
            continue
        spec_ens.append(ens)
        avail_genes.append(g)
    print(f"Specialist genes available: {avail_genes}")

    spec_expr = expr[spec_ens].copy()
    spec_expr = (spec_expr - spec_expr.mean()) / (spec_expr.std(ddof=0) + 1e-9)
    spec_expr.columns = [f"{g}_z" for g in avail_genes]
    spec_aligned = spec_expr.loc[df["submitter_id"].values].reset_index(drop=True)

    cyto_t1114 = df["cyto_t_11_14"].astype(float)
    interact = pd.DataFrame({
        f"{g}_z_x_t1114": spec_aligned[f"{g}_z"] * cyto_t1114
        for g in avail_genes
    })

    df_v115 = pd.concat([df_v11_richer, routed_feat], axis=1)
    df_v12_main = pd.concat([df_v115, spec_aligned], axis=1)
    df_v12_full = pd.concat([df_v12_main, interact], axis=1)

    variants = {
        "Cox_v11_routed_mmsygnal": pd.concat(
            [df_v115, pd.DataFrame({"t": t_days, "e": event})], axis=1),
        "Cox_v11_t1114_specialist": pd.concat(
            [df_v12_main, pd.DataFrame({"t": t_days, "e": event})], axis=1),
        "Cox_v11_t1114_interaction_specialist": pd.concat(
            [df_v12_full, pd.DataFrame({"t": t_days, "e": event})], axis=1),
    }

    summary = {}
    log_hr_per_variant = {}
    for label, df_x in variants.items():
        n_feat = df_x.shape[1] - 2
        print(f"\n=== {label} ({n_feat} feats, LOO Cox PH) ===")
        t0 = time.time()
        log_hr = cox_loo(df_x, "t", "e", penalizer=0.1)
        print(f"  done in {time.time()-t0:.1f}s")
        summary[label] = discrimination(log_hr, t_days, event, strata)
        summary[label]["n_features"] = int(n_feat)
        log_hr_per_variant[label] = log_hr
        m = summary[label]["marginal"]
        print(f"  marginal C={m['c_index']:.4f} AUC12={m.get('AUROC_12mo')}")
        for s, r in summary[label]["per_stratum"].items():
            print(f"    S{s} n={r['n']} C={r['c_index']:.4f}")

    boot = {}
    risk_baseline = log_hr_per_variant["Cox_v11_routed_mmsygnal"]
    for variant in ("Cox_v11_t1114_specialist",
                    "Cox_v11_t1114_interaction_specialist"):
        risk_v = log_hr_per_variant[variant]
        print(f"\n=== Bootstrap Δ_C  {variant}  vs  baseline (B={N_BOOT}) ===")
        boot[variant] = {}
        for s in sorted(np.unique(strata)):
            in_s = strata == s
            if in_s.sum() < 10:
                continue
            r = paired_bootstrap_delta_c(
                risk_v, risk_baseline, t_days, event, strata,
                target_stratum=s, n_boot=N_BOOT,
            )
            boot[variant][f"S{s}"] = r
            print(f"  S{s} n={int(in_s.sum())}  Δ_mean={r['delta_mean']:+.4f}  "
                  f"95%CI=[{r['ci_95_lo']:+.4f},{r['ci_95_hi']:+.4f}]  "
                  f"p_two={r['p_two_sided']:.3f}")
        r_marg = paired_bootstrap_delta_c(
            risk_v, risk_baseline, t_days, event, strata,
            target_stratum=None, n_boot=N_BOOT,
        )
        boot[variant]["marginal"] = r_marg
        print(f"  marginal n={len(t_days)}  Δ_mean={r_marg['delta_mean']:+.4f}  "
              f"95%CI=[{r_marg['ci_95_lo']:+.4f},{r_marg['ci_95_hi']:+.4f}]  "
              f"p_two={r_marg['p_two_sided']:.3f}")

    def evaluate(variant_label):
        s4_b = boot[variant_label].get("S4")
        marg = boot[variant_label]["marginal"]
        verdict = {}
        if s4_b is None:
            verdict["S4"] = "NA"
        elif s4_b["delta_mean"] > 0.05 and s4_b["ci_95_lo"] > 0:
            verdict["S4"] = "PASS"
        elif s4_b["delta_mean"] > 0 and s4_b["ci_95_lo"] > 0:
            verdict["S4"] = "WEAK_PASS"
        elif s4_b["ci_95_hi"] < 0:
            verdict["S4"] = "REGRESS"
        else:
            verdict["S4"] = "NULL"
        if marg["ci_95_lo"] >= -0.02:
            verdict["marginal"] = "NO_REGRESSION"
        else:
            verdict["marginal"] = "REGRESSION_RISK"
        return verdict

    verdicts = {v: evaluate(v) for v in
                ("Cox_v11_t1114_specialist", "Cox_v11_t1114_interaction_specialist")}

    elapsed = time.time() - t_start
    out = {
        "n_patients": int(len(df)),
        "n_events": int(event.sum()),
        "specialist_genes_used": avail_genes,
        "n_t1114_in_strict_S4": int((strata == 4).sum()),
        "n_t1114_soft_count": int((df["cyto_t_11_14"] == 1).sum()),
        "n_boot": N_BOOT,
        "summary": summary,
        "paired_bootstrap": boot,
        "verdict": verdicts,
        "verdict_rule": (
            "S4 PASS if Δ_mean > +0.05 with 95% CI excluding 0; "
            "WEAK_PASS if Δ_mean > 0 and CI excludes 0; "
            "NULL if CI contains 0; REGRESS if upper CI < 0. "
            "Marginal: NO_REGRESSION if 95% CI lower bound >= -0.02."
        ),
        "wall_time_s": elapsed,
    }
    out_path = OUT / "t1114_specialist_head.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {out_path}")
    print(f"[OK] wall time: {elapsed:.1f}s")

    print("\n=== FINAL VERDICT ===")
    base = summary["Cox_v11_routed_mmsygnal"]
    for v in ("Cox_v11_t1114_specialist",
              "Cox_v11_t1114_interaction_specialist"):
        s = summary[v]
        print(f"\n{v}:")
        print(f"  marginal C: baseline {base['marginal']['c_index']:.4f} -> "
              f"{s['marginal']['c_index']:.4f}  "
              f"(Δ = {s['marginal']['c_index'] - base['marginal']['c_index']:+.4f})")
        s4_b = base["per_stratum"].get(4)
        s4_v = s["per_stratum"].get(4)
        if s4_b is not None and s4_v is not None:
            print(f"  S4 t(11;14) C: baseline {s4_b['c_index']:.4f} -> "
                  f"{s4_v['c_index']:.4f}  "
                  f"(Δ = {s4_v['c_index'] - s4_b['c_index']:+.4f})")
        print(f"  Verdict: {verdicts[v]}")


if __name__ == "__main__":
    main()
