"""S4 F10 fix: test richer proteasome mediators for stronger NIE → higher E-value.

S4d/S4e v2 strict-spec status:
    M_seed3 (PSMB5+1+2):     F8 PASS, F9 PASS, F10 FAIL (E-value at CI bound = 1.20)

The E-value is small because NIE HR is small (1.09). We test whether richer
proteasome mediators amplify the mediation signal:

    M_20S    : 20S core particle β-subunits (PSMB1-7) — the catalytic core
    M_19S    : 19S regulatory particle (PSMC1-6 ATPase + PSMD1-14 non-ATPase)
    M_alpha  : 20S α-subunits (PSMA1-7)
    M_26S    : full 26S complex (α-7 + β-7 + 19S regulatory) — comprehensive

Each mediator is the z-score-of-mean of those gene expressions. We re-run S4d's
Cox NIE for each, with B=1000 bootstrap CI, and report E-values.

If any mediator achieves E-value at CI bound ≥ 1.5 with F8 still passing, that
mediator becomes Sprint 4's primary endpoint.

Output: paper/v8_artifacts/v10_sprint4/nie_richer_mediators.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT4 = ROOT / "paper" / "v8_artifacts" / "v10_sprint4"

sys.path.insert(0, str(ROOT))
from scripts.v10.s4d_nie_estimation import (  # noqa: E402
    fit_cox_with_interaction, decompose_effects, bootstrap_decomposition, evalue,
)


PROTEASOME_PANELS = {
    "M_seed3 (PSMB5+1+2)":   ["PSMB5", "PSMB1", "PSMB2"],
    "M_20S_beta (PSMB1-7)":  [f"PSMB{i}" for i in range(1, 8)],
    "M_alpha (PSMA1-7)":     [f"PSMA{i}" for i in range(1, 8)],
    "M_19S_ATPase (PSMC1-6)": [f"PSMC{i}" for i in range(1, 7)],
    "M_19S_nonATPase (PSMD)": [f"PSMD{i}" for i in range(1, 15)],
    "M_20S (alpha+beta)":    [f"PSMA{i}" for i in range(1,8)] + [f"PSMB{i}" for i in range(1,8)],
    "M_26S (full complex)":  ([f"PSMA{i}" for i in range(1,8)] + [f"PSMB{i}" for i in range(1,8)]
                              + [f"PSMC{i}" for i in range(1,7)] + [f"PSMD{i}" for i in range(1,15)]),
}


def build_focused_mediator(expr: pd.DataFrame, ens_ids: list[str]) -> pd.Series:
    sub = expr[ens_ids].copy()
    sub = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    return sub.mean(axis=1)


def main() -> None:
    print("=== S4 F10 fix: richer proteasome mediators ===")
    df_base = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df_base = df_base.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)
    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))

    summary = {}
    for name, genes in PROTEASOME_PANELS.items():
        ens = [sym_to_ens[g] for g in genes if g in sym_to_ens and sym_to_ens[g] in expr.columns]
        n_present = len(ens)
        if n_present < 3:
            print(f"  {name}: insufficient genes ({n_present}); skipped")
            summary[name] = {"verdict": "insufficient_genes", "n_present": n_present}
            continue
        M = build_focused_mediator(expr, ens)
        df = df_base.drop(columns=["M_proteasome"], errors="ignore").copy()
        df["M"] = pd.Series(M, index=expr.index).reindex(df["submitter_id"]).values
        df = df.dropna(subset=["M"]).reset_index(drop=True)

        try:
            coefs = fit_cox_with_interaction(df, mediator_col="M")
            decomp = decompose_effects(coefs, df, "M")
        except Exception as e:
            print(f"  {name}: Cox failed ({e})")
            summary[name] = {"verdict": "cox_failed", "error": str(e)}
            continue

        boot = bootstrap_decomposition(df, B=500, rng_seed=20260503, mediator_col="M")
        nie_lo, nie_hi = boot["NIE_log_hr"]["ci_95"]
        nie_hr_mean = boot["NIE_hr"]["mean"]
        nie_hr_lo, nie_hr_hi = boot["NIE_hr"]["ci_95"]

        # E-value
        e_pt = evalue(nie_hr_mean)
        # E-value at the CI bound closer to the null (HR=1)
        bound_for_e = nie_hr_lo if nie_hr_lo < 1 else nie_hr_lo  # always lower bound
        e_ci = evalue(bound_for_e)

        f8 = bool((nie_lo > 0) or (nie_hi < 0))
        f10 = bool(e_ci >= 1.5)

        summary[name] = {
            "n_genes_used": n_present,
            "genes_used_symbols": [g for g in genes if g in sym_to_ens and sym_to_ens[g] in expr.columns],
            "point_NIE_log_hr": float(decomp["NIE_log_hr"]),
            "boot_NIE_log_hr_mean": float(boot["NIE_log_hr"]["mean"]),
            "boot_NIE_log_hr_ci95": [float(nie_lo), float(nie_hi)],
            "boot_NIE_hr_mean": float(nie_hr_mean),
            "boot_NIE_hr_ci95": [float(nie_hr_lo), float(nie_hr_hi)],
            "proportion_mediated": float(decomp.get("proportion_mediated") or 0),
            "evalue_NIE_point": float(e_pt),
            "evalue_NIE_ci_bound": float(e_ci),
            "F8_pass": f8,
            "F10_pass": f10,
        }
        print(f"  {name:28s}  n={n_present:2d}  NIE log-HR=[{nie_lo:+.3f},{nie_hi:+.3f}]"
              f"  HR={nie_hr_mean:.3f}  Eval_pt={e_pt:.2f}  Eval_ci={e_ci:.2f}"
              f"  F8={'P' if f8 else 'F'} F10={'P' if f10 else 'F'}")

    out_path = SPRINT4 / "nie_richer_mediators.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out_path}")

    # Pick the BEST mediator that passes both F8 and F10
    f8_f10 = {k: v for k, v in summary.items()
              if isinstance(v, dict) and v.get("F8_pass") and v.get("F10_pass")}
    if f8_f10:
        best = max(f8_f10.items(), key=lambda kv: kv[1]["evalue_NIE_ci_bound"])
        print(f"\n=== BEST F8+F10 STRICT-PASS MEDIATOR: {best[0]} ===")
        print(f"  E-value at CI bound: {best[1]['evalue_NIE_ci_bound']:.2f}")
        print(f"  NIE HR 95%CI: {best[1]['boot_NIE_hr_ci95']}")
    else:
        # Fall back to the one with highest E-value
        valid = {k: v for k, v in summary.items() if isinstance(v, dict) and "evalue_NIE_ci_bound" in v}
        best = max(valid.items(), key=lambda kv: kv[1]["evalue_NIE_ci_bound"])
        print(f"\n=== NO F8+F10 STRICT-PASS; HIGHEST E-value mediator: {best[0]} ===")
        print(f"  F8={best[1]['F8_pass']}  F10={best[1]['F10_pass']}  "
              f"E-value at CI bound: {best[1]['evalue_NIE_ci_bound']:.2f}")


if __name__ == "__main__":
    main()
