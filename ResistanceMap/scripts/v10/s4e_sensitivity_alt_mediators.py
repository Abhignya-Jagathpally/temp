"""S4e sensitivity: alternative mediator constructions.

The S4d primary mediator was the z-score-weighted projection of MMRF baseline
expression onto Sprint-3 Bortezomib-RWR top-50 (50 genes, NPI-z weights). This
script tests two alternatives to rule out the possibility that mediator
construction (rather than the underlying biology) is responsible for the L2
NIE refutation:

    M_seed3 : mean log1p TPM of PSMB5 + PSMB1 + PSMB2 only (seed genes)
    M_uniform : uniform-weighted mean of the same top-50 RWR genes

Both are computed on the same 787-patient cohort and run through the same Cox
NIE pipeline as S4d.

Output: paper/v8_artifacts/v10_sprint4/nie_alt_mediators.json
"""

from __future__ import annotations

import json
import sys
import time
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


def main() -> None:
    print("=== S4e sensitivity: alternative mediators ===")
    df_base = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df_base = df_base.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)
    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))
    proteasome_top50 = json.loads(
        (ROOT / "paper" / "v8_artifacts" / "v10_sprint3"
         / "bortezomib_top50_for_mediator.json").read_text()
    )
    proteasome_genes = [e["gene"] for e in proteasome_top50]

    out_summary = {}

    # --- Alternative 1: M_seed3 (PSMB5+PSMB1+PSMB2 only) ---
    seed3_ens = [sym_to_ens[g] for g in ("PSMB5", "PSMB1", "PSMB2")]
    sub3 = expr[seed3_ens].copy()
    sub3 = (sub3 - sub3.mean()) / (sub3.std(ddof=0) + 1e-9)
    M_seed3 = sub3.mean(axis=1).values
    print(f"M_seed3 distribution: mean={M_seed3.mean():.3f} sd={M_seed3.std():.3f}")
    df1 = df_base.drop(columns=["M_proteasome"], errors="ignore").copy()
    df1["M"] = pd.Series(M_seed3, index=expr.index).reindex(df1["submitter_id"]).values
    df1 = df1.dropna(subset=["M"]).reset_index(drop=True)

    print("\n[Alt-1: M_seed3 = PSMB5+1+2] fitting...")
    coefs1 = fit_cox_with_interaction(df1, mediator_col="M")
    decomp1 = decompose_effects(coefs1, df1, "M")
    print(f"  β_M = {coefs1['beta_M']:+.4f}  β_AM = {coefs1['beta_AM']:+.4f}  "
          f"NIE log-HR = {decomp1['NIE_log_hr']:+.4f}  "
          f"prop mediated = {decomp1.get('proportion_mediated') or 0:.2%}")

    boot1 = bootstrap_decomposition(df1, B=500, rng_seed=20260503, mediator_col="M")
    print(f"  bootstrap NIE log-HR 95%CI = {boot1['NIE_log_hr']['ci_95']}")

    out_summary["M_seed3_PSMB5_PSMB1_PSMB2"] = {
        "point": {
            "beta_M": coefs1["beta_M"],
            "beta_AM": coefs1["beta_AM"],
            "NIE_log_hr": decomp1["NIE_log_hr"],
            "proportion_mediated": decomp1.get("proportion_mediated"),
        },
        "bootstrap_B500": {
            "NIE_log_hr_mean": boot1["NIE_log_hr"]["mean"],
            "NIE_log_hr_ci95": boot1["NIE_log_hr"]["ci_95"],
            "NDE_log_hr_ci95": boot1["NDE_log_hr"]["ci_95"],
            "TE_log_hr_ci95":  boot1["TE_log_hr"]["ci_95"],
        },
    }

    # --- Alternative 2: M_uniform (top-50, uniform weights) ---
    rwr_ens = []
    for g in proteasome_genes:
        ens = sym_to_ens.get(g)
        if ens is not None and ens in expr.columns:
            rwr_ens.append(ens)
    sub_u = expr[rwr_ens].copy()
    sub_u = (sub_u - sub_u.mean()) / (sub_u.std(ddof=0) + 1e-9)
    M_uniform = sub_u.mean(axis=1).values
    print(f"\nM_uniform distribution: mean={M_uniform.mean():.3f} sd={M_uniform.std():.3f}")
    df2 = df_base.drop(columns=["M_proteasome"], errors="ignore").copy()
    df2["M"] = pd.Series(M_uniform, index=expr.index).reindex(df2["submitter_id"]).values
    df2 = df2.dropna(subset=["M"]).reset_index(drop=True)

    print("\n[Alt-2: M_uniform = top-50 uniform mean] fitting...")
    coefs2 = fit_cox_with_interaction(df2, mediator_col="M")
    decomp2 = decompose_effects(coefs2, df2, "M")
    print(f"  β_M = {coefs2['beta_M']:+.4f}  β_AM = {coefs2['beta_AM']:+.4f}  "
          f"NIE log-HR = {decomp2['NIE_log_hr']:+.4f}  "
          f"prop mediated = {decomp2.get('proportion_mediated') or 0:.2%}")

    boot2 = bootstrap_decomposition(df2, B=500, rng_seed=20260503, mediator_col="M")
    print(f"  bootstrap NIE log-HR 95%CI = {boot2['NIE_log_hr']['ci_95']}")

    out_summary["M_uniform_top50"] = {
        "point": {
            "beta_M": coefs2["beta_M"],
            "beta_AM": coefs2["beta_AM"],
            "NIE_log_hr": decomp2["NIE_log_hr"],
            "proportion_mediated": decomp2.get("proportion_mediated"),
        },
        "bootstrap_B500": {
            "NIE_log_hr_mean": boot2["NIE_log_hr"]["mean"],
            "NIE_log_hr_ci95": boot2["NIE_log_hr"]["ci_95"],
            "NDE_log_hr_ci95": boot2["NDE_log_hr"]["ci_95"],
            "TE_log_hr_ci95":  boot2["TE_log_hr"]["ci_95"],
        },
    }

    out_summary["proteasome_S4d_zweighted_top50"] = json.loads(
        (SPRINT4 / "nie_proteasome_tt2L.json").read_text()
    )["bootstrap_B1000"]["NIE_log_hr"]

    out_path = SPRINT4 / "nie_alt_mediators.json"
    out_path.write_text(json.dumps(out_summary, indent=2))
    print(f"\nsaved → {out_path}")

    # Comparative summary
    print("\n=== COMPARATIVE NIE log-HR 95% CI ===")
    print(f"  S4d: z-weighted top-50  : {out_summary['proteasome_S4d_zweighted_top50']['ci_95']}")
    print(f"  Alt1: PSMB5+1+2 raw     : {out_summary['M_seed3_PSMB5_PSMB1_PSMB2']['bootstrap_B500']['NIE_log_hr_ci95']}")
    print(f"  Alt2: top-50 uniform    : {out_summary['M_uniform_top50']['bootstrap_B500']['NIE_log_hr_ci95']}")
    print("\n(any 95% CI excluding 0 = significant NIE for that mediator)")


if __name__ == "__main__":
    main()
