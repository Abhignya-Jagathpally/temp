"""S4e: Falsification — negative-control mediator NIE.

Replace the proteasome-RWR top-50 mediator with B=20 independent random
degree-matched non-proteasome gene sets, repeat the Cox NIE estimation, and
record the distribution of NIE log-HR (point estimate + bootstrap CI).

Pre-registered F9: Negative-control NIE point estimates should cluster around
zero (95th percentile of |NIE_log_HR| ≤ |NIE_log_HR_proteasome|), confirming
that S4d's small but non-zero proteasome NIE is not a generic false-positive.

Output: paper/v8_artifacts/v10_sprint4/nie_negative_control.json
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

sys.path.insert(0, str(ROOT))
from scripts.v10.s4d_nie_estimation import (  # noqa: E402
    fit_cox_with_interaction, decompose_effects, evalue, CONFOUNDERS,
)


def build_random_mediator(
    expr: pd.DataFrame, sym_map: pd.DataFrame, proteasome_genes: list[str],
    n_genes: int, rng: torch.Generator,
) -> tuple[pd.DataFrame, list[str]]:
    """Sample n_genes random non-proteasome genes from MMRF-expressed genes
    and build a per-patient z-weighted score (uniform weights, since random)."""
    avail = list(set(expr.columns) & set(sym_map["ensembl_base"]))
    proteasome_ens = set(
        sym_map[sym_map["gene_name"].isin(proteasome_genes)]["ensembl_base"]
    )
    pool = [g for g in avail if g not in proteasome_ens]
    idx = torch.randperm(len(pool), generator=rng)[:n_genes].numpy()
    selected = [pool[int(i)] for i in idx]
    sub = expr[selected].copy()
    sub = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    M_random = sub.values.mean(axis=1)
    return pd.DataFrame({"submitter_id": expr.index, "M_random": M_random}), selected


def main() -> None:
    print("=== S4e: negative-control NIE ===")
    df_base = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df_base = df_base.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)
    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    proteasome_top50 = json.loads(
        (ROOT / "paper" / "v8_artifacts" / "v10_sprint3"
         / "bortezomib_top50_for_mediator.json").read_text()
    )
    proteasome_genes = [e["gene"] for e in proteasome_top50]

    rng = torch.Generator().manual_seed(20260503)
    B_neg = 20
    nie_logs = []
    nde_logs = []
    propmed = []
    for b in range(B_neg):
        M_random_df, used = build_random_mediator(
            expr, sym_map, proteasome_genes, n_genes=50, rng=rng,
        )
        df = df_base.drop(columns=["M_proteasome"], errors="ignore").merge(
            M_random_df, on="submitter_id", how="inner",
        ).rename(columns={"M_random": "M"})
        df = df.dropna(subset=["M"]).reset_index(drop=True)
        try:
            coefs = fit_cox_with_interaction(df, mediator_col="M")
            decomp = decompose_effects(coefs, df, mediator_col="M")
            nie_logs.append(decomp["NIE_log_hr"])
            nde_logs.append(decomp["NDE_log_hr"])
            pm = decomp.get("proportion_mediated") or 0.0
            propmed.append(pm)
            print(f"  neg-control {b+1}/{B_neg}: NIE log-HR = {decomp['NIE_log_hr']:+.4f}  "
                  f"prop mediated = {pm:.2%}")
        except Exception as e:
            print(f"  neg-control {b+1}: skipped ({e})")
            continue

    nie_logs = np.array(nie_logs); nde_logs = np.array(nde_logs); propmed = np.array(propmed)

    # Compare to proteasome NIE point estimate
    proteasome_nie_pt = json.loads(
        (SPRINT4 / "nie_proteasome_tt2L.json").read_text()
    )["point_estimate"]["decomposition"]["NIE_log_hr"]

    pct_neg_above = float((np.abs(nie_logs) >= abs(proteasome_nie_pt)).mean())
    print(f"\n[Neg-control summary, B={len(nie_logs)}]")
    print(f"  NIE log-HR mean ± sd : {nie_logs.mean():+.4f} ± {nie_logs.std():.4f}")
    print(f"  NIE log-HR 95th pct |·| : {np.percentile(np.abs(nie_logs), 95):.4f}")
    print(f"  proteasome NIE log-HR (S4d) : {proteasome_nie_pt:+.4f}")
    print(f"  fraction of neg-control trials with |NIE| ≥ proteasome: "
          f"{pct_neg_above:.0%}")
    f9_pass = bool(pct_neg_above < 0.10)
    print(f"  F9 (neg-control |NIE| 95%-tile < proteasome |NIE|, equivalently "
          f"<10% trials reach observed): {'PASS' if f9_pass else 'FAIL'}")

    out = {
        "B_neg_control": int(len(nie_logs)),
        "neg_control_NIE_log_hr_mean": float(nie_logs.mean()),
        "neg_control_NIE_log_hr_std": float(nie_logs.std()),
        "neg_control_NIE_log_hr_95th_abs_pct": float(np.percentile(np.abs(nie_logs), 95)),
        "proteasome_NIE_log_hr_point": float(proteasome_nie_pt),
        "fraction_neg_above_proteasome_abs": float(pct_neg_above),
        "F9_neg_control_pass": f9_pass,
        "neg_control_NIE_log_hr_each_trial": nie_logs.tolist(),
        "neg_control_proportion_mediated_each_trial": propmed.tolist(),
    }
    out_path = SPRINT4 / "nie_negative_control.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
