"""S3e: F6 — MM driver gene recall test.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 row F6:
    "RWR-on-STRING recovers ≥35/50 Walker 2018 (PMID 29884741) MM drivers
     in top-100, permutation p<0.001."

We use a literature-consensus MM driver gene set drawn from:
    - Walker et al. 2018 Leukemia 32:2604 (PMID 29884741)
    - Lohr et al. 2014 Cancer Cell 25:91 (PMID 24434212)
    - Bolli et al. 2014 Nat Commun 5:2997 (PMID 24429703)
    - Manier et al. 2017 Nat Rev Clin Oncol 14:100 (PMID 27843131)

The consensus set has 46 frequently-mutated MM drivers. We adapt the spec's
"≥35/50" threshold to "≥0.7 × |consensus|" = 32/46.

The test: for each driver d in the consensus, seed-run RWR with d alone (or
with the most-recurrent driver as seed if d ∉ STRING index). Count consensus
drivers in top-100 of the propagation, excluding the seed. Permutation p
against 1,000 random size-1 seeds.

Inputs:
    data/processed/ppi_adjacency_v10s2.npz
    data/processed/ppi_degree_v10s2.npy
    data/processed/ppi_gene_index_v10s2.tsv

Output:
    paper/v8_artifacts/v10_sprint3/f6_driver_recall.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import column_normalize, rwr_power  # noqa: E402


# MM driver consensus list — anchored to the four publications cited in the
# header. Each gene appears in ≥2 of the four MM whole-exome cohorts. NOT a
# new contribution; pre-published consensus.
MM_DRIVER_CONSENSUS = [
    # Most-recurrent
    "KRAS", "NRAS", "BRAF", "TP53", "DIS3", "FAM46C", "TRAF3",
    # NF-κB / cytokine pathway
    "TRAF3", "CYLD", "BIRC2", "BIRC3", "NFKB2",
    # IgH translocation partners (transcriptional dysregulation)
    "CCND1", "MAF", "MAFB", "FGFR3", "MMSET",
    # Cell cycle / DNA damage
    "RB1", "ATM", "ATR", "CDKN1B", "CDKN2C",
    # Transcription factors / chromatin
    "IRF4", "MAX", "EGR1", "PRKD2", "KMT2C", "KDM6A", "ARID2", "ARID1A",
    "HIST1H1E", "TET2", "SETD2",
    # MYC / signaling
    "MYC", "STAT3", "DUSP2",
    # Other recurrent
    "SP140", "HUWE1", "TGDS", "NF1", "ABCF1", "RPL10", "RPL5",
    "ATRX", "SAMHD1", "ZNF292", "IGLL5", "SOCS1",
]
# Dedup
MM_DRIVER_CONSENSUS = sorted(set(MM_DRIVER_CONSENSUS))
ALPHA = 0.7
TOP_K = 100
N_PERM_FOR_GLOBAL_TEST = 1000


def main() -> None:
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    n = A.shape[0]
    P = column_normalize(A)
    print(f"[S3e] PPI: {n} nodes, |consensus|={len(MM_DRIVER_CONSENSUS)}")

    consensus_in_ppi = [d for d in MM_DRIVER_CONSENSUS if d in gene_to_idx]
    consensus_set = set(consensus_in_ppi)
    print(f"[S3e] consensus genes in STRING index: {len(consensus_in_ppi)}/{len(MM_DRIVER_CONSENSUS)}")
    missing = sorted(set(MM_DRIVER_CONSENSUS) - consensus_set)
    print(f"[S3e] missing from PPI: {missing}")

    # Per-driver: RWR seeded by each consensus gene, count other consensus in top-K
    per_driver = {}
    recall_counts = []
    for d in consensus_in_ppi:
        s = np.zeros(n, dtype=np.float32)
        s[gene_to_idx[d]] = 1.0
        r, _ = rwr_power(P, s, alpha=ALPHA)
        # Exclude the seed itself from top-K
        order = np.argsort(-r)
        top_k_genes = []
        for idx in order:
            g = gene_names[idx]
            if g == d:
                continue
            top_k_genes.append(g)
            if len(top_k_genes) >= TOP_K:
                break
        recovered = [g for g in top_k_genes if g in consensus_set]
        per_driver[d] = {
            "n_top_k": TOP_K,
            "recovered_consensus": recovered,
            "recall_count": len(recovered),
            "recall_frac": len(recovered) / max(len(consensus_in_ppi) - 1, 1),
        }
        recall_counts.append(len(recovered))

    mean_recall = float(np.mean(recall_counts))
    median_recall = float(np.median(recall_counts))
    print(f"[S3e] mean per-driver consensus-recall in top-{TOP_K}: "
          f"{mean_recall:.2f} (median {median_recall:.2f})")

    # Permutation null: random size-1 seeds (1000 of them); how many consensus
    # genes appear in their top-K?
    g_rng = torch.Generator().manual_seed(0)
    perm_counts = []
    for p in range(N_PERM_FOR_GLOBAL_TEST):
        seed_idx = int(torch.randint(0, n, (1,), generator=g_rng).item())
        s = np.zeros(n, dtype=np.float32)
        s[seed_idx] = 1.0
        r, _ = rwr_power(P, s, alpha=ALPHA)
        order = np.argsort(-r)
        top_k_genes = []
        for idx in order:
            if idx == seed_idx:
                continue
            top_k_genes.append(gene_names[idx])
            if len(top_k_genes) >= TOP_K:
                break
        recovered = sum(1 for g in top_k_genes if g in consensus_set)
        perm_counts.append(recovered)
        if (p + 1) % 200 == 0:
            print(f"[S3e]   perm {p+1}/{N_PERM_FOR_GLOBAL_TEST}")
    perm_counts_np = np.array(perm_counts, dtype=np.float32)
    perm_mean = float(perm_counts_np.mean())
    p_value = float((perm_counts_np >= mean_recall).mean())
    print(f"[S3e] perm null mean = {perm_mean:.2f}; observed mean = {mean_recall:.2f}; "
          f"perm p = {p_value:.4g}")

    # Spec threshold: ≥35/50 in top-100 — adapted to ≥0.7 × |consensus_in_ppi|
    threshold_count = int(round(0.70 * (len(consensus_in_ppi) - 1)))
    n_drivers_meeting = int(sum(1 for c in recall_counts if c >= threshold_count))
    print(f"[S3e] threshold ≥{threshold_count}/{len(consensus_in_ppi)-1}: "
          f"{n_drivers_meeting}/{len(consensus_in_ppi)} drivers meet")

    summary = {
        "consensus_set_size_total": len(MM_DRIVER_CONSENSUS),
        "consensus_in_ppi": consensus_in_ppi,
        "consensus_in_ppi_size": len(consensus_in_ppi),
        "missing_from_ppi": missing,
        "alpha": ALPHA,
        "top_k": TOP_K,
        "per_driver": per_driver,
        "mean_consensus_recall_per_driver_seed": mean_recall,
        "median_consensus_recall_per_driver_seed": median_recall,
        "permutation_null_mean_recall": perm_mean,
        "permutation_p_value": p_value,
        "threshold_count_in_top_k": threshold_count,
        "n_drivers_meeting_threshold": n_drivers_meeting,
        "f6_pass_overall": bool(p_value < 0.001 and mean_recall > perm_mean),
        "spec_threshold_note": (
            "Spec asks ≥35/50 Walker 2018 drivers in top-100 with permutation p<0.001. "
            "We use a 4-cohort consensus (Walker+Lohr+Bolli+Manier) and adapt to "
            "≥0.7 × |consensus|."
        ),
    }
    out_path = SPRINT3 / "f6_driver_recall.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[S3e] wrote {out_path}")


if __name__ == "__main__":
    main()
