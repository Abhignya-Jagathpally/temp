"""
Pilot: PPI propagation validation — RWR on STRING vs DepMap MM essentiality.

Run:
    python3 scripts/pilot_ppi_propagation.py

Stdlib + numpy + pandas + networkx + scipy only.
Permutation baseline shuffles COLUMNS of real DepMap data — no synthetic values.
"""

import json
import gzip
import sys
import time
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import ranksums, spearmanr

warnings.filterwarnings("ignore")

ROOT = Path("/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap")
STRING_FILE = ROOT / "data/raw/string_ppi.txt"
STRING_INFO = ROOT / "data/raw/string/9606.protein.info.v12.0.txt.gz"
DEPMAP_FILE = ROOT / "data/raw/depmap/CRISPRGeneEffect.csv"
DRUG_META   = ROOT / "data/raw/drug_metadata.json"

MM_CELL_LINES = ["ACH-000035", "ACH-000547", "ACH-000001"]  # OPM2, KMS34, U266
ALPHA = 0.7
TOP_K = 50
N_PERM = 1000
MIN_CONF = 700  # STRING combined_score threshold

# Seed Python stdlib random for reproducible permutations (uses real data columns)
random.seed(42)

# ---------------------------------------------------------------------------
# 1. Build ENSP -> gene-name mapping from STRING protein info
# ---------------------------------------------------------------------------
def load_string_info(path: Path) -> dict:
    print("[1/5] Loading STRING protein info (ENSP -> gene name) ...")
    mapping = {}
    with gzip.open(path, "rt") as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ensp = parts[0]   # e.g. 9606.ENSP00000000233
                gene = parts[1]   # preferred_name e.g. ARF5
                mapping[ensp] = gene
    print(f"    {len(mapping):,} ENSP IDs mapped to gene names.")
    return mapping

# ---------------------------------------------------------------------------
# 2. Build STRING undirected weighted graph (gene-name nodes)
# ---------------------------------------------------------------------------
def build_graph(string_file: Path, ensp_to_gene: dict) -> nx.Graph:
    print("[2/5] Building STRING graph (confidence >= 700) ...")
    G = nx.Graph()
    skipped = 0
    with open(string_file) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])
            if score < MIN_CONF:
                continue
            g1 = ensp_to_gene.get(p1)
            g2 = ensp_to_gene.get(p2)
            if g1 is None or g2 is None:
                skipped += 1
                continue
            w = score / 1000.0
            if G.has_edge(g1, g2):
                if G[g1][g2]["weight"] < w:
                    G[g1][g2]["weight"] = w
            else:
                G.add_edge(g1, g2, weight=w)
    print(f"    Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"(skipped {skipped:,} unmapped).")
    return G

# ---------------------------------------------------------------------------
# 3. RWR — Random Walk with Restart
# r = alpha * W_norm * r + (1-alpha) * s
# W_norm is column-normalised adjacency (weighted by STRING confidence).
# ---------------------------------------------------------------------------
def rwr(G: nx.Graph, seed_gene: str, alpha: float = 0.7, max_iter: int = 100,
        tol: float = 1e-6) -> pd.Series:
    """Return a Series of RWR scores indexed by gene name, sorted descending."""
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {g: i for i, g in enumerate(nodes)}

    if seed_gene not in idx:
        return None

    # Build column-normalised adjacency dict for efficient power iteration
    adj_norm = {}
    for v in nodes:
        nbrs = list(G[v].items())
        total_w = sum(d["weight"] for _, d in nbrs)
        if total_w == 0:
            adj_norm[v] = []
        else:
            adj_norm[v] = [(u, d["weight"] / total_w) for u, d in nbrs]

    seed_idx = idx[seed_gene]
    s = np.zeros(n)
    s[seed_idx] = 1.0

    r = s.copy()
    for _ in range(max_iter):
        r_new = np.zeros(n)
        for v, v_idx in idx.items():
            for u, w_norm in adj_norm[v]:
                r_new[idx[u]] += w_norm * r[v_idx]
        r_new = alpha * r_new + (1 - alpha) * s
        diff = np.abs(r_new - r).sum()
        r = r_new
        if diff < tol:
            break

    return pd.Series(r, index=nodes).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# 4. Load DepMap — extract MM cell lines
# ---------------------------------------------------------------------------
def load_depmap_mm(path: Path, cell_lines: list) -> pd.DataFrame:
    print("[3/5] Loading DepMap CRISPR Chronos scores for MM cell lines ...")
    df = pd.read_csv(path, index_col=0)
    # Columns are like "GENE (entrez_id)" — strip to gene symbol
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    available = [c for c in cell_lines if c in df.index]
    missing   = [c for c in cell_lines if c not in df.index]
    if missing:
        print(f"    WARNING: cell lines not found: {missing}")
    mm_df = df.loc[available]
    print(f"    MM cell lines loaded: {available} — {mm_df.shape[1]:,} genes.")
    return mm_df

# ---------------------------------------------------------------------------
# 5. Per-drug validation
# ---------------------------------------------------------------------------
def drug_validation(drug_name: str, seed_gene: str, G: nx.Graph,
                    mm_df: pd.DataFrame, top_k: int = TOP_K, n_perm: int = N_PERM):
    """Returns dict with results for one drug."""
    result = {
        "drug": drug_name,
        "seed_gene": seed_gene,
        "in_graph": seed_gene in G,
        "top_10": [],
        "mean_chronos_top_k": np.nan,
        "mean_chronos_random": np.nan,
        "p_value": np.nan,
        "p_bonferroni": np.nan,
        "hub_ranks": {},
        "rwr_scores": None,
    }

    if not result["in_graph"]:
        print(f"    [{drug_name}] seed '{seed_gene}' NOT in STRING — skipping.")
        return result

    t0 = time.time()
    scores = rwr(G, seed_gene, alpha=ALPHA)
    elapsed = time.time() - t0

    if scores is None:
        result["in_graph"] = False
        return result

    result["rwr_scores"] = scores
    result["top_10"] = scores.index[:10].tolist()

    # Hub bias: rank of TP53, MYC, AKT1 in RWR
    for hub in ["TP53", "MYC", "AKT1"]:
        if hub in scores.index:
            result["hub_ranks"][hub] = int((scores.index == hub).argmax()) + 1
        else:
            result["hub_ranks"][hub] = "absent"

    # DepMap overlap: Chronos for top_k genes that have DepMap coverage
    genes_in_depmap = [g for g in scores.index[:top_k] if g in mm_df.columns]
    if len(genes_in_depmap) < 5:
        print(f"    [{drug_name}] Only {len(genes_in_depmap)} top-{top_k} genes in DepMap.")
        return result

    topk_chronos = mm_df[genes_in_depmap].values.flatten()
    topk_chronos = topk_chronos[~np.isnan(topk_chronos)]

    # Permutation: sample random columns from real DepMap data (no synthetic values)
    all_depmap_genes = mm_df.columns.tolist()
    k = len(genes_in_depmap)
    perm_means = []
    for _ in range(n_perm):
        samp = random.sample(all_depmap_genes, k)
        vals = mm_df[samp].values.flatten()
        vals = vals[~np.isnan(vals)]
        perm_means.append(float(vals.mean()))

    perm_arr = np.array(perm_means)
    result["mean_chronos_top_k"] = float(topk_chronos.mean())
    result["mean_chronos_random"] = float(perm_arr.mean())

    # One-sided Wilcoxon: is top-k more essential (lower Chronos) than permutations?
    stat, p = ranksums(topk_chronos, perm_arr, alternative="less")
    result["p_value"] = float(p)

    print(f"    [{drug_name}] seed={seed_gene} | top-k Chr={result['mean_chronos_top_k']:.4f} "
          f"| random Chr={result['mean_chronos_random']:.4f} | p={p:.3e}  ({elapsed:.1f}s)")
    return result

# ---------------------------------------------------------------------------
# 6. Cross-drug specificity: Spearman pairwise on RWR rankings
# ---------------------------------------------------------------------------
def cross_drug_specificity(results: list) -> pd.DataFrame:
    drugs_with_scores = [(r["drug"], r["rwr_scores"]) for r in results
                         if r["rwr_scores"] is not None]
    labels = [d for d, _ in drugs_with_scores]
    n = len(drugs_with_scores)
    corr_mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    for i in range(n):
        corr_mat.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            d1, s1 = drugs_with_scores[i]
            d2, s2 = drugs_with_scores[j]
            common = s1.index.intersection(s2.index)
            if len(common) < 100:
                continue
            rho, _ = spearmanr(s1[common].values, s2[common].values)
            corr_mat.loc[d1, d2] = round(rho, 3)
            corr_mat.loc[d2, d1] = round(rho, 3)
    return corr_mat

# ---------------------------------------------------------------------------
# 7. Proteasome subunit ranks for Bortezomib (Question A)
# ---------------------------------------------------------------------------
PROTEASOME_GENES = (
    [f"PSMA{i}" for i in range(1, 8)] +
    [f"PSMB{i}" for i in range(1, 11)] +
    [f"PSMC{i}" for i in range(1, 7)] +
    [f"PSMD{i}" for i in range(1, 15)]
)

def bortezomib_proteasome_check(btz_result: dict) -> list:
    if btz_result is None or btz_result["rwr_scores"] is None:
        return []
    scores = btz_result["rwr_scores"]
    all_genes = scores.index.tolist()
    rows = []
    for g in PROTEASOME_GENES:
        if g in scores.index:
            rank = all_genes.index(g) + 1
            rows.append((g, rank))
    rows.sort(key=lambda x: x[1])
    return rows

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ensp_to_gene = load_string_info(STRING_INFO)
    G = build_graph(STRING_FILE, ensp_to_gene)

    with open(DRUG_META) as fh:
        drug_meta = json.load(fh)

    mm_df = load_depmap_mm(DEPMAP_FILE, MM_CELL_LINES)

    print("[4/5] Running RWR + DepMap validation per drug ...")
    drug_order = [
        "Bortezomib", "Lenalidomide", "Panobinostat", "Vorinostat", "Romidepsin",
        "Venetoclax", "Dinaciclib", "Palbociclib", "Doxorubicin", "Etoposide",
        "Cyclophosphamide",
    ]

    results = []
    for drug in drug_order:
        meta = drug_meta.get(drug, {})
        seed = meta.get("primary_target_gene_symbol", "")
        if not seed or seed == "DNA":
            print(f"    [{drug}] Non-protein target ('{seed}') — excluded from propagation.")
            results.append({
                "drug": drug, "seed_gene": seed, "in_graph": False,
                "top_10": [], "mean_chronos_top_k": np.nan,
                "mean_chronos_random": np.nan, "p_value": np.nan,
                "p_bonferroni": np.nan, "hub_ranks": {}, "rwr_scores": None,
            })
        else:
            results.append(drug_validation(drug, seed, G, mm_df))

    # Bonferroni correction
    n_tests = sum(1 for r in results if not np.isnan(r["p_value"]))
    for r in results:
        if not np.isnan(r["p_value"]):
            r["p_bonferroni"] = min(r["p_value"] * n_tests, 1.0)

    # Cross-drug specificity
    print("[5/5] Computing cross-drug Spearman correlations ...")
    corr_df = cross_drug_specificity(results)

    # Proteasome check
    btz_result = next((r for r in results if r["drug"] == "Bortezomib"), None)
    psm_ranks = bortezomib_proteasome_check(btz_result)

    # ----------------------------------------------------------------
    # Print console summary
    # ----------------------------------------------------------------
    print("\n" + "="*100)
    print("RESULTS TABLE")
    print("="*100)
    print(f"{'Drug':<18} {'Seed':<8} {'InGraph':<8} {'Mean_Chr_K':>11} {'Mean_Chr_R':>11} {'p_raw':>10} {'p_bonf':>10}")
    print("-"*100)
    for r in results:
        print(f"{r['drug']:<18} {r['seed_gene']:<8} {str(r['in_graph']):<8} "
              f"{r['mean_chronos_top_k']:>11.4f} {r['mean_chronos_random']:>11.4f} "
              f"{r['p_value']:>10.3e} {r['p_bonferroni']:>10.3e}")

    print("\n--- Hub ranks (rank in full RWR score vector) ---")
    print(f"{'Drug':<18} {'TP53':>8} {'MYC':>8} {'AKT1':>8}")
    for r in results:
        h = r["hub_ranks"]
        print(f"{r['drug']:<18} {str(h.get('TP53','—')):>8} {str(h.get('MYC','—')):>8} {str(h.get('AKT1','—')):>8}")

    print("\n--- Bortezomib: proteasome subunit ranks (top-15) ---")
    for g, rank in psm_ranks[:15]:
        print(f"  {g}: rank {rank}")

    print("\n--- Cross-drug Spearman correlation ---")
    print(corr_df.to_string())

    # ----------------------------------------------------------------
    # Save artefacts
    # ----------------------------------------------------------------
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    corr_df.to_csv(docs_dir / "ppi_cross_drug_corr.csv")

    rows_out = []
    for r in results:
        rows_out.append({
            "drug": r["drug"],
            "seed_gene": r["seed_gene"],
            "in_graph": r["in_graph"],
            "top_10": "; ".join(r["top_10"]),
            "mean_chronos_top_k": r["mean_chronos_top_k"],
            "mean_chronos_random": r["mean_chronos_random"],
            "p_value_raw": r["p_value"],
            "p_bonferroni": r["p_bonferroni"],
            "hub_TP53_rank": r["hub_ranks"].get("TP53", "N/A"),
            "hub_MYC_rank":  r["hub_ranks"].get("MYC",  "N/A"),
            "hub_AKT1_rank": r["hub_ranks"].get("AKT1", "N/A"),
        })
    pd.DataFrame(rows_out).to_csv(docs_dir / "ppi_pilot_results.csv", index=False)

    write_markdown(results, corr_df, psm_ranks, docs_dir, n_tests)

    return results, corr_df, psm_ranks


def write_markdown(results, corr_df, psm_ranks, docs_dir: Path, n_tests: int):
    valid_results = [r for r in results if not np.isnan(r["p_value"])]
    sig_bonf     = [r for r in valid_results if r["p_bonferroni"] < 0.05]
    sig_nominal  = [r for r in valid_results if r["p_value"] < 0.05]

    frac_sig = len(sig_bonf) / max(len(valid_results), 1)
    if frac_sig >= 0.5:
        verdict = "PASS"
        verdict_note = "Majority of tested drugs show Bonferroni-significant essentiality enrichment in MM DepMap lines."
        recommendation = (
            "RWR on STRING recovers mechanism-relevant proteins and shows statistically "
            "significant essentiality enrichment in MM DepMap lines. Keep RWR as the "
            "propagation primitive for v10, with multi-seed extension (§Required changes)."
        )
    elif len(sig_nominal) >= 3:
        verdict = "CONDITIONAL"
        verdict_note = "Several drugs reach nominal significance but Bonferroni threshold is not broadly met."
        recommendation = (
            "RWR shows partial mechanistic recovery. Consider: (a) raising alpha toward 0.85, "
            "(b) multi-seed (all ChEMBL targets per drug), (c) tissue-specific expression weighting. "
            "Conditional retention for v10 pending sensitivity analysis."
        )
    else:
        verdict = "FAIL"
        verdict_note = ("RWR top-k genes are not significantly more essential than random in MM lines. "
                        "Low power (n=3 cell lines) is a confound.")
        recommendation = (
            "Expand to all haematological DepMap lines (~50 lines), aggregate at pathway level, "
            "or switch to HotNet2-style diffusion. Do not use as sole driver-function head."
        )

    # Table rows
    table_rows = []
    for r in results:
        seed = r["seed_gene"] if r["seed_gene"] else "N/A"
        in_g = "yes" if r["in_graph"] else f"no"
        top10 = "; ".join(r["top_10"][:10]) if r["top_10"] else "—"
        mc_k  = f"{r['mean_chronos_top_k']:.4f}"  if not np.isnan(r['mean_chronos_top_k']) else "—"
        mc_r  = f"{r['mean_chronos_random']:.4f}"  if not np.isnan(r['mean_chronos_random']) else "—"
        pb    = f"{r['p_bonferroni']:.3e}"          if not np.isnan(r['p_bonferroni']) else "—"
        table_rows.append(f"| {r['drug']:<20} | {seed:<8} | {in_g:<8} | {top10} | {mc_k} | {mc_r} | {pb} |")

    hub_rows = []
    for r in results:
        h = r["hub_ranks"]
        tp53 = str(h.get("TP53", "—"))
        myc  = str(h.get("MYC",  "—"))
        akt1 = str(h.get("AKT1", "—"))
        hub_rows.append(f"| {r['drug']:<20} | {tp53:>8} | {myc:>8} | {akt1:>8} |")

    psm_text = "\n".join([f"- {g}: rank {rank}" for g, rank in psm_ranks[:15]]) if psm_ranks else "RWR failed — no data."

    corr_str = corr_df.to_string()

    sig_bonf_list  = ", ".join([r["drug"] for r in sig_bonf])  or "none"
    sig_nom_list   = ", ".join([r["drug"] for r in sig_nominal]) or "none"

    md = f"""# PPI Propagation Pilot — ResistanceMap

## §1 Methods

**Random Walk with Restart (RWR)**

Convergence equation:

    r(t+1) = alpha * W_norm * r(t) + (1 - alpha) * s

Parameters:
- alpha = {ALPHA} (restart probability = {1 - ALPHA})
- s = one-hot seed vector at the drug's primary target gene
- W_norm = column-normalised adjacency matrix, weights = STRING combined_score / 1000
- Convergence: L1(r(t+1) - r(t)) < 1e-6, maximum 100 iterations

**Graph.** STRING v12 human PPI from `data/raw/string_ppi.txt` filtered at
combined_score >= {MIN_CONF}. ENSP IDs mapped to HGNC gene symbols via
`data/raw/string/9606.protein.info.v12.0.txt.gz`.

**Drug seeds.** Primary target gene from `data/raw/drug_metadata.json`.
Cyclophosphamide targets DNA (non-protein) — excluded; propagation is undefined
for non-protein targets.

**DepMap validation.** `data/raw/depmap/CRISPRGeneEffect.csv` Chronos scores,
restricted to 3 confirmed MM cell lines: OPM2 (ACH-000035), KMS34 (ACH-000547),
U266 (ACH-000001). Top-{TOP_K} RWR genes with DepMap coverage were compared to
{N_PERM} random column subsets of the same size drawn from real DepMap data
(no synthetic values). One-sided Wilcoxon rank-sum test (hypothesis: top-k has
lower/more-negative Chronos than random). Bonferroni correction over {n_tests} tests.

---

## §2 Results

### Per-drug table

| Drug                 | Seed     | InGraph  | Top-10 RWR proteins | Mean Chr K | Mean Chr R | p (Bonferroni) |
|:---------------------|:---------|:---------|:--------------------|:----------:|:----------:|:--------------:|
{chr(10).join(table_rows)}

**Mean Chr K** = mean Chronos in top-{TOP_K} RWR genes (3 MM lines). **Mean Chr R** = mean of {N_PERM} random same-size subsets.

Bonferroni-significant (p < 0.05): **{sig_bonf_list}**
Nominal significance only (p < 0.05, uncorrected): **{sig_nom_list}**

---

## §3 Network-bias check

Hub protein ranks in each drug's full RWR score vector (rank 1 = most proximal to seed):

| Drug                 | TP53 rank | MYC rank | AKT1 rank |
|:---------------------|----------:|---------:|----------:|
{chr(10).join(hub_rows)}

If hubs appear in the top-50 across unrelated drug seeds, the RWR is dominated by network
topology rather than drug mechanism. Check whether any hub rank < 50 for unrelated seeds.

---

## §4 Cross-drug specificity (Spearman of RWR rankings)

```
{corr_str}
```

Expected pattern: HDAC inhibitors (Panobinostat, Vorinostat, Romidepsin) should show
high mutual correlation (shared HDAC1 seed). Bortezomib vs Doxorubicin should show
low correlation (proteasome vs topoisomerase II — distinct cellular machineries).

---

## §5 Bortezomib — proteasome subunit ranks

PSMB5-seeded RWR ranks for proteasome complex members (lower rank = more proximal):

{psm_text}

Question A validation: if PSMA/PSMB/PSMC/PSMD subunits cluster in the top-50 and show
lower Chronos than random genes in MM lines, mechanism-specific propagation is confirmed.

---

## §6 Verdict

**{verdict}** — {verdict_note}

**Recommendation:** {recommendation}

### Required changes for v10

1. **Multi-seed RWR**: seed all ChEMBL-curated protein targets per drug, not just the primary.
2. **Expand MM cohort**: include all haematological malignancy DepMap lines (approx. 50) for adequate statistical power.
3. **Pathway-level aggregation**: aggregate Chronos at pathway level (proteasome, HDAC complex, CDK complex) before testing — gene-level test is underpowered at n=3 cell lines.
4. **Alpha sensitivity sweep**: test alpha in [0.5, 0.6, 0.7, 0.8] to characterise hub-bias vs specificity trade-off.
5. **Cyclophosphamide**: non-protein seed — use CYP2B6 (primary bioactivating enzyme) as proxy seed for the DNA-damage arm, or exclude from propagation.
"""

    out_path = docs_dir / "PPI_PROPAGATION_PILOT.md"
    out_path.write_text(md)
    print(f"\nMarkdown written to {out_path}")


if __name__ == "__main__":
    main()
