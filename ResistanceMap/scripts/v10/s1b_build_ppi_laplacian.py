"""S1b: Build PPI Laplacian from STRING v12, filtered for non-leakage with MMRF.

Filter (per docs/PPI_PROPAGATION_DRIVER_THEORY.md §3):
    - combined_score >= 700  (STRING uses 0-1000 scale internally)
    - non-observational channels (experiments + database) sum >= 400
      → removes co-expression and text-mining leakage with MMRF expression.

STRING v12 file fields (whitespace-separated):
    protein1 protein2 neighborhood fusion cooccurence coexpression \
        experiments database textmining combined_score

Output:
    data/processed/ppi_laplacian.npz   (sparse symmetric-normalized L)
    data/processed/ppi_gene_index.tsv  (gene_symbol -> row index)

Maps STRING ENSP -> gene symbol via 9606.protein.info.v12.0.txt.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "string"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

COMBINED_THRESH = 700      # 0.7 × 1000
NONOBS_THRESH = 400        # 0.4 × 1000 (experiments + database)


def load_protein_info() -> dict[str, str]:
    info_path = RAW / "9606.protein.info.v12.0.txt.gz"
    print(f"[S1b] loading {info_path}")
    with gzip.open(info_path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    print(f"[S1b] protein.info rows: {len(df)}; columns: {df.columns.tolist()[:4]}")
    # Column is `#string_protein_id` and `preferred_name` in v12
    pid_col = [c for c in df.columns if "protein_id" in c][0]
    name_col = "preferred_name" if "preferred_name" in df.columns else df.columns[1]
    return dict(zip(df[pid_col].astype(str), df[name_col].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--links-file",
        type=str,
        default="",
        help="Explicit path to STRING links file. If empty, auto-detect (full-channel preferred).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix to append to output filenames (e.g. v10s1, v10s2). Empty → default paths.",
    )
    parser.add_argument(
        "--sprint-dir",
        type=str,
        default="v10_sprint1",
        help="Subdirectory under paper/v8_artifacts for the summary JSON.",
    )
    args = parser.parse_args()

    if args.links_file:
        links_path = Path(args.links_file)
        if not links_path.exists():
            sys.exit(f"explicit --links-file does not exist: {links_path}")
        full_channels = "full" in links_path.name
        print(f"[S1b] explicit links file: {links_path}, full_channels={full_channels}")
    else:
        links_path = RAW / "9606.protein.links.full.v12.0.txt.gz"
        if not links_path.exists():
            plain = RAW / "9606.protein.links.v12.0.txt.gz"
            if plain.exists():
                print(f"[S1b] WARNING — only combined-score links available; cannot apply non-observational filter.")
                print(f"[S1b] using {plain} with combined>={COMBINED_THRESH} only.")
                links_path = plain
                full_channels = False
            else:
                sys.exit("no STRING links file found")
        else:
            full_channels = True

    suffix = f"_{args.tag}" if args.tag else ""

    pid2gene = load_protein_info()

    print(f"[S1b] streaming {links_path} (filter combined>={COMBINED_THRESH})...")
    rows: list[tuple[str, str, int]] = []
    with gzip.open(links_path, "rt") as fh:
        header = fh.readline().split()
        col_idx = {c: i for i, c in enumerate(header)}
        n_combined_pass = 0
        n_total = 0
        for line in fh:
            parts = line.split()
            n_total += 1
            try:
                comb = int(parts[col_idx["combined_score"]])
            except (ValueError, KeyError):
                continue
            if comb < COMBINED_THRESH:
                continue
            n_combined_pass += 1
            if full_channels:
                exp = int(parts[col_idx.get("experiments", -1)])
                db = int(parts[col_idx.get("database", -1)])
                if exp + db < NONOBS_THRESH:
                    continue
            p1 = parts[col_idx["protein1"]]
            p2 = parts[col_idx["protein2"]]
            rows.append((p1, p2, comb))
            if n_total % 5_000_000 == 0:
                print(f"[S1b]   ... {n_total:,} lines, {len(rows):,} kept")
    print(f"[S1b] total lines={n_total:,}, combined>={COMBINED_THRESH}: {n_combined_pass:,}, after non-obs filter: {len(rows):,}")

    # Symbol-map and dedup
    edges_df = pd.DataFrame(rows, columns=["p1", "p2", "score"])
    edges_df["g1"] = edges_df["p1"].map(pid2gene)
    edges_df["g2"] = edges_df["p2"].map(pid2gene)
    edges_df = edges_df.dropna(subset=["g1", "g2"])
    edges_df = edges_df[edges_df["g1"] != edges_df["g2"]]
    print(f"[S1b] edges with gene-symbol both ends: {len(edges_df):,}")

    # Use combined_score / 1000 as edge weight (per Vanunu 2010)
    edges_df["w"] = edges_df["score"].astype(np.float32) / 1000.0

    # Canonicalize symmetric (g1<g2) to dedup
    a = np.minimum(edges_df["g1"].values, edges_df["g2"].values)
    b = np.maximum(edges_df["g1"].values, edges_df["g2"].values)
    canon = pd.DataFrame({"g1": a, "g2": b, "w": edges_df["w"].values})
    canon = canon.groupby(["g1", "g2"], as_index=False)["w"].max()
    print(f"[S1b] canonical undirected edges: {len(canon):,}")

    # Build gene index
    genes = pd.unique(np.concatenate([canon["g1"].values, canon["g2"].values]))
    genes.sort()
    gene2idx = {g: i for i, g in enumerate(genes)}
    print(f"[S1b] node count: {len(genes):,}")

    # Sparse adjacency
    rows_i = np.array([gene2idx[g] for g in canon["g1"]], dtype=np.int32)
    cols_i = np.array([gene2idx[g] for g in canon["g2"]], dtype=np.int32)
    weights = canon["w"].values.astype(np.float32)
    n = len(genes)
    A = sp.coo_matrix((weights, (rows_i, cols_i)), shape=(n, n))
    A = A + A.T  # symmetrize
    A = A.tocsr()

    # Symmetric-normalized Laplacian: L = I - D^-1/2 A D^-1/2
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    L = sp.identity(n, dtype=np.float32) - A_norm
    L = L.tocsr()

    out_npz = PROC / f"ppi_laplacian{suffix}.npz"
    sp.save_npz(out_npz, L)
    print(f"[S1b] wrote {out_npz} ({L.nnz:,} nonzeros)")

    idx_path = PROC / f"ppi_gene_index{suffix}.tsv"
    pd.DataFrame({"gene": genes, "idx": np.arange(n)}).to_csv(idx_path, sep="\t", index=False)
    print(f"[S1b] wrote {idx_path}")

    summary = {
        "n_genes": int(n),
        "n_edges_canonical": int(len(canon)),
        "combined_threshold": COMBINED_THRESH,
        "nonobs_threshold": NONOBS_THRESH if full_channels else None,
        "full_channels_available": full_channels,
        "links_file": str(links_path),
        "tag": args.tag,
        "laplacian_nnz": int(L.nnz),
        "mean_degree": float(deg.mean()),
        "median_degree": float(np.median(deg)),
    }
    summary_path = ROOT / "paper" / "v8_artifacts" / args.sprint_dir / f"s1b_summary{suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S1b] wrote {summary_path}")


if __name__ == "__main__":
    main()
