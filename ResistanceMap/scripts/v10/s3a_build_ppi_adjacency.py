"""S3a: Build PPI weighted adjacency from STRING for RWR propagation.

s1b saves the symmetric-normalized Laplacian L_sym = I - D^(-1/2) A D^(-1/2).
RWR needs the column-normalized adjacency P = A D^(-1) (column-stochastic).

This script reads the same STRING file s1b used (with the same combined+non-obs
filter) and writes the un-normalized weighted adjacency A (sparse symmetric)
plus its per-node degree, so downstream RWR can build P = A · diag(1/deg) on
demand without re-reading 13M lines of STRING.

Outputs:
    data/processed/ppi_adjacency_v10s2.npz   # sparse symmetric A
    data/processed/ppi_degree_v10s2.npy      # node degree vector
    paper/v8_artifacts/v10_sprint3/s3a_summary.json
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
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"
SPRINT3.mkdir(parents=True, exist_ok=True)

COMBINED_THRESH = 700
NONOBS_THRESH = 400


def load_protein_info() -> dict[str, str]:
    info_path = RAW / "9606.protein.info.v12.0.txt.gz"
    with gzip.open(info_path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    pid_col = [c for c in df.columns if "protein_id" in c][0]
    return dict(zip(df[pid_col].astype(str), df["preferred_name"].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links-file", type=str,
        default=str(RAW / "9606.protein.links.full.v12.0.txt.gz"))
    parser.add_argument("--gene-index", type=str,
        default=str(PROC / "ppi_gene_index_v10s2.tsv"))
    parser.add_argument("--tag", type=str, default="v10s2")
    args = parser.parse_args()

    links_path = Path(args.links_file)
    full_channels = "full" in links_path.name
    print(f"[S3a] links: {links_path} (full_channels={full_channels})")

    pid2gene = load_protein_info()

    # Load the existing gene index from s1b's output so adjacency is row-aligned
    idx_df = pd.read_csv(args.gene_index, sep="\t")
    gene2idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    n = len(gene2idx)
    print(f"[S3a] aligned gene index: {n} nodes (from {args.gene_index})")

    rows: list[tuple[int, int, float]] = []
    n_total = 0
    n_combined_pass = 0
    n_kept = 0
    with gzip.open(links_path, "rt") as fh:
        header = fh.readline().split()
        col_idx = {c: i for i, c in enumerate(header)}
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
            g1 = pid2gene.get(p1)
            g2 = pid2gene.get(p2)
            if g1 is None or g2 is None or g1 == g2:
                continue
            i = gene2idx.get(g1)
            j = gene2idx.get(g2)
            if i is None or j is None:
                continue
            # Canonicalize i < j
            if i > j:
                i, j = j, i
            rows.append((i, j, comb / 1000.0))
            n_kept += 1
            if n_total % 5_000_000 == 0:
                print(f"[S3a]   ... {n_total:,} lines, kept {n_kept:,}")

    df = pd.DataFrame(rows, columns=["i", "j", "w"])
    # Dedup with max score (matches s1b convention)
    df = df.groupby(["i", "j"], as_index=False)["w"].max()
    print(f"[S3a] canonical undirected edges: {len(df):,}")

    weights = df["w"].values.astype(np.float32)
    i_arr = df["i"].values.astype(np.int32)
    j_arr = df["j"].values.astype(np.int32)
    A = sp.coo_matrix((weights, (i_arr, j_arr)), shape=(n, n))
    A = A + A.T  # symmetrize
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    print(f"[S3a] adjacency: {A.shape} nnz={A.nnz}, mean_degree={deg.mean():.2f}")

    out_npz = PROC / f"ppi_adjacency_{args.tag}.npz"
    sp.save_npz(out_npz, A)
    print(f"[S3a] wrote {out_npz}")

    deg_path = PROC / f"ppi_degree_{args.tag}.npy"
    np.save(deg_path, deg)
    print(f"[S3a] wrote {deg_path}")

    summary = {
        "tag": args.tag,
        "n_nodes": int(n),
        "n_edges_canonical": int(len(df)),
        "adjacency_nnz": int(A.nnz),
        "mean_degree": float(deg.mean()),
        "median_degree": float(np.median(deg)),
        "min_degree": float(deg.min()),
        "max_degree": float(deg.max()),
        "links_file": str(links_path),
        "filter_combined_threshold": COMBINED_THRESH,
        "filter_nonobs_threshold": NONOBS_THRESH if full_channels else None,
    }
    summary_path = SPRINT3 / "s3a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S3a] wrote {summary_path}")


if __name__ == "__main__":
    main()
