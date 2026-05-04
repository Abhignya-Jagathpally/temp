"""S1d: Train scalar Waddington potential U_θ via denoising score matching.

Loss = L_DSM(σ) + λ_lat · ||∇_z U||² + λ_ppi · ⟨∇_z U, L_PPI ∇_z U⟩_gene

The PPI Tikhonov term operates on the *gene-space projection* of the latent
gradient via the (frozen) PCA decoder. Only genes that are simultaneously in
both the MMRF top-5000 and the STRING gene index participate.

Inputs:
    data/processed/mmrf_z64.npy
    data/processed/mmrf_z64_sample_ids.json
    data/processed/mmrf_z64_pca.npz                # PCA decoder back to standardized log1p-TPM
    data/processed/mmrf_gene_features.tsv          # Ensembl IDs in top-5000 order
    data/processed/ppi_laplacian.npz               # 16,201 × 16,201 sparse, HGNC indexed
    data/processed/ppi_gene_index.tsv              # HGNC → row-idx
    data/raw/mmrf_commpass/rna/<one>.tsv           # used to build ENSG -> HGNC

Output:
    checkpoints/u_theta_v10s1.pt
    paper/v8_artifacts/v10_sprint1/s1d_train_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
CHK = ROOT / "checkpoints"
CHK.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential,
    ScalarPotentialConfig,
    dsm_loss,
)


def load_latents() -> tuple[torch.Tensor, list[str]]:
    Z = np.load(PROC / "mmrf_z64.npy")
    ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    return torch.from_numpy(Z), ids


def load_pca_decoder() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (components, mean, gene_index_into_full_expr_columns)."""
    pca = np.load(PROC / "mmrf_z64_pca.npz")
    return pca["components"], pca["mean"], pca["gene_index"]


def build_ensg_to_symbol() -> dict[str, str]:
    """Read one per-aliquot RNA-seq TSV to harvest gene_id → gene_name mapping."""
    rna_dir = RAW / "rna"
    sample = next(rna_dir.glob("*.rna_seq.augmented_star_gene_counts.tsv"), None)
    if sample is None:
        return {}
    print(f"[S1d] harvesting ENSG→symbol from {sample.name}")
    df = pd.read_csv(sample, sep="\t", comment="#", skiprows=1)
    df = df.dropna(subset=["gene_id", "gene_name"])
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
    df["gene_id_unversioned"] = df["gene_id"].str.split(".").str[0]
    return dict(zip(df["gene_id_unversioned"], df["gene_name"]))


def build_ppi_decoder_pieces(
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    top_ensg: list[str],
    ensg_to_sym: dict[str, str],
    ppi_gene_idx: dict[str, int],
    L_ppi: sp.csr_matrix,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Build the linear gene-projector and the L_PPI submatrix for matched genes."""
    n_top = len(top_ensg)
    matched_rows: list[int] = []
    matched_ppi_idx: list[int] = []
    matched_symbols: list[str] = []
    for i, ensg in enumerate(top_ensg):
        sym = ensg_to_sym.get(ensg)
        if sym is None:
            continue
        ppi_i = ppi_gene_idx.get(sym)
        if ppi_i is None:
            continue
        matched_rows.append(i)
        matched_ppi_idx.append(ppi_i)
        matched_symbols.append(sym)
    print(f"[S1d] PPI matching: {len(matched_symbols)}/{n_top} top-genes mapped to STRING nodes")

    W_full = pca_components.T  # (5000, 64)
    W_match = W_full[matched_rows, :]
    W_t = torch.from_numpy(W_match.astype(np.float32))

    L_sub = L_ppi[matched_ppi_idx, :][:, matched_ppi_idx].tocoo()
    indices = torch.from_numpy(np.vstack([L_sub.row, L_sub.col]).astype(np.int64))
    values = torch.from_numpy(L_sub.data.astype(np.float32))
    print(f"[S1d] sub-Laplacian: {L_sub.shape[0]}×{L_sub.shape[1]}, nnz={L_sub.nnz}")
    return W_t, indices, values, matched_symbols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-latgrad", type=float, default=1e-4)
    parser.add_argument("--lambda-ppi", type=float, default=1e-3)
    parser.add_argument("--no-ppi", action="store_true", help="Disable PPI Tikhonov (sanity baseline)")
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", type=str, default="v10s1",
        help="Sprint tag for output naming. Reads ppi_laplacian{laplacian_suffix}.npz, writes u_theta_{tag}.pt.")
    parser.add_argument("--laplacian-suffix", type=str, default="",
        help="Suffix on the input PPI Laplacian filename (e.g. _v10s2 reads ppi_laplacian_v10s2.npz).")
    parser.add_argument("--sprint-dir", type=str, default="v10_sprint1",
        help="Subdir under paper/v8_artifacts/ for the train log JSON.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[S1d] device={device}, epochs={args.epochs}, batch_size={args.batch_size}")

    Z, ids = load_latents()
    Z = Z.to(device)
    print(f"[S1d] latents Z: {tuple(Z.shape)} on {Z.device}")

    pca_components, pca_mean, gene_index_full = load_pca_decoder()
    feat_df = pd.read_csv(PROC / "mmrf_gene_features.tsv", sep="\t")
    top_ensg = [g.split(".")[0] for g in feat_df["ensembl_id"].tolist()]
    print(f"[S1d] top-5000 ENSG ids loaded ({len(top_ensg)})")

    use_ppi = (not args.no_ppi)
    ppi_pieces = None
    lsuf = args.laplacian_suffix
    if use_ppi:
        ensg_to_sym = build_ensg_to_symbol()
        idx_path = PROC / f"ppi_gene_index{lsuf}.tsv"
        lap_path = PROC / f"ppi_laplacian{lsuf}.npz"
        if not idx_path.exists() or not lap_path.exists():
            sys.exit(f"missing PPI artifacts: {idx_path} / {lap_path}; run s1b with matching --tag first")
        print(f"[S1d] reading PPI: {lap_path}")
        idx_df = pd.read_csv(idx_path, sep="\t")
        ppi_gene_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
        L_ppi = sp.load_npz(lap_path).tocsr()
        W_t, L_idx, L_val, matched_symbols = build_ppi_decoder_pieces(
            pca_components, pca_mean, top_ensg, ensg_to_sym, ppi_gene_idx, L_ppi
        )
        W_t = W_t.to(device)
        L_idx = L_idx.to(device)
        L_val = L_val.to(device)
        n_genes_matched = W_t.shape[0]
        ppi_pieces = (W_t, L_idx, L_val, n_genes_matched)
    else:
        print("[S1d] PPI Tikhonov disabled by --no-ppi")
        n_genes_matched = 0

    cfg = ScalarPotentialConfig(
        latent_dim=Z.shape[1],
        hidden_dims=(256, 256, 128),
        gene_proj_dim=0,
    )
    model = ScalarPotential(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[S1d] model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = Z.shape[0]
    history = []
    sigma_low = torch.tensor(args.sigma_min, device=device)
    sigma_high = torch.tensor(args.sigma_max, device=device)

    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=rng, device=device)
        ep_dsm, ep_lat, ep_ppi = 0.0, 0.0, 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            zb = Z[idx]
            sigma = float(sigma_low + (sigma_high - sigma_low) * torch.rand((), generator=rng, device=device))
            l_dsm = dsm_loss(model, zb, sigma)

            grad_z = model.grad_z(zb)
            l_lat = (grad_z.pow(2).sum(dim=-1)).mean()

            if ppi_pieces is not None:
                W_t, L_idx, L_val, n_g = ppi_pieces
                g_gene = grad_z @ W_t.T
                L_sparse = torch.sparse_coo_tensor(L_idx, L_val, size=(n_g, n_g))
                Lg = torch.sparse.mm(L_sparse, g_gene.T).T
                l_ppi = (g_gene * Lg).sum(dim=-1).mean()
            else:
                l_ppi = torch.zeros((), device=device)

            loss = l_dsm + args.lambda_latgrad * l_lat + args.lambda_ppi * l_ppi
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ep_dsm += float(l_dsm.detach())
            ep_lat += float(l_lat.detach())
            ep_ppi += float(l_ppi.detach())
            n_batches += 1
        sched.step()

        if (ep + 1) % 20 == 0 or ep == 0:
            print(
                f"[S1d] ep={ep+1:4d}  dsm={ep_dsm/n_batches:.4f}  "
                f"latgrad={ep_lat/n_batches:.4f}  ppi={ep_ppi/n_batches:.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )
        history.append({
            "epoch": ep + 1,
            "dsm": ep_dsm / n_batches,
            "latgrad": ep_lat / n_batches,
            "ppi": ep_ppi / n_batches,
            "lr": sched.get_last_lr()[0],
        })

    ckpt_path = CHK / f"u_theta_{args.tag}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "args": vars(args),
        "n_genes_matched": n_genes_matched,
    }, ckpt_path)
    print(f"[S1d] saved checkpoint → {ckpt_path}")

    log_dir = ROOT / "paper" / "v8_artifacts" / args.sprint_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"s1d_train_log_{args.tag}.json"
    log_path.write_text(json.dumps({"history": history, "args": vars(args)}, indent=2))
    print(f"[S1d] wrote {log_path}")


if __name__ == "__main__":
    main()
