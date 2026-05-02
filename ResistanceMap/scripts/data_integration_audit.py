#!/usr/bin/env python3
"""v8 data-integration audit.

Compares ResistanceMap's CrossModalFusionNet against four head-to-head
integration baselines on the SAME train/val/test split, on real CCLE
proteomics + epigenomics + GDSC drug-sensitivity data:

  A. Naive concat -> Ridge
  B. PCA(256) per modality, concat -> Ridge
  C. MOFA+-style sparse-PCA shared factors -> Ridge
  D. Late fusion (avg per-modality Ridge predictions)

Plus batch-effect metrics on the latent embedding.

Output: paper/v8_artifacts/data_integration_audit.json + .md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "checkpoints" / "data_ready.pt"
RM_VAL = REPO / "checkpoints" / "pipeline_validated.pt"
OUT_DIR = REPO / "paper" / "v8_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load():
    c = torch.load(DATA, map_location="cpu", weights_only=False)
    ds = c["dataset"]
    splits = c["splits"]
    Xp = ds.proteomics.numpy().astype(np.float32)
    Xe = ds.epigenomics.numpy().astype(np.float32)
    y = ds.drug_sensitivity.numpy().astype(np.float32)
    drug_names = list(ds.drug_names)
    sp = {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}
    return Xp, Xe, y, drug_names, sp


def masked_mse_per_drug(model_factory, Xtr, ytr, Xva, yva, Xte, yte):
    """Train one model per drug on observed cells; pool squared residuals (matches main.py)."""
    n_drugs = ytr.shape[1]
    test_residuals, val_residuals = [], []
    for d in range(n_drugs):
        tr = ~np.isnan(ytr[:, d]); va = ~np.isnan(yva[:, d]); te = ~np.isnan(yte[:, d])
        if tr.sum() < 5: continue
        m = model_factory()
        m.fit(Xtr[tr], ytr[tr, d])
        if va.sum(): val_residuals.append(m.predict(Xva[va]) - yva[va, d])
        if te.sum(): test_residuals.append(m.predict(Xte[te]) - yte[te, d])
    test_mse = float(np.mean(np.concatenate(test_residuals) ** 2)) if test_residuals else float("nan")
    val_mse = float(np.mean(np.concatenate(val_residuals) ** 2)) if val_residuals else float("nan")
    return test_mse, val_mse


def split_arrays(X, y, sp):
    return X[sp["train"]], y[sp["train"]], X[sp["val"]], y[sp["val"]], X[sp["test"]], y[sp["test"]]


def baseline_concat_ridge(Xp, Xe, y, sp):
    X = np.concatenate([Xp, Xe], axis=1)
    Xtr, ytr, Xva, yva, Xte, yte = split_arrays(X, y, sp)
    return masked_mse_per_drug(lambda: Ridge(alpha=10.0), Xtr, ytr, Xva, yva, Xte, yte)


def baseline_pca_per_modality(Xp, Xe, y, sp, k=256):
    pca_p = PCA(n_components=min(k, min(Xp.shape) - 1)).fit(Xp[sp["train"]])
    pca_e = PCA(n_components=min(20, min(Xe.shape) - 1)).fit(Xe[sp["train"]])
    Zp = pca_p.transform(Xp); Ze = pca_e.transform(Xe)
    X = np.concatenate([Zp, Ze], axis=1)
    Xtr, ytr, Xva, yva, Xte, yte = split_arrays(X, y, sp)
    return masked_mse_per_drug(lambda: Ridge(alpha=1.0), Xtr, ytr, Xva, yva, Xte, yte)


def baseline_mofa_like(Xp, Xe, y, sp, k=64):
    """MOFA+-like: shared sparse factors via dictionary learning on concatenated, normalized data.

    True MOFA+ (Argelaguet 2020) does block-wise variational sparse PCA. Without
    `mofapy2`, MiniBatchDictionaryLearning gives a comparable shared-factor model
    that's interpretable and fast. We mark this clearly in the report.
    """
    # Standardize each modality on train
    def z(X, idx):
        mu = X[idx].mean(0); sd = X[idx].std(0) + 1e-6
        return (X - mu) / sd
    Zp = z(Xp, sp["train"]); Ze = z(Xe, sp["train"])
    X = np.concatenate([Zp, Ze], axis=1)
    dl = MiniBatchDictionaryLearning(n_components=k, alpha=0.5, max_iter=50,
                                     batch_size=64, random_state=42).fit(X[sp["train"]])
    F = dl.transform(X)
    Xtr, ytr, Xva, yva, Xte, yte = split_arrays(F, y, sp)
    return masked_mse_per_drug(lambda: Ridge(alpha=1.0), Xtr, ytr, Xva, yva, Xte, yte)


def baseline_late_fusion(Xp, Xe, y, sp):
    """Per-modality Ridge predictions averaged."""
    Xtr_p, ytr, Xva_p, yva, Xte_p, yte = split_arrays(Xp, y, sp)
    Xtr_e, _, Xva_e, _, Xte_e, _ = split_arrays(Xe, y, sp)
    n_drugs = ytr.shape[1]
    test_resid, val_resid = [], []
    for d in range(n_drugs):
        tr = ~np.isnan(ytr[:, d]); va = ~np.isnan(yva[:, d]); te = ~np.isnan(yte[:, d])
        if tr.sum() < 5: continue
        m_p = Ridge(alpha=10.0).fit(Xtr_p[tr], ytr[tr, d])
        m_e = Ridge(alpha=10.0).fit(Xtr_e[tr], ytr[tr, d])
        if va.sum():
            pred = 0.5 * (m_p.predict(Xva_p[va]) + m_e.predict(Xva_e[va]))
            val_resid.append(pred - yva[va, d])
        if te.sum():
            pred = 0.5 * (m_p.predict(Xte_p[te]) + m_e.predict(Xte_e[te]))
            test_resid.append(pred - yte[te, d])
    return (float(np.mean(np.concatenate(test_resid) ** 2)) if test_resid else float("nan"),
            float(np.mean(np.concatenate(val_resid) ** 2)) if val_resid else float("nan"))


def get_resistancemap_test_mse():
    if not RM_VAL.exists(): return None
    c = torch.load(RM_VAL, map_location="cpu", weights_only=False)
    m = c.get("metrics", {}) if isinstance(c, dict) else {}
    return m.get("test_mse")


def batch_effect_metrics(Z, batches, k=15):
    """Simplified kBET, iLISI, silhouette on an embedding Z with a categorical batch label."""
    if len(np.unique(batches)) < 2 or Z.shape[0] < k + 1:
        return {"kBET": None, "iLISI": None, "silhouette": None,
                "note": "insufficient batch diversity or sample size"}
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]  # drop self
    expected = np.bincount(batches) / len(batches)
    chi2_pvalues = []
    inv_lisis = []
    for i in range(len(Z)):
        nbatches = batches[idx[i]]
        observed = np.bincount(nbatches, minlength=len(expected)) / k
        # chi^2-style departure from global frequency
        chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-9))
        chi2_pvalues.append(chi2)
        # iLISI = 1/sum(p_b^2)
        p = observed
        inv_lisis.append(1.0 / max(np.sum(p ** 2), 1e-9))
    sil = float(silhouette_score(Z, batches)) if len(np.unique(batches)) > 1 else float("nan")
    return {"kBET_mean_chi2": float(np.mean(chi2_pvalues)),
            "iLISI_mean": float(np.mean(inv_lisis)),
            "silhouette_by_batch": sil,
            "n_batches": int(len(np.unique(batches))),
            "note": "kBET reported as mean chi^2 (lower=better integrated); iLISI close to n_batches=better"}


def main():
    print("[v8 data-integration audit] starting")
    Xp, Xe, y, drug_names, sp = load()
    print(f"  shapes: Xp={Xp.shape} Xe={Xe.shape} y={y.shape} "
          f"train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")

    results = []
    for name, fn in [
        ("A. Naive concat -> Ridge", lambda: baseline_concat_ridge(Xp, Xe, y, sp)),
        ("B. PCA(256/20) per-modality -> Ridge", lambda: baseline_pca_per_modality(Xp, Xe, y, sp)),
        ("C. MOFA+-like shared factors(64) -> Ridge", lambda: baseline_mofa_like(Xp, Xe, y, sp)),
        ("D. Late fusion (avg)", lambda: baseline_late_fusion(Xp, Xe, y, sp)),
    ]:
        t0 = time.time()
        try:
            test_mse, val_mse = fn()
            results.append({"method": name, "test_mse": test_mse, "val_mse": val_mse,
                            "wall_seconds": round(time.time() - t0, 2)})
            print(f"  {name:50s} test_mse={test_mse:.4f} val={val_mse:.4f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            results.append({"method": name, "test_mse": None, "val_mse": None, "error": str(e)})
            print(f"  {name:50s} FAILED: {e}")

    rm_mse = get_resistancemap_test_mse()
    results.append({"method": "ResistanceMap CrossModalFusionNet (v6)",
                    "test_mse": rm_mse, "val_mse": None,
                    "notes": "from checkpoints/pipeline_validated.pt"})
    print(f"  ResistanceMap CrossModalFusionNet                test_mse={rm_mse}")

    # Batch-effect proxy: split cell lines into "batches" by quartile of mean proteomics
    # signal (a rough proxy for technical variability when explicit batch info is absent).
    Xtr = Xp[sp["train"]]
    pca = PCA(n_components=min(64, min(Xtr.shape) - 1)).fit(Xtr)
    Z_train = pca.transform(Xtr)
    proxy_batch = (np.argsort(Xp[sp["train"]].mean(1)) * 4 // len(sp["train"])).astype(int)
    bm = batch_effect_metrics(Z_train, proxy_batch, k=15)
    print(f"  batch-effect proxy: {bm}")

    out_json = OUT_DIR / "data_integration_audit.json"
    out_md = OUT_DIR / "data_integration_audit.md"

    out_json.write_text(json.dumps({
        "data_sha": "see checkpoints/data_ready.pt",
        "n_train": int(len(sp["train"])), "n_val": int(len(sp["val"])), "n_test": int(len(sp["test"])),
        "n_features_p": int(Xp.shape[1]), "n_features_e": int(Xe.shape[1]),
        "n_drugs": int(y.shape[1]), "drugs": drug_names,
        "results": results,
        "batch_effect_proxy": bm,
    }, indent=2))

    # Sort by test_mse for the markdown table
    ranked = sorted([r for r in results if r.get("test_mse") is not None],
                    key=lambda r: r["test_mse"])
    rows = ["| Rank | Method | test_mse | val_mse | notes |", "|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        val = r.get("val_mse")
        val_str = f"{val:.4f}" if val is not None else "n/a"
        rows.append(f"| {i} | {r['method']} | {r['test_mse']:.4f} | {val_str} | {r.get('notes', '')} |")

    md = f"""# Data Integration Audit — ResistanceMap v6

Auto-generated by `scripts/data_integration_audit.py`. Ground truth read from
`checkpoints/data_ready.pt` and `checkpoints/pipeline_validated.pt`. Splits match
the original training (random seed 42).

## Setup
- Train / val / test: {len(sp['train'])} / {len(sp['val'])} / {len(sp['test'])}
- Features: proteomics={Xp.shape[1]} + epigenomics={Xe.shape[1]}
- Drugs: {y.shape[1]} ({', '.join(drug_names)})
- Evaluation: NaN-masked MSE pooled across observed (sample, drug) cells (matches main.py:1442-1443)

## Head-to-head (lower is better)

{chr(10).join(rows)}

## Batch-effect proxy (PCA(64) on train proteomics, quartile pseudo-batch)
- kBET mean chi² (lower = better integrated): {bm.get('kBET_mean_chi2')}
- iLISI mean (close to n_batches = {bm.get('n_batches')} = better): {bm.get('iLISI_mean')}
- silhouette by pseudo-batch: {bm.get('silhouette_by_batch')}
- Note: {bm.get('note')}

## Verdict (auto-derived; sota-comparator and chair refine)
"""
    if rm_mse is not None and ranked:
        rm_rank = next((i for i, r in enumerate(ranked, 1)
                        if "ResistanceMap" in r["method"]), None)
        beats_concat = next((r for r in ranked if "Naive concat" in r["method"]), None)
        if rm_rank == 1:
            md += "PASS — ResistanceMap's cross-attention fusion outperforms all integration baselines.\n"
        elif beats_concat and rm_mse < beats_concat["test_mse"]:
            md += f"CONDITIONAL — ResistanceMap (rank {rm_rank}) beats naive concat but is dominated by simpler baselines.\n"
        else:
            md += f"FAIL — ResistanceMap (rank {rm_rank}) does not improve over the naive-concat baseline. The cross-attention fusion is not justified by current data.\n"
    else:
        md += "INDETERMINATE — pipeline_validated.pt missing or no ranked baselines.\n"

    md += """
## Comparison to platform baselines (qualitative)

- **MOFA+** (Argelaguet 2020): provides interpretable factors with sparse loadings. ResistanceMap's
  64-d VAE latent has no sparsity prior and no per-modality factor decomposition; interpretability
  gap. Approximation here uses MiniBatchDictionaryLearning.
- **Galaxy**: workflow platform, not algorithm. ResistanceMap's `--config configs/default.yaml`
  reproducibility surface is competitive but lacks Galaxy's tool-versioning lockfile.
- **HiPlot** (Meta): high-dim parallel-coordinates viz; ResistanceMap's UMAP layer in
  `landscape_train` is not directly comparable (single 2-D layout vs HiPlot's interactive PCs).
- **OmicsNet** (Zhou et al.): graph-based integration. ResistanceMap's PPI-GAT on STRING is the
  closest analog; OmicsNet adds metabolite/microbiome layers that ResistanceMap doesn't model.
- **TCGA Pan-Cancer Atlas integrators** (iCluster+, JIVE): pan-cancer, not MM-specific.
  ResistanceMap's hematologic specialization is a positioning advantage but reduces transferability.

## Required changes
1. Add a sparsity prior to the VAE latent (or replace with MOFA+'s `mofapy2`) for interpretability.
2. Persist per-modality factor loadings so pathway-validator can ground them.
3. Document train/val/test split policy more explicitly (cell-line vs patient leakage check).
"""
    out_md.write_text(md)
    print(f"\n[done] {out_json}\n[done] {out_md}")


if __name__ == "__main__":
    main()
