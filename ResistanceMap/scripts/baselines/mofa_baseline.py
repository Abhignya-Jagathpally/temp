#!/usr/bin/env python3
"""MOFA+ baseline against ResistanceMap's deep multi-omics integration.

Probabilistic multi-omics factor analysis with ~20 latent factors over three
views (proteomics, epigenomics, crispr_effect). Trains on the EXACT train split
already saved in checkpoints/data_ready.pt, projects val/test, then fits a
per-drug Ridge regression from the 20 MOFA factors -> drug-response z-score.

Compares per-drug test MSE / Spearman to ResistanceMap (logs/20260503T094942Z).

Output:
    paper/v8_artifacts/baselines/mofa_factors.npy
    paper/v8_artifacts/baselines/mofa_sample_ids.json
    paper/v8_artifacts/baselines/mofa_results.json

NO synthetic data. Uses checkpoints/data_ready.pt splits as-is.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


REPO = Path(__file__).resolve().parents[2]
DATA_CKPT = REPO / "checkpoints" / "data_ready.pt"
OUT_DIR = REPO / "paper" / "v8_artifacts" / "baselines"
OUT_FACTORS = OUT_DIR / "mofa_factors.npy"
OUT_SAMPLE_IDS = OUT_DIR / "mofa_sample_ids.json"
OUT_RESULTS = OUT_DIR / "mofa_results.json"

# ResistanceMap per-drug metrics from logs/20260503T094942Z/stdout.log
# (validate stage, test split, n_test=132 samples, 11 drugs).
RM_PER_DRUG = {
    "Bortezomib":       {"n_obs": 102, "mse": 0.8821372985839844,  "spearman":  0.346953084800199,    "drug_class": "Proteasome"},
    "Cyclophosphamide": {"n_obs":  76, "mse": 0.3054813742637634,  "spearman":  0.19064935064935065,  "drug_class": "DNA-damage"},
    "Dinaciclib":       {"n_obs": 100, "mse": 0.1354023963212967,  "spearman":  0.3158355835583558,   "drug_class": "CDK"},
    "Doxorubicin":      {"n_obs": 100, "mse": 1.9848915338516235,  "spearman":  0.11335133513351334,  "drug_class": "DNA-damage"},
    "Etoposide":        {"n_obs":  93, "mse": 0.358908087015152,   "spearman":  0.16275252902032167,  "drug_class": "DNA-damage"},
    "Lenalidomide":     {"n_obs":  95, "mse": 0.813334584236145,   "spearman": -0.07162653975363942,  "drug_class": "IMiD"},
    "Palbociclib":      {"n_obs":  81, "mse": 0.19919045269489288, "spearman":  0.024570912375790423, "drug_class": "CDK"},
    "Panobinostat":     {"n_obs": 101, "mse": 22.62211799621582,   "spearman":  0.06008153756552125,  "drug_class": "HDAC"},
    "Romidepsin":       {"n_obs":  93, "mse": 1.2600728273391724,  "spearman":  0.2365044313807406,   "drug_class": "HDAC"},
    "Venetoclax":       {"n_obs":  74, "mse": 0.01999061368405819, "spearman":  0.26844872269529796,  "drug_class": "BCL2"},
    "Vorinostat":       {"n_obs": 100, "mse": 0.23433221876621246, "spearman":  0.02125412541254125,  "drug_class": "HDAC"},
}
RM_OVERALL_MSE = 2.8364
RM_RUN_LOG = "logs/20260503T094942Z/stdout.log"

N_FACTORS = 20
TOPK_PROT = 2000
TOPK_CRISPR = 2000
RIDGE_ALPHA = 1.0
SEED = 42


def load_real_data():
    print(f"[load] {DATA_CKPT}")
    c = torch.load(DATA_CKPT, map_location="cpu", weights_only=False)
    ds = c["dataset"]
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in c["splits"].items()}
    proteomics = ds.proteomics.numpy().astype(np.float32)
    epigenomics = ds.epigenomics.numpy().astype(np.float32)
    crispr = ds.crispr_effect.numpy().astype(np.float32)
    drugs = ds.drug_sensitivity.numpy().astype(np.float32)
    sample_ids = list(ds.sample_ids)
    drug_names = list(ds.drug_names)
    return proteomics, epigenomics, crispr, drugs, sample_ids, drug_names, splits


def variance_topk(M_train: np.ndarray, k: int):
    """Indices of top-k features by NaN-safe variance computed on TRAIN only."""
    if M_train.shape[1] <= k:
        return np.arange(M_train.shape[1])
    var = np.nanvar(M_train, axis=0)
    var[~np.isfinite(var)] = -1.0
    return np.argpartition(-var, k)[:k]


def standardize_train_apply(M_train, M_full, fill_nan_with: str = "skip"):
    """Standardize using train mean/std (NaN-safe). Returns full matrix.

    fill_nan_with='skip' keeps NaNs (MOFA handles missingness natively).
    """
    mu = np.nanmean(M_train, axis=0)
    sd = np.nanstd(M_train, axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (M_full - mu) / sd
    if fill_nan_with == "zero":
        Z = np.nan_to_num(Z, nan=0.0)
    return Z.astype(np.float32)


def train_mofa(views_train, view_names, factors=N_FACTORS, seed=SEED):
    """Train MOFA+ on train rows only."""
    from mofapy2.run.entry_point import entry_point

    data = [[v] for v in views_train]  # one group per view

    ent = entry_point()
    ent.set_data_options(scale_views=False, scale_groups=False, center_groups=True, use_float32=True)
    ent.set_data_matrix(
        data,
        likelihoods=["gaussian"] * len(views_train),
        views_names=view_names,
        groups_names=["all"],
    )
    ent.set_model_options(factors=factors, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(
        iter=500, convergence_mode="medium", seed=seed, verbose=False, quiet=True
    )
    t0 = time.time()
    ent.build()
    ent.run()
    fit_time = time.time() - t0
    Z_train = ent.model.getExpectations()["Z"]["E"]
    W = [ent.model.getExpectations()["W"][m]["E"] for m in range(len(views_train))]
    return ent, np.asarray(Z_train, dtype=np.float32), W, fit_time


def project_factors(W_list, views_full_std):
    """Project new samples into trained factor space via ridge LS on stacked W."""
    K = W_list[0].shape[1]
    n_samples = views_full_std[0].shape[0]
    Z = np.zeros((n_samples, K), dtype=np.float32)
    W_full = np.vstack(W_list).astype(np.float32)
    lam = 1e-2
    eye_K = np.eye(K, dtype=np.float32)
    for i in range(n_samples):
        x = np.concatenate([views_full_std[v][i] for v in range(len(views_full_std))])
        mask = np.isfinite(x)
        if mask.sum() < K:
            continue
        Wm = W_full[mask]
        xm = x[mask]
        A = Wm.T @ Wm + lam * eye_K
        b = Wm.T @ xm
        try:
            Z[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            Z[i] = np.linalg.lstsq(Wm, xm, rcond=None)[0]
    return Z


def per_drug_ridge(Z_tr, y_tr, Z_va, y_va, Z_te, y_te, drug_names, alpha=RIDGE_ALPHA):
    rows = []
    pooled_te_resid = []
    pooled_va_resid = []
    for d, name in enumerate(drug_names):
        tr_mask = ~np.isnan(y_tr[:, d])
        va_mask = ~np.isnan(y_va[:, d])
        te_mask = ~np.isnan(y_te[:, d])
        if tr_mask.sum() < 5 or te_mask.sum() < 2:
            rows.append({"drug": name, "n_train": int(tr_mask.sum()),
                         "n_test": int(te_mask.sum()), "test_mse": None,
                         "test_spearman": None})
            continue
        m = Ridge(alpha=alpha, random_state=SEED)
        m.fit(Z_tr[tr_mask], y_tr[tr_mask, d])
        pred_te = m.predict(Z_te[te_mask])
        resid_te = pred_te - y_te[te_mask, d]
        test_mse = float(np.mean(resid_te ** 2))
        if te_mask.sum() >= 3 and np.std(pred_te) > 0 and np.std(y_te[te_mask, d]) > 0:
            sp, _ = spearmanr(pred_te, y_te[te_mask, d])
            test_spearman = float(sp) if np.isfinite(sp) else None
        else:
            test_spearman = None

        if va_mask.sum() >= 2:
            pred_va = m.predict(Z_va[va_mask])
            resid_va = pred_va - y_va[va_mask, d]
            val_mse = float(np.mean(resid_va ** 2))
            pooled_va_resid.append(resid_va)
        else:
            val_mse = None

        pooled_te_resid.append(resid_te)
        rows.append({
            "drug": name,
            "n_train": int(tr_mask.sum()),
            "n_val": int(va_mask.sum()),
            "n_test": int(te_mask.sum()),
            "test_mse": test_mse,
            "val_mse": val_mse,
            "test_spearman": test_spearman,
        })
    overall_test_mse = float(np.mean(np.concatenate(pooled_te_resid) ** 2)) if pooled_te_resid else None
    overall_val_mse = float(np.mean(np.concatenate(pooled_va_resid) ** 2)) if pooled_va_resid else None
    return rows, overall_test_mse, overall_val_mse


def compare_to_resistancemap(per_drug_rows):
    out = []
    n_beat = n_tie = n_lose = 0
    for row in per_drug_rows:
        d = row["drug"]
        rm = RM_PER_DRUG.get(d)
        if rm is None or row["test_mse"] is None:
            out.append({"drug": d, "verdict": "n/a"})
            continue
        rm_mse = rm["mse"]
        ratio = row["test_mse"] / rm_mse if rm_mse > 0 else float("inf")
        if ratio < 0.95:
            v = "beat"; n_beat += 1
        elif ratio > 1.05:
            v = "lose"; n_lose += 1
        else:
            v = "tie"; n_tie += 1
        out.append({
            "drug": d,
            "drug_class": rm["drug_class"],
            "mofa_test_mse": row["test_mse"],
            "rm_test_mse": rm_mse,
            "ratio_mofa_over_rm": ratio,
            "mofa_test_spearman": row["test_spearman"],
            "rm_test_spearman": rm["spearman"],
            "n_test": row["n_test"],
            "verdict": v,
        })
    return out, {"n_beat": n_beat, "n_tie": n_tie, "n_lose": n_lose}


def build_verdict(overall_test_mse, summary):
    n_beat = summary["n_beat"]; n_tie = summary["n_tie"]; n_lose = summary["n_lose"]
    if overall_test_mse is None:
        return "MOFA+Ridge produced no usable predictions on the test split (likely a fit failure)."
    delta = overall_test_mse - RM_OVERALL_MSE
    if overall_test_mse <= RM_OVERALL_MSE * 0.95:
        stance = "MOFA+Ridge BEATS ResistanceMap overall"
    elif overall_test_mse >= RM_OVERALL_MSE * 1.05:
        stance = "MOFA+Ridge LOSES to ResistanceMap overall"
    else:
        stance = "MOFA+Ridge TIES ResistanceMap overall"
    return (
        f"{stance}: MOFA+Ridge pooled test_mse={overall_test_mse:.4f} vs "
        f"ResistanceMap test_mse={RM_OVERALL_MSE:.4f} (delta={delta:+.4f}). "
        f"Per-drug breakdown: MOFA beats RM on {n_beat}/{n_beat+n_tie+n_lose} drugs, "
        f"ties on {n_tie}, loses on {n_lose}. "
        "ResistanceMap's reported test_mse (2.84) is dominated by a single Panobinostat "
        "outlier (mse=22.6) and is essentially the zero-prediction floor; MOFA's 20 "
        "probabilistic factors offer a smoothed, NaN-tolerant integrated representation "
        "that should expose whether any signal beyond the floor exists. If the per-drug "
        "verdicts are mostly 'tie' and the pooled MSE is similar, multi-omics integration "
        "is not the bottleneck on this 886-cell-line / 11-drug task -- sample size and "
        "label sparsity are."
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    proteomics, epigenomics, crispr, drugs, sample_ids, drug_names, splits = load_real_data()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    n = proteomics.shape[0]
    print(f"[data] n={n}  train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"[data] proteomics={proteomics.shape}  epigenomics={epigenomics.shape}  crispr={crispr.shape}")
    print(f"[data] drug_sensitivity={drugs.shape}  drug NaN frac={np.isnan(drugs).mean():.3f}")
    print(f"[data] crispr NaN frac={np.isnan(crispr).mean():.3f}")

    prot_top = variance_topk(proteomics[tr], TOPK_PROT)
    crispr_top = variance_topk(crispr[tr], TOPK_CRISPR)
    print(f"[filter] proteomics: {proteomics.shape[1]} -> {len(prot_top)} (top-var)")
    print(f"[filter] crispr_effect: {crispr.shape[1]} -> {len(crispr_top)} (top-var, NaN-safe)")

    P_full = proteomics[:, prot_top]
    E_full = epigenomics
    C_full = crispr[:, crispr_top]

    P_full_std = standardize_train_apply(P_full[tr], P_full, fill_nan_with="skip")
    E_full_std = standardize_train_apply(E_full[tr], E_full, fill_nan_with="skip")
    C_full_std = standardize_train_apply(C_full[tr], C_full, fill_nan_with="skip")

    print(f"[mofa] training factors={N_FACTORS} on train (n={len(tr)})...")
    ent, Z_train, W_list, fit_time = train_mofa(
        [P_full_std[tr], E_full_std[tr], C_full_std[tr]],
        view_names=["proteomics", "epigenomics", "crispr_effect"],
        factors=N_FACTORS,
        seed=SEED,
    )
    print(f"[mofa] trained in {fit_time:.1f}s, Z_train shape={Z_train.shape}")
    print(f"[mofa] view weight shapes: {[w.shape for w in W_list]}")

    Z_full = project_factors(W_list, [P_full_std, E_full_std, C_full_std])
    Z_full[tr] = Z_train  # use inferred train Z (more accurate than projection)
    print(f"[mofa] full factor matrix: {Z_full.shape}")

    np.save(OUT_FACTORS, Z_full)
    OUT_SAMPLE_IDS.write_text(json.dumps({
        "sample_ids": sample_ids,
        "splits": {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
        "n_factors": N_FACTORS,
        "view_feature_indices": {
            "proteomics_topvar_idx": prot_top.tolist(),
            "crispr_topvar_idx": crispr_top.tolist(),
        },
    }, indent=2))
    print(f"[save] {OUT_FACTORS}")
    print(f"[save] {OUT_SAMPLE_IDS}")

    Z_tr = Z_full[tr]; Z_va = Z_full[va]; Z_te = Z_full[te]
    y_tr = drugs[tr]; y_va = drugs[va]; y_te = drugs[te]
    per_drug, overall_te_mse, overall_va_mse = per_drug_ridge(
        Z_tr, y_tr, Z_va, y_va, Z_te, y_te, drug_names
    )
    print(f"[ridge] overall pooled test_mse={overall_te_mse:.4f}  val_mse={overall_va_mse:.4f}")

    cmp_rows, cmp_summary = compare_to_resistancemap(per_drug)
    print(f"[compare] beat={cmp_summary['n_beat']}  tie={cmp_summary['n_tie']}  lose={cmp_summary['n_lose']}")
    for r in cmp_rows:
        if r["verdict"] == "n/a":
            print(f"  {r['drug']:18s} n/a")
            continue
        print(f"  {r['drug']:18s} mofa_mse={r['mofa_test_mse']:.4f}  rm_mse={r['rm_test_mse']:.4f}  "
              f"ratio={r['ratio_mofa_over_rm']:.2f}  -> {r['verdict']}")

    verdict = build_verdict(overall_te_mse, cmp_summary)

    out = {
        "verdict": verdict,
        "config": {
            "n_factors": N_FACTORS,
            "topk_proteomics": TOPK_PROT,
            "topk_crispr": TOPK_CRISPR,
            "ridge_alpha": RIDGE_ALPHA,
            "seed": SEED,
            "views": ["proteomics", "epigenomics", "crispr_effect"],
            "fit_seconds": round(fit_time, 1),
        },
        "data_ready_path": str(DATA_CKPT.relative_to(REPO)),
        "splits": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "view_dims_after_filter": {
            "proteomics": int(P_full.shape[1]),
            "epigenomics": int(E_full.shape[1]),
            "crispr_effect": int(C_full.shape[1]),
        },
        "per_drug": per_drug,
        "overall_pooled_test_mse": overall_te_mse,
        "overall_pooled_val_mse": overall_va_mse,
        "comparison_to_resistancemap": {
            "rm_overall_test_mse": RM_OVERALL_MSE,
            "rm_run_log": RM_RUN_LOG,
            "summary": cmp_summary,
            "per_drug": cmp_rows,
        },
    }
    OUT_RESULTS.write_text(json.dumps(out, indent=2))
    print(f"[save] {OUT_RESULTS}")


if __name__ == "__main__":
    main()
