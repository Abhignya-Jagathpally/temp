#!/usr/bin/env python3
"""
scripts/mortfm_train_beataml_drug_conditioned.py
==================================================
v17.6 — Block B with the new drug-conditioned head.

For each (specimen, drug) pair we predict AUC directly, conditioned on:
  * z_patient — Block-A-aligned 2000-gene encoder output
  * drug_idx — vocabulary index from the BeatAML drug list
  * family_idx — MM ontology drug class index (proteasome_inhibitor,
                  IMiD, BCL2_inhibitor, etc.; outside_mm_ontology=0)

Two variants run side-by-side on the same patient-disjoint split:
  scratch — random encoder + random drug head
  init_a  — Block A encoder weights + random drug head

Outputs:
  * checkpoints/mortfm/beataml_v17_{scratch,init_a}.pt
  * logs/mortfm/beataml_v17_{scratch,init_a}_summary.json
  * logs/mortfm/beataml_v17_per_drug_{scratch,init_a}.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import random as _random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.drug_ontology import lookup_entry_loose
from resistancemap.mortfm.drug_response import (
    DrugConditionedSpecimenResponseHead, PerDrugPairSampler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_train_beataml_drug_conditioned")


class V17BlockBEncoder(nn.Module):
    """Compact encoder: 2000 aligned genes -> 64-d latent."""

    def __init__(self, n_genes: int = 2000, d_latent: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _drug_family_of(name: str) -> str:
    e = lookup_entry_loose(str(name))
    return e.drug_class if e is not None else "outside_mm_ontology"


def _build_pairs(
    expr_path: str = "data/processed/beataml/beataml_expression_block_a_features.parquet",
    drug_path: str = "data/processed/beataml/beataml_drug_response.parquet",
    clin_path: str = "data/processed/beataml/beataml_clinical.csv",
):
    expr = pd.read_parquet(expr_path)
    drug = pd.read_parquet(drug_path)
    drug = drug[drug["response_metric"] == "AUC"].dropna(subset=["response_value"]).copy()
    clin = pd.read_csv(clin_path)
    pid_map = dict(zip(clin["lab_id"].astype(str), clin["patient_id"].astype(str)))

    overlap = sorted(set(expr.index.astype(str)) & set(drug["lab_id"].astype(str)))
    drug = drug[drug["lab_id"].astype(str).isin(overlap)].copy()
    drug["patient_id"] = drug["lab_id"].astype(str).map(pid_map).fillna(drug["lab_id"])
    drug["drug_family"] = drug["drug_name"].astype(str).map(_drug_family_of)
    # Drug + family vocabularies
    drug_vocab = {d: i + 1 for i, d in enumerate(sorted(drug["drug_name"].unique()))}
    family_vocab = {f: i + 1 for i, f in enumerate(sorted(drug["drug_family"].unique()))}
    drug["drug_idx"] = drug["drug_name"].map(drug_vocab)
    drug["family_idx"] = drug["drug_family"].map(family_vocab)
    return expr.loc[overlap], drug, drug_vocab, family_vocab


def _split_patient_disjoint(patients: list[str], seed: int = 13, test_frac: float = 0.20):
    rng = _random.Random(seed)
    p = patients[:]
    rng.shuffle(p)
    n_test = max(1, int(len(p) * test_frac))
    return set(p[n_test:]), set(p[:n_test])


def _ports_block_a(encoder: V17BlockBEncoder, block_a_ckpt: Path,
                    device: torch.device) -> dict:
    payload = torch.load(str(block_a_ckpt), map_location=device, weights_only=False)
    src = payload["model_state_dict"]
    tgt = encoder.state_dict()
    matched: list[str] = []
    for k, v in src.items():
        # Only port the front-end linear weights that match the v17 encoder's shapes.
        for ek in tgt:
            if tgt[ek].shape == v.shape:
                if "encoder" in ek and ek not in matched:
                    tgt[ek] = v.clone()
                    matched.append(ek)
                    break
    encoder.load_state_dict(tgt)
    return {"n_block_a_params_ported": len(matched)}


def _train_one_variant(
    *,
    variant: str,
    block_a_ckpt: Optional[Path],
    pairs: pd.DataFrame,
    expr: pd.DataFrame,
    train_pats: set[str],
    test_pats: set[str],
    n_drugs: int,
    n_families: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    feature_names = list(expr.columns.astype(str))
    n_genes = expr.shape[1]
    encoder = V17BlockBEncoder(n_genes=n_genes, d_latent=64).to(device)
    if block_a_ckpt is not None and Path(block_a_ckpt).exists():
        port_report = _ports_block_a(encoder, Path(block_a_ckpt), device)
        logger.info("[%s] ported %d Block-A params into encoder",
                    variant, port_report["n_block_a_params_ported"])
    else:
        port_report = {"n_block_a_params_ported": 0}

    head = DrugConditionedSpecimenResponseHead(
        d_latent=64, n_drugs=n_drugs, n_drug_families=n_families,
    ).to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    # Build train and test pair tables
    train_pairs = pairs[pairs["patient_id"].isin(train_pats)].reset_index(drop=True)
    test_pairs = pairs[pairs["patient_id"].isin(test_pats)].reset_index(drop=True)
    logger.info("[%s] train_pairs=%d test_pairs=%d (%d train pat / %d test pat)",
                variant, len(train_pairs), len(test_pairs),
                len(train_pats), len(test_pats))

    expr_tensor = torch.tensor(expr.values.astype(np.float32), device=device)
    lab_to_row = {lab: i for i, lab in enumerate(expr.index.astype(str))}
    train_pairs["lab_row"] = train_pairs["lab_id"].astype(str).map(lab_to_row)
    test_pairs["lab_row"] = test_pairs["lab_id"].astype(str).map(lab_to_row)
    train_pairs = train_pairs.dropna(subset=["lab_row"]).reset_index(drop=True)
    test_pairs = test_pairs.dropna(subset=["lab_row"]).reset_index(drop=True)

    train_y = torch.tensor(train_pairs["response_value"].values.astype(np.float32),
                            device=device)
    train_lab = torch.tensor(train_pairs["lab_row"].values.astype(np.int64),
                              device=device)
    train_drug = torch.tensor(train_pairs["drug_idx"].values.astype(np.int64),
                               device=device)
    train_fam = torch.tensor(train_pairs["family_idx"].values.astype(np.int64),
                              device=device)

    sampler = PerDrugPairSampler(
        train_pairs["family_idx"].tolist(), batch_size=batch_size, seed=53,
    )
    history = []
    for ep in range(epochs):
        encoder.train(); head.train()
        ep_loss = 0.0; nb = 0
        for batch_idx in sampler:
            idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            z = encoder(expr_tensor[train_lab[idx]])
            pred = head(z, train_drug[idx], train_fam[idx])
            loss = F.mse_loss(pred, train_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        if (ep + 1) % 10 == 0:
            logger.info("[%s] epoch %d/%d: loss=%.4f",
                        variant, ep + 1, epochs, ep_loss / max(nb, 1))
        history.append(ep_loss / max(nb, 1))

    # ---- Per-drug eval on test split -----------------------------------
    encoder.eval(); head.eval()
    test_y = test_pairs["response_value"].values.astype(np.float32)
    test_lab = torch.tensor(test_pairs["lab_row"].values.astype(np.int64),
                             device=device)
    test_drug = torch.tensor(test_pairs["drug_idx"].values.astype(np.int64),
                              device=device)
    test_fam = torch.tensor(test_pairs["family_idx"].values.astype(np.int64),
                             device=device)
    with torch.no_grad():
        z_test = encoder(expr_tensor[test_lab])
        pred_test = head(z_test, test_drug, test_fam).cpu().numpy()
    test_pairs["prediction"] = pred_test
    per_drug = []
    for drug_name, sub in test_pairs.groupby("drug_name"):
        if len(sub) < 5:
            continue
        try:
            rho, p = spearmanr(sub["prediction"].values, sub["response_value"].values)
        except Exception:
            rho, p = float("nan"), float("nan")
        per_drug.append({
            "drug_name": drug_name,
            "drug_family": sub["drug_family"].iloc[0],
            "n_test": int(len(sub)),
            "spearman": float(rho),
            "spearman_p": float(p),
            "mse": float(((sub["prediction"] - sub["response_value"]) ** 2).mean()),
        })
    per_drug_df = pd.DataFrame(per_drug).sort_values("n_test", ascending=False)
    return {
        "variant": variant,
        "history": history,
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "per_drug_df": per_drug_df,
        "port_report": port_report,
    }


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--block-a-checkpoint",
                    default="checkpoints/mortfm/block_a_cellline_foundation.pt")
    ap.add_argument("--out-summary", default="logs/mortfm/beataml_v17_compare.json")
    args = ap.parse_args()

    expr, pairs, drug_vocab, family_vocab = _build_pairs()
    pats = sorted(pairs["patient_id"].astype(str).unique())
    train_pats, test_pats = _split_patient_disjoint(pats, seed=13)
    logger.info("BeatAML v17: %d patients (%d train / %d test); %d drugs / %d families",
                len(pats), len(train_pats), len(test_pats),
                len(drug_vocab), len(family_vocab))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for variant, ckpt in [("scratch", None), ("init_a", args.block_a_checkpoint)]:
        out = _train_one_variant(
            variant=variant,
            block_a_ckpt=Path(ckpt) if ckpt else None,
            pairs=pairs, expr=expr,
            train_pats=train_pats, test_pats=test_pats,
            n_drugs=len(drug_vocab), n_families=len(family_vocab),
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device,
        )
        ckpt_out = Path("checkpoints/mortfm") / f"beataml_v17_{variant}.pt"
        torch.save({
            "encoder_state": out["encoder_state"],
            "head_state": out["head_state"],
            "config": {"epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "variant": variant},
        }, ckpt_out)
        csv_out = Path(f"logs/mortfm/beataml_v17_per_drug_{variant}.csv")
        out["per_drug_df"].to_csv(csv_out, index=False)
        results[variant] = {
            "median_spearman": float(out["per_drug_df"]["spearman"].median()),
            "n_drugs_eval": int(len(out["per_drug_df"])),
            "n_significant_p_lt_0_05": int(
                (out["per_drug_df"]["spearman_p"] < 0.05).sum()
            ),
            "port_report": out["port_report"],
            "checkpoint": str(ckpt_out),
            "per_drug_csv": str(csv_out),
        }

    # Compare
    scratch_med = results["scratch"]["median_spearman"]
    init_med = results["init_a"]["median_spearman"]
    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-beataml-v17-drug-conditioned",
        "n_patients_total": int(len(pats)),
        "n_train_patients": int(len(train_pats)),
        "n_test_patients": int(len(test_pats)),
        "n_drug_vocab": int(len(drug_vocab)),
        "n_family_vocab": int(len(family_vocab)),
        "epochs": args.epochs,
        "results": results,
        "transfer_gain_median": init_med - scratch_med,
        "v16_30epoch_baseline": {
            "scratch_median": -0.006, "init_a_median": 0.103,
            "transfer_gain": 0.110,
        },
        "honest_note": (
            "v17 head is DRUG-CONDITIONED — predicts (specimen, drug) -> AUC "
            "with explicit drug + family embeddings, replacing the v16 "
            "mean-aggregated head. Both variants are trained on the SAME "
            "patient-disjoint split + same per-drug supervision. The "
            "transfer_gain is the apples-to-apples comparison the v17 "
            "plan asked for."
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("v17 BeatAML compare: scratch=%.3f, init_a=%.3f, transfer=%+.3f",
                scratch_med, init_med, init_med - scratch_med)
    return 0


if __name__ == "__main__":
    sys.exit(main())
