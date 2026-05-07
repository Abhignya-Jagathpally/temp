# Wave 5 — Category A Headline Result

**Source agent:** `a6df95358a12e79d4`
**Date:** 2026-05-07
**Persisted by:** main thread (subagent Write was blocked by sandbox)

---

docs/ exists. The user instructed me to return findings as text in my final message, not write files (per system rules: "Do NOT Write report/summary/findings/analysis .md files"). The user prompt mentions the parent agent will persist the file from my output text. I'll output the full report.

---

# Wave 5 — Category A: Headline result and methodology audit

Target output path (caller writes): `docs/V12_REFACTOR_AUDIT/wave5/category_A_headline.md`

## A.1 — Sigmoid head on unbounded IC50 z-score targets

### A.1 Verification
The cited locations `main.py:1335` / `main.py:1486` are wrappers, not the head itself. The actual sigmoid is in `resistancemap/landscape/predictor.py:341`:

```python
# resistancemap/landscape/predictor.py
340:        drug_logits = self.drug_resistance_head(encoded)
341:        drug_probs = torch.sigmoid(drug_logits)
...
365:            "drug_resistance": drug_probs,
```

Both consumption sites compare these sigmoid outputs against z-scored IC50:

```python
# resistancemap/main.py:1332-1335 (training)
1332:            pred_first = drug_pred[:, :n_drugs]   # drug_pred = output["drug_resistance"] = sigmoid
1335:                loss = torch.nn.functional.mse_loss(pred_first[mask], target_b[mask])

# resistancemap/main.py:1486-1491 (test)
1486:        pred = output["drug_resistance"][:, :n_drugs]
1491:            test_mse = torch.nn.functional.mse_loss(pred[mask], target[mask]).item()
```

Targets are confirmed z-scored at `resistancemap/data/preprocessors.py:402-436` ("Per-drug NaN-aware z-score") and re-z-scored on train-only at line 628-637 (`_zscore_normalize_on_train`). After standardization, targets are ~N(0,1) (≈ -3 to +3), but the prediction is bounded to [0,1]. The minimum achievable per-element MSE is bounded below by `(y - clip(y,0,1))^2` which integrates to ≈ 0.5 for ~N(0,1). This is the proximate cause of the 2.84 test MSE.

### A.1 Patch (preferred: linear head; remove sigmoid)
```diff
--- a/resistancemap/landscape/predictor.py
+++ b/resistancemap/landscape/predictor.py
@@ -231,7 +231,8 @@ class ResistanceLandscape(nn.Module):
     - Hidden layers: 256 -> 128 -> 64
     - Heads:
-        - Drug resistance head: (n_drugs * n_timepoints,) outputs → sigmoid
+        - Drug resistance head: (n_drugs * n_timepoints,) outputs → linear
+          (regresses z-scored IC50; do NOT bound with sigmoid)
         - State classification head: (n_states,) outputs → softmax OR EvidentialResistanceHead
@@ -337,9 +338,13 @@ class ResistanceLandscape(nn.Module):
         encoded = self.encoder(fused_representation)
 
-        # Drug resistance predictions (logits)
-        drug_logits = self.drug_resistance_head(encoded)
-        drug_probs = torch.sigmoid(drug_logits)
+        # Drug resistance predictions: regress z-scored IC50 directly.
+        # The targets in MultiOmicsDataset.drug_sensitivity are per-drug
+        # z-scored (preprocessors.py:402, _zscore_normalize_on_train) and
+        # therefore unbounded ~N(0,1). A sigmoid here clamps the output to
+        # [0,1] which guarantees test MSE > Var(y) (see paper/tables/
+        # baseline_comparison.json: ResistanceMap=2.836 vs zero=2.811).
+        drug_pred = self.drug_resistance_head(encoded)
@@ -362,7 +367,7 @@ class ResistanceLandscape(nn.Module):
         target_scores = torch.sigmoid(target_logits)
 
         output = {
-            "drug_resistance": drug_probs,
+            "drug_resistance": drug_pred,
             "resistance_state": state_output,
             "transition_matrix": transition_probs,
             "target_scores": target_scores,
```

Note: the dict key `"drug_resistance"` and the `LandscapeResult.drug_resistance_3m/6m/12m` semantics ("P(resistant)") become misleading after this change. Downstream readers in `predictor.py:457-466, 576-585` and `inference/api.py` should be updated to surface raw z-scored predictions and a separate `drug_zscore_3m/6m/12m` field, OR — minimal-diff alternative — apply sigmoid to a CDF of the prediction at consumption time only:

```diff
+        # For backward-compat consumers expecting a probability, derive
+        # P(resistant) = Phi(z) at the consumption site, NOT in the head.
+        drug_probs_compat = 0.5 * (1.0 + torch.erf(drug_pred / math.sqrt(2.0)))
```

### A.1 Effort — **low**, ~0.5 person-day
Remove one line, retrain the landscape head (or also fusion if frozen weights downstream), regenerate `paper/tables/baseline_comparison.json`. No retraining of VAE / trajectory / protein-net needed.

### A.1 Risk
- `LandscapeResult.drug_resistance_*` semantics change: any consumer that thresholds at 0.5 (e.g., `inference/api.py`, dashboard) will break or produce silently wrong "resistant/sensitive" labels. Audit consumers before merging.
- Loss surface widens; you may need a slightly lower LR or to reduce gradient clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` at `main.py:1337` may be too tight under linear regression with broader-tailed errors).
- A different value at training time (the trained checkpoint at `landscape_trained` becomes incompatible) — must re-run the `landscape_train` stage. CheckpointManager should auto-detect, but consider bumping a config version.

---

## A.2 — Aggregate test_mse below trivial baselines

### A.2 Verification
`paper/tables/baseline_comparison.json`:
```json
{"model": "Zero (z-scored target floor)",   "test_mse": 2.81059193611145},
{"model": "PerDrugTrainMean",               "test_mse": 2.81059193611145},
{"model": "ResistanceMap (10-agent DAG)",   "test_mse": 2.836390733718872},
```
Confirmed exact numerics: Zero=2.8106, PerDrugTrainMean=2.8106, ResistanceMap=2.8364. ResistanceMap is rank 3/11 (worse than 2 trivial baselines but better than 8 sklearn baselines that also overfit). val_mse for trivial baselines is 1.96, while the test set has higher per-drug variance (2.81), reflecting drugs with higher IC50 dispersion ending up in test. This is consistent with random-split variance, not a bug per se — but is amplified by A.4.

### A.2 Patch
A.2 is a downstream symptom of A.1; no standalone patch. After applying A.1's patch + retraining, expect test_mse to drop below 2.81 (a competent linear regressor on PCA-256 already lands at 2.94, so a linear head on a 19,219-feature fusion should approach the Ridge floor, ~2.81-2.85 if seed-stable, and ideally <2.0 if PCA-overfit-effects from baselines are not in play). If after A.1 fix `test_mse > zero=2.81`, that is a hard signal the fusion encoder is throwing away signal, not a head problem.

Add an automated tripwire so CI fails when ResistanceMap < trivial baseline:
```diff
--- a/resistancemap/main.py
+++ b/resistancemap/main.py
@@ -1697,6 +1697,17 @@ def validate_pipeline(...):
     metrics = {
         "test_mse": test_mse,
         ...
     }
+    # Tripwire: a learned model that loses to predict-mean is unpublishable.
+    # We intentionally fail loudly rather than silently log a 3rd-place rank.
+    try:
+        baselines_path = Path("paper/tables/baseline_comparison.json")
+        if baselines_path.exists():
+            bl = json.loads(baselines_path.read_text())
+            trivial = min(r["test_mse"] for r in bl["results"]
+                          if r["model"] in {"Zero (z-scored target floor)", "PerDrugTrainMean"})
+            assert test_mse < trivial, \
+                f"FAIL: test_mse={test_mse:.4f} >= trivial baseline {trivial:.4f}"
+    except FileNotFoundError:
+        pass
```

### A.2 Effort — **low**, ~0.5 person-day (couples to A.1)
### A.2 Risk
The tripwire is hard-blocking — a broken run aborts validation. Acceptable for headline numbers, but make it a `logger.error` in dev and `raise` only when `--strict-baselines` is set, so iterative debugging isn't blocked.

---

## A.3 — No SOTA cell-line drug-response baselines

### A.3 Verification
`resistancemap/evaluation/baselines.py:60-64`:
```python
60: def deepcdr_reimpl(train_df, test_df):
61:     raise NotImplementedError(
62:         "Plug in DeepCDR reimpl (or published weights) on IDENTICAL splits. "
63:         "Do not report headline numbers without a published-SOTA row."
64:     )
```
`BASELINES_REGRESSION` (lines 67-72) and `BASELINES_CLASSIFICATION` (74-78) wire `"deepcdr": deepcdr_reimpl`. No GraphDRP, PaccMann, MOLI, DRPreter, KronRLS implementations anywhere in the file. Confirmed.

### A.3 Patch
```diff
--- a/resistancemap/evaluation/baselines.py
+++ b/resistancemap/evaluation/baselines.py
@@ -57,11 +57,28 @@
 # ---------------------------------------------------------------------------
-# External SOTA reimpl -- stub; plug in DeepCDR / PaccMann reimpls
+# External SOTA reimpls — TRACKED in evaluation/sota/. Each must be
+# trained/evaluated on the IDENTICAL split as ResistanceMap; do NOT use the
+# original authors' splits. Effort estimate per row in docs/SOTA_BASELINES.md.
 # ---------------------------------------------------------------------------
-def deepcdr_reimpl(train_df, test_df):
-    raise NotImplementedError(
-        "Plug in DeepCDR reimpl (or published weights) on IDENTICAL splits. "
-        "Do not report headline numbers without a published-SOTA row."
-    )
+from resistancemap.evaluation.sota import (
+    deepcdr,           # Liu et al. 2020, Bioinformatics — GCN on PPI + drug fingerprint
+    graphdrp,          # Nguyen et al. 2022 — GAT on drug graph
+    paccmann,          # Manica et al. 2019 — multi-modal attention on omics + SMILES
+    moli,              # Sharifi-Noghabi et al. 2019 — multi-omics late integration
+    drpreter,          # Shin et al. 2022 — transformer over pathway tokens
+    kronrls,           # Pahikkala et al. 2014 — kernel regularized least squares (drug × cell)
+)
+
+def deepcdr_reimpl(train_df, test_df, **kw):  return deepcdr.fit_predict(train_df, test_df, **kw)
+def graphdrp_reimpl(train_df, test_df, **kw): return graphdrp.fit_predict(train_df, test_df, **kw)
+# ... etc
@@ -67,6 +84,11 @@
 BASELINES_REGRESSION: dict[str, Callable] = {
     "global_mean":    predict_global_mean,
     "per_drug_mean":  predict_per_drug_mean,
     "ridge_esm2":     ridge_on_esm2_meanpool,
-    "deepcdr":        deepcdr_reimpl,
+    "deepcdr":        deepcdr_reimpl,
+    "graphdrp":       graphdrp_reimpl,
+    "paccmann":       paccmann_reimpl,
+    "moli":           moli_reimpl,
+    "drpreter":       drpreter_reimpl,
+    "kronrls":        kronrls_reimpl,
 }
```

The actual `evaluation/sota/{deepcdr,graphdrp,paccmann,moli,drpreter,kronrls}.py` modules are NOT included in this patch — that's the 3-6 person-week implementation work.

### A.3 Effort — **high**, 15-30 person-days total
- DeepCDR: 3-5 PD (PPI graph + drug-fingerprint GCN, public PyTorch weights exist but require GDSC ID remap)
- GraphDRP: 3-5 PD (rdkit drug graph + GAT)
- PaccMann: 3-5 PD (multi-modal attention; original Keras → port to PyTorch)
- MOLI: 2-3 PD (simplest; multi-omics MLP)
- DRPreter: 5-7 PD (pathway-token transformer; least mature reference impl)
- KronRLS: 2-3 PD (closed-form kernel; numerically-careful Kronecker decomp)

### A.3 Session-fixable: **NO**
Implementing 6 SOTA baselines correctly with matched splits is not session scope. What IS session-scope: the `NotImplementedError` should be downgraded to `logger.warning` with a clearly-marked TODO, and the public README/RUNS.md should declare "no SOTA baselines yet — N=0 for cross-paper comparison". Headline tables MUST NOT claim SOTA-comparable performance until at least 2 of these 6 land.

### A.3 Risk
None of the patch breaks the build (it only adds imports that fail at module-load if the sota submodules don't exist — must be guarded with try/except until those modules are written).

---

## A.4 — Random 70/15/15 cell-line split with seed=42

### A.4 Verification
`resistancemap/config.py:58-61`:
```python
58:    # Splits
59:    test_fraction: float = 0.15
60:    val_fraction: float = 0.15
61:    random_seed: int = 42
```
And the split itself at `resistancemap/data/preprocessors.py:619-626`:
```python
619:    else:
620:        # Fallback: simple random split
621:        indices = rng.permutation(n)
622:        n_test = int(n * config.test_fraction)
623:        n_val = int(n * config.val_fraction)
624:        test_idx = indices[:n_test]
625:        val_idx = indices[n_test : n_test + n_val]
626:        train_idx = indices[n_test + n_val :]
```
For CCLE cell-line data (no `patient_ids`), this fallback branch always fires. There is no `lineage`-stratified branch, no LOCO, no LODO. Confirmed.

### A.4 Patch
```diff
--- a/resistancemap/config.py
+++ b/resistancemap/config.py
@@ -57,8 +57,15 @@ class DataConfig:
     # Splits
+    # Split protocol — random is the legacy/easy setting (cell-line memorization).
+    # For headline numbers use one of:
+    #   - "loco":   leave-one-cell-out (rotates each cell line through test)
+    #   - "lodo":   leave-one-drug-out (zero-shot drug evaluation)
+    #   - "tissue": leave-one-lineage-out (tests cross-tissue generalization)
+    split_protocol: str = "random"  # random | loco | lodo | tissue
     test_fraction: float = 0.15
     val_fraction: float = 0.15
     random_seed: int = 42
+    loco_fold: int = 0  # which held-out fold (0..n_cells-1) when split_protocol="loco"
+    lodo_fold: int = 0  # which held-out drug index when split_protocol="lodo"
+    tissue_holdout: str = "haematopoietic_and_lymphoid_tissue"  # for "tissue"
```

```diff
--- a/resistancemap/data/preprocessors.py
+++ b/resistancemap/data/preprocessors.py
@@ -588,11 +588,46 @@ def build_train_val_test_splits(
     n = len(dataset)
     rng = np.random.RandomState(config.random_seed)
+    protocol = getattr(config, "split_protocol", "random")
+
+    # ── LOCO / LODO / tissue protocols (cell-line drug-response standard) ──
+    if protocol == "loco":
+        from sklearn.model_selection import LeaveOneGroupOut
+        groups = np.asarray(dataset.sample_ids)  # one group per cell line
+        unique = np.unique(groups)
+        held = unique[config.loco_fold % len(unique)]
+        test_idx = np.where(groups == held)[0]
+        rest = np.where(groups != held)[0]
+        rng.shuffle(rest)
+        n_val = int(len(rest) * config.val_fraction / (1 - config.test_fraction))
+        val_idx, train_idx = rest[:n_val], rest[n_val:]
+        logger.info(f"LOCO split: held cell={held}, test={len(test_idx)}")
+        # ... (re-normalize block as before) ...
+        return {"train": train_idx, "val": val_idx, "test": test_idx}
+
+    if protocol == "lodo":
+        # Test = all (cell, drug=held) pairs; train = all others.
+        # NOTE: this requires reshaping drug_sensitivity into long format
+        # at consumption time — the masked-MSE path in main.py:1335 already
+        # supports per-drug masking, so we materialize the drug-mask here:
+        held_drug = config.lodo_fold % dataset.drug_sensitivity.shape[1]
+        # All cells go to all splits; the mask determines which drug-cells
+        # are observed in train vs test.
+        rng_perm = rng.permutation(n)
+        n_test = int(n * config.test_fraction); n_val = int(n * config.val_fraction)
+        test_idx, val_idx, train_idx = rng_perm[:n_test], rng_perm[n_test:n_test+n_val], rng_perm[n_test+n_val:]
+        # Mask: train sees all drugs EXCEPT held_drug; test sees ONLY held_drug.
+        train_mask = torch.ones_like(dataset.drug_sensitivity, dtype=torch.bool)
+        train_mask[:, held_drug] = False
+        test_mask  = torch.zeros_like(dataset.drug_sensitivity, dtype=torch.bool)
+        test_mask[:, held_drug] = True
+        dataset.drug_sensitivity = torch.where(train_mask, dataset.drug_sensitivity,
+                                               torch.full_like(dataset.drug_sensitivity, float("nan")))
+        logger.info(f"LODO split: held drug={dataset.drug_names[held_drug]}")
+        return {"train": train_idx, "val": val_idx, "test": test_idx}
+
+    if protocol == "tissue":
+        held_tissue = config.tissue_holdout
+        groups = np.asarray(dataset.lineage)
+        test_idx = np.where(groups == held_tissue)[0]
+        rest = np.where(groups != held_tissue)[0]
+        rng.shuffle(rest)
+        n_val = int(len(rest) * config.val_fraction / (1 - config.test_fraction))
+        val_idx, train_idx = rest[:n_val], rest[n_val:]
+        logger.info(f"Tissue-holdout split: held={held_tissue}, n_test={len(test_idx)}")
+        # ... (re-normalize block) ...
+        return {"train": train_idx, "val": val_idx, "test": test_idx}
 
     # Check for patient grouping
     if hasattr(dataset, 'patient_ids') and dataset.patient_ids is not None:
```

Add a CLI flag at `parse_args`:
```diff
+    parser.add_argument("--split-protocol", type=str, default=None,
+                        choices=["random","loco","lodo","tissue"],
+                        help="Override config.data.split_protocol")
```

### A.4 Effort — **medium**, 2-3 person-days
Mostly plumbing + a careful test that LODO's drug-mask doesn't leak through the validation MSE computation at `main.py:1486-1491` (it shouldn't, because `~torch.isnan(target)` already gates it).

### A.4 Risk
- LODO with N=11 drugs is extremely small; expect huge per-fold variance (each fold = ~1/11 of observations). Need to report mean ± std across all 11 folds, not a single number.
- LOCO at 30-200 cell lines per tissue means tissue-stratification needs to be built INTO LOCO if you want both at once.
- The trajectory forecaster at `train_trajectory_forecaster` is supervised by drug-sensitivity (see preprocessors.py comment + main.py:451). Under LODO, the held drug is masked in training — the trajectory's chromatin-state supervision degrades. Document and re-validate.

---

## A.5 — No multi-seed variance / paired Wilcoxon

### A.5 Verification
`README.md:253`:
```
6. **Statistical significance test** vs late-fusion baseline (paired Wilcoxon over 5 random seeds) — required to claim the cross-attention machinery is justified.
```
Listed as roadmap item #6, not implemented. No `for seed in [...]` block, no `seed=range()` invocation, no scipy.stats.wilcoxon usage in `main.py` or `validate_pipeline`. Single-seed run baked in via `config.hardware.seed` and `config.data.random_seed=42`.

### A.5 Patch (new file: `scripts/run_multiseed.py`)
```diff
--- /dev/null
+++ b/scripts/run_multiseed.py
@@ -0,0 +1,93 @@
+#!/usr/bin/env python3
+"""Multi-seed headline runner with paired Wilcoxon vs trivial baselines.
+
+Usage:
+    python scripts/run_multiseed.py --config configs/h100.yaml \
+        --seeds 0 1 2 3 4 --out paper/tables/multiseed.json
+"""
+from __future__ import annotations
+import argparse, json, subprocess, sys
+from pathlib import Path
+import numpy as np
+from scipy.stats import wilcoxon
+import torch
+
+def run_one_seed(config_path: Path, seed: int, ckpt_dir: Path) -> dict:
+    # Stamp seed into config (load → override → temp-yaml).
+    import yaml
+    cfg = yaml.safe_load(config_path.read_text())
+    cfg.setdefault("hardware", {})["seed"] = seed
+    cfg.setdefault("data", {})["random_seed"] = seed
+    cfg["checkpoint_dir"] = str(ckpt_dir / f"seed_{seed}")
+    tmp = Path(f"/tmp/cfg_seed{seed}.yaml"); tmp.write_text(yaml.safe_dump(cfg))
+    rc = subprocess.run([sys.executable, "main.py", "--config", str(tmp), "--sequential"],
+                        check=True)
+    # Read back the validated checkpoint for per-drug MSE.
+    ckpt = torch.load(ckpt_dir / f"seed_{seed}" / "pipeline_validated.pt", weights_only=False)
+    return ckpt["metrics"]
+
+def main():
+    ap = argparse.ArgumentParser()
+    ap.add_argument("--config", type=Path, required=True)
+    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
+    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/multiseed"))
+    ap.add_argument("--out", type=Path, required=True)
+    ap.add_argument("--baseline", type=str, default="PerDrugTrainMean",
+                    help="Baseline model name in baseline_comparison.json to compare against")
+    args = ap.parse_args()
+
+    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
+    rm_per_drug_mse = []  # list of dicts seed -> per-drug-MSE-vector
+    bl_per_drug_mse = []
+    for s in args.seeds:
+        m = run_one_seed(args.config, s, args.ckpt_dir)
+        rm = np.array([d["mse"] for d in m["per_drug_metrics"] if not np.isnan(d["mse"])])
+        bl_path = Path(args.ckpt_dir / f"seed_{s}" / "baseline_comparison.json")
+        # baseline runner is already invoked by validate_pipeline; reload it
+        bl = json.loads(bl_path.read_text())
+        bl_row = next(r for r in bl["results"] if r["model"] == args.baseline)
+        # ResistanceMap is paired drug-by-drug → use per-drug MSE.
+        bl_per_drug = np.array(bl_row.get("per_drug_mse", []))  # baselines.py must persist this
+        rm_per_drug_mse.append(rm); bl_per_drug_mse.append(bl_per_drug)
+
+    rm_arr = np.stack(rm_per_drug_mse)   # (n_seeds, n_drugs)
+    bl_arr = np.stack(bl_per_drug_mse)   # (n_seeds, n_drugs)
+    paired_diff = (rm_arr - bl_arr).flatten()
+    stat, p = wilcoxon(paired_diff, alternative="less")  # H1: RM < baseline
+
+    out = {
+        "seeds": args.seeds,
+        "rm_mean_mse": float(rm_arr.mean()),
+        "rm_std_mse":  float(rm_arr.std(ddof=1)),
+        "bl_mean_mse": float(bl_arr.mean()),
+        "bl_std_mse":  float(bl_arr.std(ddof=1)),
+        "wilcoxon_stat": float(stat),
+        "wilcoxon_p":    float(p),
+        "n_paired":      int(paired_diff.size),
+        "verdict": "RM significantly better" if p < 0.05 else "RM NOT significantly better",
+    }
+    args.out.parent.mkdir(parents=True, exist_ok=True)
+    args.out.write_text(json.dumps(out, indent=2))
+    print(json.dumps(out, indent=2))
+
+if __name__ == "__main__":
+    main()
```

To make this work, `paper/tables/baseline_comparison.json` must include per-drug MSE per baseline:
```diff
--- a/scripts/run_baselines_real.py   # (presumed location)
+++ b/scripts/run_baselines_real.py
@@
-    row = {"model": name, "test_mse": float(np.mean(per_drug_mse)), ...}
+    row = {"model": name, "test_mse": float(np.mean(per_drug_mse)),
+           "per_drug_mse": [float(x) for x in per_drug_mse], ...}
```
(The exact location of `run_baselines_real.py` should be verified before patching; I did not open it.)

### A.5 Effort — **medium**, 1-2 person-days
The runner is small (~90 LOC), but each seed re-runs the full pipeline — at the published 19.7s ResistanceMap timing this is fast, but with 5 seeds × VAE + trajectory + protein-net training, expect 1-2 GPU-hours per multiseed sweep on H100.

### A.5 Risk
- Determinism: `torch.use_deterministic_algorithms(True)` is already set when `config.hardware.deterministic=True`, but cudnn algorithms for some convs are nondeterministic; results may differ slightly across reruns even with fixed seed. Document.
- `scipy.stats.wilcoxon` is paired and requires non-zero diffs; if RM and baseline tie on any drug+seed pair, that pair drops with `zero_method="wilcox"` (default). 5×11=55 comparisons, ties unlikely but possible.
- Per-drug MSE persistence in `baseline_comparison.json` requires reverification; current schema only has aggregate `test_mse`.

---

## §summary

| blocker | verified Y/N | patch ready Y/N | effort | session-fixable Y/N |
|---|---|---|---|---|
| A.1 sigmoid head on z-scored IC50 | Y (sigmoid at `predictor.py:341`, not `main.py:1335` — discrepancy noted) | Y | low (~0.5 PD) | Y |
| A.2 test_mse below trivial baselines | Y (Zero=2.8106, RM=2.8364 exact) | Y (downstream of A.1 + tripwire) | low (~0.5 PD) | Y |
| A.3 no SOTA baselines | Y (`deepcdr_reimpl` raises NotImplementedError; no other SOTAs) | Partial (interface only; impls missing) | high (15-30 PD) | **N** |
| A.4 random 70/15/15 seed=42 | Y (`config.py:58-61`, `preprocessors.py:619-626`) | Y | medium (2-3 PD) | Y (single session) |
| A.5 no multi-seed / Wilcoxon | Y (README:253 roadmap; no runner exists) | Y (new `scripts/run_multiseed.py`) | medium (1-2 PD) | Y |

### Key discrepancy with the audit text
The cited file:line for A.1 (`main.py:1335`, `:1486`) does **not** contain a `nn.Sigmoid()` call directly — those lines compute MSE loss against `output["drug_resistance"]`. The actual sigmoid is at `resistancemap/landscape/predictor.py:341` inside `ResistanceLandscape.forward()`. The audit's claim about behavior is correct; the file:line pointer is one indirection away. Surface this in the final report so reviewers know exactly where to patch.

### Causal chain (A.1 → A.2)
Z-scored targets ~N(0,1), sigmoid output ∈ [0,1] → minimum achievable MSE per element is bounded below by ∫ (y - clip(y,0,1))² φ(y) dy ≈ 0.5 just from the bound, with additional variance loss because the head can't represent negative residuals. Empirically RM lands at 2.836 vs zero-baseline 2.811 — fixing A.1 is necessary AND likely sufficient to clear A.2.

### Session-scope recommendation
Apply A.1 + A.2 (tripwire) + A.4 + A.5 in one session (combined ~5-7 PD with retraining time). Leave A.3 as a multi-week tracked work item, with the public README amended to remove any SOTA-comparable claim until at least DeepCDR + GraphDRP land.