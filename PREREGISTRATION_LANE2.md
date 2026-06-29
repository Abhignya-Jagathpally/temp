# PRE-REGISTRATION — Contribution-2, Lane-2 (treatment-conditioned)
**Committed BEFORE the Lane-2 run (timestamp + git SHA in the bundle). Report whatever it shows.**

## 0. Why this document exists
H1 (a=0, cross-sectional) is an honest NEGATIVE: GeoFlow/manifold do not beat the
~0.62 bulk-RNA→PFS discrimination ceiling, and calibration did not win either. But
`a=0` never exercised GeoFlow's differentiator (treatment-conditioning + counterfactual
flow). Testing the treatment axis now is **completing the original ResistanceMap thesis
(resistance = relative to therapy; TT2L endpoint)**, not a post-hoc pivot. To keep it
honest rather than HARKing, the hypothesis, success axis, feasibility gate, and
falsification criteria are fixed *here, in advance*.

## 1. Hypothesis (pre-registered)
**H-C2:** A *treatment-conditioned* geometry-aware first-passage flow yields counterfactual
treatment-benefit orderings concordant with randomized RRMM trial HRs (RQ9), and/or
improves time-to-event prediction on the **non-PH treatment substrate** over a
proportional-hazards baseline. The unique, primary claim is the **counterfactual axis**,
which does NOT require beating C-index.

## 2. Substrate & variables (fixed)
- **Lane-2** = mmSYGNAL programs ∩ `treatments.tsv`, n=712. Documented non-PH signal:
  `n_lines` PH-violation χ²=14.7, p=1e-4 (a static Cox is misspecified here).
- **Endpoints:** PFS **and TT2L** (time-to-next-treatment — the resistance-native target,
  finally appropriate). Report both + OS; never cherry-pick.
- **Treatment vector** `a` = regimen-class exposure [PI, IMiD, antiCD38, BCMA_Tcell,
  GPRC5D_Tcell, ADC_BCMA, steroid] (extend `configs/geoflow.yaml`).
- Splits: identical `patient_kfold(k=5, seed=0)`; leakage discipline unchanged (OOS
  manifold, train-fold fit-only, claim-gate).

## 3. Models compared (identical folds)
| Model | Role |
|---|---|
| PH-Cox (programs+clin) | the proportional-hazards bar |
| **RSF + landmark/time-varying Cox** | **feasibility gate: does ANY non-PH model beat PH-Cox here?** |
| Treatment-conditioned GeoFlowSurv (a≠0) | the contribution |
| mmSYGNAL / gep70 / sky92 | reference bars |

## 4. FEASIBILITY GATE (run FIRST — cheap, ~hours, before the full build)
Statistical non-PH (χ²=14.7) does **not** guarantee predictive gain. Gate:
- Compute C-index/IBS on Lane-2 for PH-Cox vs RSF vs time-varying/landmark Cox, same folds.
- **PASS** if a non-PH baseline's CI-lower exceeds PH-Cox on TT2L or PFS → exploitable signal
  exists → build the full treatment-conditioned GeoFlow.
- **FAIL** if no non-PH model beats PH-Cox → the predictive axis is dead; proceed ONLY on the
  counterfactual axis (§5) or stop and write the complete negative (§6).

## 5. PRIMARY success axis — counterfactual concordance (RQ9)
Using `counterfactual/trial_concordance.py` against `data/tables/rrmm_trials.csv` (16 trials):
- **SUPPORTED** if out-of-fold counterfactual benefit ordering vs trial −log(HR) has
  **Spearman ≥ 0.5 AND sign-agreement ≥ 0.8**, with a permutation-null p<0.05.
- Secondary: GeoFlow CI-lower > best non-PH baseline on **TT2L** discrimination, and/or
  better IBS/D-cal (after the D-cal extraction fix, §7).

## 6. FALSIFICATION criteria (pre-committed)
The thesis is **falsified** if, on Lane-2 with a≠0: counterfactual concordance Spearman ≤ 0
or sign-agreement ≤ 0.5, **AND** GeoFlow ≤ baselines on TT2L/PFS discrimination and calibration.
→ Then write the **complete, bulletproof negative**: geometry+flow+treatment-conditioning do
not exceed simple baselines for MM outcome on open multi-modal data, with the ceiling
characterization (§8) as the positive scientific contribution. A complete falsification
across both modalities is a *stronger, more publishable* result than the partial a=0 negative.

## 7. Required fixes before judging (Step A — ~30 min, do first)
1. **D-calibration extraction bug:** stat ≈395 / p≈0 for *all* models incl. well-calibrated
   baselines ⇒ S(T_i) extraction / curve scaling is wrong, not the models. Fix the S(t) grid
   eval (time-unit-scaled IG mean) so H3 can be judged fairly.
2. **Honest baseline bar:** `run_h1.py`'s same-optimizer `cox_baseline` (0.573) under-converges
   vs the real bar (0.62). Label it a *control*, and report the real reproduced baselines
   (Stacked 0.644 / GBS 0.643 / Cox 0.632) as THE bar in the leaderboard.
3. Add the honest survival-only torch GeoFlow row (0.575) as the "best GeoFlow" so the negative
   is stated at its strongest.

## 8. Capture regardless — the ceiling finding (standalone result)
Across 6 model classes (clinical, sparse-gene gep70/sky92, GBS, Stacked, v1 PH-free, GeoFlow),
static bulk-RNA→PFS discrimination saturates at **C≈0.62**. Message: the ceiling is a property
of the **modality**, not the model; gains require **treatment dynamics** (this Lane-2 work) or
**finer resolution** (scRNA, RQ10). This is honest, citable, and needle-moving on its own.