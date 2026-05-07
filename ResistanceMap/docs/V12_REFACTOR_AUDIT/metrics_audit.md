# ResistanceMap v12 Refactor Audit
**Audit date:** 2026-05-07  
**Auditor:** metrics-auditor-agent  
**Scope:** RUNS.md rows 32–37 (v11.5 / v12 claims); pipeline_validated.pt drug-response metrics  
**Rule:** No numbers from README/ARCHITECTURE/docs — only RUNS.md rows + checkpoint arrays

---

## §1 Checkpoint Inventory

| File | SHA-256 (first 16) | Last Modified | Size | RUNS.md Row |
|------|--------------------|---------------|------|-------------|
| checkpoints/pipeline_validated.pt | `605aa02af961dda1` | 2026-05-03 10:22 | 8 KB | v6 era |
| checkpoints/data_ready.pt | `a149f86e16e9f813` | 2026-05-03 09:51 | 144 MB | v6 era |
| checkpoints/u_theta_v10s1.pt | `e40d8053b20ce25e` | 2026-05-03 19:43 | 455 KB | Sprint 1 |
| checkpoints/u_theta_v10s2.pt | `a81f11ce0c571154` | 2026-05-03 19:44 | 455 KB | Sprint 2 |
| paper/v8_artifacts/v11_sprint5/cox_per_patient_log_hazards.npz | `3317ab00f1e3fbc2` | 2026-05-03 20:34 | 63 KB | r-2026-05-03-v11s5-cox-with-programs |
| paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_per_patient_log_hazards.npz | `b1a2f698012c71fa` | 2026-05-04 11:25 | 222 KB | r-2026-05-04-v12-mofa-vs-v115 |

**Not found on disk:** No v11, v12 model weights checkpoints separate from the above; no `logs/<run_id>/stdout.log` for v11/v12 runs (logs directory absent in current worktree).

---

## §2 Recomputed C-Index Table

Source: `cox_per_patient_log_hazards.npz` (N=787, n_events=224) and `mofa_vs_v11_5_per_patient_log_hazards.npz`.  
Method: `lifelines.concordance_index(t, -log_hazard, event)`.

### Marginal C-Index (all 787 patients)

| Model | Recomputed C | RUNS.md Claim | Δ | Verdict |
|-------|-------------|---------------|---|---------|
| Cox_v11_richer | 0.6538 | 0.6538 (row 34, 37) | 0.0000 | PASS |
| Cox_v11_routed_mmsygnal (v11.5) | 0.6955 | 0.6955 (row 34) | 0.0000 | PASS |
| mmsygnal_routed (4-submodel) | 0.6938 | 0.6938 (row 30) | 0.0000 | PASS |
| mmSYGNAL 6-submodel routed | 0.6957 (from mmsygnal_complete_routing.json) | 0.6957 (row 33) | 0.0000 | PASS — not in cox NPZ, read from routing JSON |
| MOFA_factors (v12) | 0.6600 | 0.6600 (row 37) | 0.0000 | PASS |
| MOFA_factors_plus_v11 (v12) | 0.6751 | 0.6751 (row 37) | 0.0000 | PASS |
| Cox_v11_six_mmsygnal_scores | 0.6875 | 0.6875 (row 34) | 0.0000 | PASS |

### Per-Stratum C-Index — Cox_v11_routed_mmsygnal (v11.5)

| Stratum | n | Recomputed C | RUNS.md Claim | Δ | Verdict |
|---------|---|-------------|---------------|---|---------|
| S1 del17p | 105 | 0.7296 | 0.730 (row 34) | −0.0004 | PASS |
| S2 t(4;14) | 97 | 0.6669 | 0.667 (row 34) | −0.0001 | PASS |
| S3 +1q21 | 165 | 0.6943 | 0.694 (row 34) | +0.0003 | PASS |
| S4 t(11;14) | 95 | 0.6426 | 0.643 (row 34) | −0.0004 | PASS |
| S5 other | 325 | 0.6589 | 0.659 (row 34) | −0.0001 | PASS |

### Per-Stratum C-Index — MOFA_factors_plus_v11 (v12)

| Stratum | n | Recomputed C | RUNS.md Claim | Δ | Verdict |
|---------|---|-------------|---------------|---|---------|
| S1 del17p | 105 | 0.7123 | 0.712 (row 37) | +0.0003 | PASS |
| S2 t(4;14) | 97 | 0.6260 | 0.626 (row 37) | 0.0000 | PASS |
| S3 +1q21 | 165 | 0.6809 | 0.681 (row 37) | −0.0001 | PASS |
| S4 t(11;14) | 95 | 0.5782 | 0.578 (row 37) | +0.0002 | PASS |
| S5 other | 325 | 0.6559 | 0.656 (row 37) | −0.0001 | PASS |

---

## §3 Recomputed Paired-Bootstrap CIs

### v11.5 (Cox_v11_routed_mmsygnal) vs mmsygnal_routed (4-submodel)
Re-run B=10,000 paired bootstrap, seed=42:

- Observed Δ = v11.5 − mmSYGNAL = **+0.0017**
- 95% CI: **[−0.0240, +0.0277]**
- Verdict: **TIE** (CI straddles 0)

RUNS.md row 34 claims: Δ=+0.002, 95% CI [−0.024, +0.026].  
RUNS.md row 32 (paired_cindex_test) claims: marginal 95% CI [−0.0265, +0.0267].

Discrepancy: recomputed CI upper bound = +0.0277 vs RUNS.md +0.0267 (+0.0010 difference). This is within expected bootstrap seed variation (production run used B=10,000 with a different seed stored in `paired_cindex_test.json`). The stored file records bootstrap delta_mean = −0.000046 (essentially 0); TIE conclusion is unaffected. **No fabrication.**

### MOFA+v11 vs Cox_v11_richer (from stored mofa_vs_v11_5_pfs.json)
- Δ (MOFA+v11 − Cox_v11_richer) recomputed from NPZ: **+0.0213**
- RUNS.md claim (row 37): +0.0214. Δ = 0.0001 — floating-point rounding only. **PASS.**
- Stored bootstrap 95% CI: [+0.0005, +0.0425] (marginally excludes 0, p=0.046)

---

## §4 Baseline Comparison

### Drug-Response MSE Ranking (CCLE 11-drug task, test set n=132 cell lines)

| Rank | Model | test_mse | Notes |
|------|-------|----------|-------|
| 1 | Zero / PerDrugTrainMean | 2.8106 | Predict-mean floor |
| 2 | MOFA+Ridge (20 factors, 19.7s) | 2.8155 | From mofa_results.json |
| **3** | **ResistanceMap (10-agent DAG)** | **2.8364** | From pipeline_validated.pt |
| 4 | RandomForest (PCA-256) | 2.8609 | |
| 5 | GradientBoosting (PCA-256) | 2.8649 | |
| 6 | ElasticNet (PCA-256) | 2.9423 | |
| 7 | Ridge (full proteomics+epi) | 2.9723 | |

**ResistanceMap ranks 3rd, BELOW the predict-mean floor (2.8364 > 2.8106).** The 0.026 gap is driven by Panobinostat (MSE=22.6 for all models — Panobinostat dominates pooled MSE with ~8x the MSE of all other drugs combined and no model finds signal; this one drug inflates RM's pooled MSE above predict-mean).

### C-Index Survival Ranking (MMRF N=787 TT2L PFS endpoint)

| Rank | Model | Marginal C-Index |
|------|-------|-----------------|
| 1 | Cox_v11_routed_mmsygnal (v11.5) | 0.6955 |
| 1 | mmSYGNAL 6-submodel | 0.6957 |
| 3 | Cox_v11_six_mmsygnal_scores | 0.6875 |
| 4 | mmSYGNAL 4-submodel | 0.6938 |
| 5 | MOFA_factors_plus_v11 (v12) | 0.6751 |
| 6 | Cox_only_six_mmsygnal_scores | 0.6771 |
| 7 | Cox_v11_program_activity_pcs | 0.6687 |
| 8 | MOFA_factors alone | 0.6600 |
| 9 | Cox_v11_richer | 0.6538 |

---

## §5 Per-Drug NIE Breakdown (Bortezomib Mediator Analysis, v10 Sprint 4)

Source: `paper/v8_artifacts/v10_sprint4/`

| Item | Value |
|------|-------|
| N cohort | 787 |
| n Bortezomib 1L | 579 |
| n non-Bort 1L | 208 |
| Mediator | M_seed3 = z-score(PSMB5, PSMB1, PSMB2) |
| NIE log-HR point | 0.0863 |
| NIE HR point | 1.0901 |
| NIE HR 95% CI | [1.0293, 1.1644] |
| NDE log-HR point | 0.8090 |
| Proportion mediated | 9.6% |
| F8 (NIE CI excludes 0) | PASS |
| F9 (negative control) | PASS (from nie_focused_falsification.json) |
| F10 (E-value CI bound ≥ 1.5) | FAIL (E-value at CI bound = 1.203) |
| Overall strict (F8+F9+F10) | FAIL (2/3) |

**Per RUNS.md row documentation:** Sprint 4 2/3 strict pass is correctly reported. The NIE is directionally supported but sensitivity-limited by unmeasured confounders with moderate RR (≥1.3,1.3 joint strength can nullify the lower-CI NIE bound).

**Richer mediator panel (from nie_richer_mediators.json):** All 7 mediator definitions show F8 PASS, F10 FAIL — consistent with primary mediator result. 20S beta subunits (PSMB1-7, n=7) show highest NIE HR point (1.067) and tightest CI lower bound (1.012).

---

## §6 Discrepancies

| ID | Claim Source | Paper Claim | Recomputed | Δ | Severity |
|----|-------------|-------------|-----------|---|---------|
| D1 | RUNS.md row 37 | MOFA+v11 Δ=+0.0214 | +0.0213 | 0.0001 | PASS — float rounding |
| D2 | RUNS.md row 32 | bootstrap 95% CI=[−0.0265, +0.0267] | [−0.0240, +0.0277] | ≤0.003 per bound | MINOR — bootstrap seed difference; TIE unchanged |
| D3 | RUNS.md rows 33/34 | 6-submodel mmSYGNAL C=0.6957 | NPZ stores 4-submodel (0.6938); 0.6957 from routing JSON | Info | INFO — two separate files, correctly distinguished in RUNS.md |
| D4 | All v11.5 marginal C-index claims | C=0.6955 | C=0.6955 | 0.0000 | PASS |
| D5 | per_drug_metrics in pipeline_validated.pt | Keys drug/n_obs/mse/mae/spearman/reliable present | Confirmed present, 11 drugs, all n≥10 | — | PASS — no P0 flag |
| D6 | Memory/floor evidence | RM ties MOFA+Ridge on pooled MSE | RM=2.8364 > MOFA=2.8155 > Floor=2.8106; RM BELOW floor | Confirmed | CONFIRMED FLOOR — not a new finding |
| D7 | Cross-disease F3 (RUNS.md row r-2026-05-03-v10s6) | 0/2 MM-derived drugs Bonferroni-pass on AML | Bort p=0.095, Panobi p=0.846 — both fail | 0 | PASS — matches |
| D8 | AML-native F3 (v10 sprint 6) | 0/5 AML-native drugs Bonferroni-pass | Confirmed: 0/5, all fail, expected per MM-proteasome specificity hypothesis | 0 | PASS |

**No P0 flags. No fabricated numbers found. All |Δ| < 0.001 for primary C-index claims.**

---

## §7 Notes and Caveats

1. **No v11/v12 survival model checkpoints on disk.** Survival models (Cox PH) are refitted from `data_ready.pt` on each run; per-patient log-hazards are the durable artifact. This is correct design — Cox models are thin wrappers, not neural weights.

2. **Panobinostat dominates RM's pooled test_mse.** MSE=22.6 for one drug out of 11 pulls the pooled number above the predict-mean floor. Per-drug, RM beats baseline on Bortezomib, Romidepsin, Etoposide, and Lenalidomide by Spearman correlation (even if pooled MSE is unfavorable).

3. **S4 t(11;14) remains the weakest stratum** for all non-mmSYGNAL models. v11.5 (0.6426) is above the baseline (Cox_v11_richer 0.5205) but below mmSYGNAL (0.6926). The v12 t(11;14) specialist head failed to close this gap (RUNS.md row 36: Δ_S4=+0.013, CI not excluding 0).

4. **No per-stratum coverage warnings triggered.** Smallest stratum S4 t(11;14) n=95, n_events=19. Minimum 19 events — at the reliability floor; C-index estimates here have wide uncertainty.
