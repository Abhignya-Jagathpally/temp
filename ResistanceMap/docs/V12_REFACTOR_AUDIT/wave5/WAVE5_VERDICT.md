# WAVE 5 — Chair Verdict

**Date:** 2026-05-07
**Chair:** Wave-5 integrator
**Inputs:** `category_{A..H}_*.md` (this directory).

---

## §1 Executive verdict

Wave 5 does **not** flip the pre-Wave-5 submission posture but hardens venue
conditions and surfaces one panel-elevated clinical-safety emergency.

- **bioRxiv — YES**, conditional on Tier-1 landing in one week: (i) A.1
  sigmoid removal + retrain, (ii) D.4 imaging `NotImplementedError` gate
  before any imaging row flows into a paper table, (iii) F.2/F.3 README
  replacement (live README has `test_mse` 2.3678/2.3699/2.3731 vs recomputed
  2.8106/2.8649/2.8364 — +18-20 % drift, W5.F §F.3). Slip → NOT-YET.
- **Briefings in Bioinformatics / Cell Reports Methods — CONDITIONAL_PASS**,
  sunset 2026-12-31. Adds Tier 2: C.4 IBS kwarg+risk-vs-survival fix, E
  `FittedGuardMixin` test pass, G Option-A audit log, A.5 multiseed Wilcoxon,
  D.3/D.5 honest renames.
- **Nature Methods — NOT YET**, unchanged. Tier 3 gates: external-cohort F_S5
  (W5.H §M1 — no loader exists at all, worse than briefed), G Option-B HMAC,
  and either retire the imaging branch or wire real UNI/GigaPath behind a
  fresh F2 leak audit.

Wave-5 deltas: **D.2 downgraded to docstring-clarification** — DSM in
`landscape/scalar_potential.py:88-101` is mathematically correct under the
declared `−∇U=score, ż=−∇U` convention; only line 91 has a sign-typo docstring
(W5.D §D.2). **W5.H M1/M2/M3 are NOT REPRODUCIBLE.** **D.4 is the worst
remaining blocker:** `TileEncoder.forward()` returns `torch.randn` per call
(W5.D §D.4, `infrastructure/imaging_radiomics.py:55`).

---

## §2 Effective BLOCKER count post-Wave-5

25 → 24 (D.2 downgrade) → **21 actionable** when M1/M2/M3 leave the W5.H
tally because the cited files do not exist
(`evaluation/external_validation.py:262`, `fairness_audit.py:111-114`,
`models/{identifiable_ode,disentangled_vae}.py`; real fairness module
`evaluation/tier_a/bias_fairness.py:146-152` is honest scaffolding).
M4/M5/M6/M7 are reproducible and add 4 engineering items.

| ID | Cat | Severity | Verified at | Action | Effort | Tier |
|---|---|---|---|---|---|---|
| A.1 | head | BLOCKER | `landscape/predictor.py:341` | Remove sigmoid; retrain; regen `paper/tables/baseline_comparison.json` | 0.5 PD | T1 |
| A.2 | head | BLOCKER (downstream A.1) | `paper/tables/baseline_comparison.json` (chair spot-check: Zero=2.8106, RM=2.8364) | `--strict-baselines` tripwire | 0.5 PD | T1 |
| A.3 | head | BLOCKER | `evaluation/baselines.py:60-78` | Downgrade `NotImplementedError` to warning; impl ≥2 of {DeepCDR,GraphDRP,PaccMann,MOLI,DRPreter,KronRLS} | 15-30 PD | T2/T3 |
| A.4 | head | BLOCKER | `config.py:58-61`, `preprocessors.py:619-626` | `split_protocol ∈ {random,loco,lodo,tissue}` + CLI | 2-3 PD | T2 |
| A.5 | head | BLOCKER | README:253 roadmap | `scripts/run_multiseed.py` + paired Wilcoxon | 1-2 PD | T2 |
| B.1 | repro | BLOCKER | `data/__init__.py:17-32`, `tests/test_leakage.py:8` | Land `loaders.py`/`preprocessors.py`/`splits.py` per W5.B inline | 90 min | T1 |
| B.2 | repro | BLOCKER | `reproduce.sh:13,16,27,29` | Rewrite to v0.1 minimum | 15 min | T1 |
| B.3 | repro | BLOCKER | `pyproject.toml:107` | `setuptools.find` + exclude tests/scripts | 10 min | T1 |
| B.4 | repro | BLOCKER | `scripts/generate_latex_tables.py:253-273` | Raise `FabricationError` instead of fall-through | 30 min | T1 |
| B.5 | repro | BLOCKER | `.github/workflows/` absent | Add `ci.yml` + `check_no_fabrication.py` | 1-2 hr | T1 |
| B.6 | repro | BLOCKER | `evaluation/ablations.py` byte-id to `external_validation.py` | `git rm` dupe; rename `ablation_harness.py`; add hash-test | 30 min | T1 |
| C.1 | math | BLOCKER | `post_hoc_calibration.py:79-89` | 1-D branch → numerically-stable `sigmoid(z)`; re-run binary `TemperatureScaling` | 30 min | T2 |
| C.2 | math | BLOCKER | `power_analysis.py:206-207` | `z_α=√2·erfcinv(α)` (1.43 → 1.96) | 15 min | T2 |
| C.3 | math | BLOCKER | `competing_risks.py:111-143` | Cause-specific hazards (Putter 2007); `mixture_prob` → empirical fraction | 1-2 hr | T2 |
| C.4 | math | BLOCKER (active corruption) | `tier_c/forecasting_uncertainty.py:172-194` vs `metrics/survival.py:103,107,182,186` | `eval_times=horizons`; `surv_pred=1-risk`; reshape (N,T); narrow `except` | 1 hr | T2 |
| C.5 | math | BLOCKER | `cytogenetic_stratification.py:418-461` | `EXPERIMENTAL_NOT_VALIDATED=True` raises `NotImplementedError` | 30 min | T2 |
| D.1 | sci | BLOCKER (under "MM SoC" framing) | `config.py:67-71`, `configs/default.yaml:46-66` | Add `soc_2026_rrmm_biologics`/`soc_2026_rrmm_smallmol`; tighten README scope | 30 min | T2 |
| ~~D.2~~ | sci | **docstring only** | `landscape/scalar_potential.py:88-101` | Rewrite line-91 docstring; add 5-line descent unit test | 10 min | T1 (with A.1) |
| D.3 | sci | BLOCKER | `infrastructure/scrna_integration.py:15-58` | Rename → `LinearProjectionPlaceholder`; back-compat `DeprecationWarning`; strip "scGPT/scFoundation" | 30 min | T2 |
| **D.4** | sci | **WORST REMAINING** | `infrastructure/imaging_radiomics.py:55` | `forward → NotImplementedError`; gate `ImagingRadiomicsIntegrator`; `git grep paper/` for any imaging row; AST test | 1 hr | T1 |
| D.5 | sci | BLOCKER | `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py:217-224` | Rename + docstring + RUNS.md row + verdict-doc title; add Wave-3 `pca_substitution.md` | 2 hr | T2 |
| E.1 | clin | BLOCKER (panel-elevated) | `clinical/dynamic_treatment_regimes.py:91-94,708,825,872-917` | Env-gate `RM_RESEARCH_USE_ACKNOWLEDGED`; `@_not_for_clinical_use`; delete `__main__` | ~30 LoC | T1 |
| E.2 | clin | BLOCKER (panel-elevated) | `early_relapse_detection.py:577,673`; `smm_progression.py:438`; `mrd_prediction.py:595`; `emd_prediction.py:609`; `immunotherapy_response.py:536/638/673`; `treatment_conditioned.py:404` | Land `clinical/_safety.py` (`FittedGuardMixin`+`@requires_fitted`); mix into 7 classes; add `tests/test_clinical_safety.py` | ~80+150 LoC | T1 |
| F.1 | writ | BLOCKER | only `.tex` is the 8-method results table | 6-phase 3-6 PW manuscript sprint (W5.F §F.1) | 3-6 PW | T2 (parallel) |
| F.2 | writ | BLOCKER | README missing DModel/mmSYGNAL/RSF/DeepSurv/SCOPE/TrajectoryNet/MMRF C=0.6955/Bonferroni m=8 | Paste W5.F's full ~270-line README body | 10 min paste | T1 |
| F.3 | writ | BLOCKER | README:161-168 vs `baseline_comparison.json` (chair spot-check confirms drift) | Bundled with F.2; corrected 7-row table | bundled | T1 |
| G.1 | verif | BLOCKER (architectural) | `agents/base.py:137-144,189-215`; `verification/zero_trust.py:26-28,550-557,592-598` | Option A only now: `verification/audit_log.py` + JSONL append; strip "blockchain/immutable/even if compromised" | 1 PD | T1 |

W5.H reproducible items folded into Tier 2 engineering: **M4** (`tracer.py`
async-unsafe span stack at 227,293-296,327-329 → `contextvars.ContextVar`),
**M5** (delete `optimizer.py` dead `record_*`/`suggest_optimizations` loop;
queue v13 LinUCB), **M6** (`scheduler.py:216-224` placeholder + never started
from `main.py:159` → implement loop + start/stop), **M7** (delete silent-stub
MCP connectors at `mcp_integration/connectors.py`; replace with subagent
README pointer). **M8** RUNS.md row hand-off is partial here — the file is
absent from this worktree but tracked in the parent checkout; outcomes are
real per project memory and `docs/V12_*` files.

---

## §3 Ranked patch-application order

### Tier 1 — Land-this-week (minimum bioRxiv-ready set)

Dependency rationale: B.1 unblocks every test; B.3 makes editable install
ship the package; B.4 must precede B.5 (the fabrication sentinel will trip
otherwise); E.1+E.2 are clinical-safety emergencies; D.4 must precede any
push because the imaging branch returns pure noise; A.1+D.2 share a retrain
window; F.2/F.3 are paste-only after the sandbox-block surfaced in W5.F.

1. **B.1** Land `data/{loaders,preprocessors,splits}.py` (90 min) → unblocks
   `pytest tests/test_leakage.py`.
2. **B.3** `pyproject.toml` auto-discovery (10 min).
3. **B.6** Delete duplicate `ablations.py`; rename `ablation_harness.py`
   (30 min).
4. **B.4** Replace fabricated-AUROC fall-through with `FabricationError`
   (30 min).
5. **B.2** v0.1 `reproduce.sh` (15 min).
6. **B.5** `ci.yml` + fabrication sentinel (1-2 hr; depends on B.4).
7. **E.2** `clinical/_safety.py`, mix-in into all 7 predictors,
   `tests/test_clinical_safety.py`.
8. **E.1** Env-gate + decorator on DTR; delete `__main__` demo.
9. **D.4** `TileEncoder.forward → NotImplementedError`; gate
   `ImagingRadiomicsIntegrator`; AST `tests/test_no_random_in_forward.py`.
10. **A.1 + D.2** in one retrain window: remove sigmoid at
    `landscape/predictor.py:341`; rewrite `scalar_potential.py:91` docstring
    + DSM unit test; regenerate `baseline_comparison.json`.
11. **A.2** `--strict-baselines` tripwire in `validate_pipeline`.
12. **F.2 + F.3** Paste W5.F's README body into `README.md`.
13. **G.1 Option A** `verification/audit_log.py`; append from `_make_result`;
    strip over-promising docstrings.

### Tier 2 — Pre-CRM submission (sunset 2026-12-31)

Dependency rationale: C.4 corrupts metrics that are *currently reported* and
must land before CRM Methods is finalized. A.4/A.5 require A.1's retrain to
be meaningful. D.1/D.3/D.5 are honest-renaming with paper-language reach.

1. **C.4** IBS/td-AUC kwarg fix + risk→survival + reshape; narrow `except`;
   re-run all Tier-C `forecasting_uncertainty_agent` evals.
2. **C.1** 1-D `_softmax → sigmoid(z)`; re-run binary calibration.
3. **C.2** `z_α = √2·erfcinv(α)`.
4. **C.3** Cause-specific hazards reparameterisation.
5. **C.5** `MAMLLearner` gate.
6. **A.4** `split_protocol` + CLI.
7. **A.5** `run_multiseed.py` + per-drug-MSE persistence + Wilcoxon.
8. **D.1** SoC annotations + README scope tighten.
9. **D.3** `LinearProjectionPlaceholder` rename + back-compat shim.
10. **D.5** Rename mofapy2 script; create `wave3/pca_substitution.md`.
11. **F.1** Manuscript W1-W3 (1.5-3 PW to Discussion).
12. **W5.H M4** `contextvars.ContextVar` rewrite + 32-span `asyncio.gather`
    stress test.
13. **W5.H M5** Delete optimizer dead loop; v13 LinUCB ticket.
14. **W5.H M6** `_check_completed_jobs` body + start/stop in `main.py:159`.
15. **W5.H M7** Delete in-process MCP connectors; 5-line README pointer.
16. **W5.H M8** Coordinate RUNS.md row enumeration into rewritten README
    (Sprint 5, T1114, MOFA-vs-v11.5, Geneformer F2-leak, paired-C-index N=29).

### Tier 3 — Pre-Nature-Methods

1. **G.1 Option B** HMAC-chain audit log behind `RM_VERIFICATION_MODE=hmac`;
   key in CI secret store; rotation policy.
2. **G.1 Option C** Rekor / OpenTimestamps anchoring; `verify_run.py`; OIDC.
3. **A.3 (real)** ≥2 SOTA baselines (DeepCDR + GraphDRP) on identical splits.
4. **D.3 (real)** Genuine Geneformer/scGPT integration with F2 leak audit.
5. **D.4 (real)** Real UNI/GigaPath integration: WSI tiling + MahmoodLab DUA
   + F2 audit (1-2 weeks + compute).
6. **W5.H M1 (real)** External-cohort loaders for IFM2009 / Myeloma XI /
   HOVON-65 / MMRC.
7. **F_S5 external replication** — bioRxiv→Nature Methods gating.

---

## §4 Honesty corrections to surface in the manuscript (per W5.F §F.3)

These textual corrections must appear in any abstract / README / paper draft.

1. **+19 % baseline drift.** README's 2.3678/2.3699/2.3731 had no on-disk
   provenance; recomputed JSON gives 2.8106/2.8649/2.8364 (drift +18.0/+20.0/
   +18.7 %). Earlier numbers are removed (W5.F §F.3).
2. **Predict-mean floor result for Study A.** RM ranks 3rd of 7 by pooled
   drug-response MSE; predict-mean wins by Δ=+0.0258; MOFA+Ridge ranks 2nd
   (2.8155). The deep stack is not yet justified at the available data scale
   (622 train cells; ~340 obs/drug). W5.F §F.3 row D6: "CONFIRMED FLOOR — no
   fix possible without architectural change."
3. **mmSYGNAL beats DModel in 3 of 5 strata (S2, S4, S5).** The marginal-C
   tie holds because S1+S3 advantages offset S2+S4+S5 deficits. Any abstract
   quoting marginal C must accompany it with the per-stratum table.
4. **S4 t(11;14) safety flag.** DModel S4 C=0.643 is the lowest of any model;
   mmSYGNAL routed at 0.693 wins by 5 pp. v12 BCL2-family specialist STRICT
   FAIL (Δ_S4=+0.0135, p=0.446); Sprint C Touzeau pre-Cox refute (BAX wrong
   direction; only BCL2L1 cleared |d|>0.5). DModel must NOT route
   BCL2-targeted therapy until F_S4_PASS triggers on independent paired
   BH3-profiling.
5. **Niche-A "to our knowledge" hedge mandatory.** 0/11 audited papers
   satisfy Q1+Q2+Q3 (numeric threshold + published failure + pre-registration).
6. **DModel is curated-feature engineering + a falsification framework, NOT
   a new architecture.** Stripped of `mmsygnal_routed` it collapses to
   v11_richer (C=0.6538).
7. **mofapy2 here is single-view PPCA-with-ARD on log1p(TPM).** PCA(K=29)
   reproduces the +0.0217 lift in <1 s vs mofapy2's ~77 s. "MOFA+ multi-omics"
   framing is removed (W5.D §D.5).
8. **Imaging branch returns `torch.randn`.** Every `paper/` row that flowed
   through `ImagingRadiomicsIntegrator` must be removed or annotated
   "PLACEHOLDER" (W5.D §D.4).

---

## §5 v13 sprint plan post-Wave-5 (≤10 items)

1. External-cohort F_S5 replication: build `evaluation/external_cohorts.py`
   (W5.H §M1 patch); initiate IFM2009 DUA (rank-1); ship
   `verify_external_replication.py`. Gates Nature Methods.
2. Real SOTA baselines for Study A: DeepCDR + GraphDRP minimum (A.3),
   identical splits, paired-bootstrap p-values.
3. G.1 Option B → C migration: HMAC chain default + Rekor anchor on run
   completion; required for any "non-repudiable" claim.
4. Real foundation-model integration — only one of pathology/scRNA. Recommend
   pathology UNI (D.4 is currently the worst data hazard); F2 leak audit on
   embeddings before any number flows.
5. Multi-view MOFA+ on a paired modality (CCLE proteomics+RNA, or per-segment
   CNV joined to expression) — replaces single-view placeholder (D.5).
6. Real bandit in `agentops/optimizer.py`: LinUCB or Thompson over
   `{agent_choice × prompt_template}` (M5 v13 ticket).
7. NUTS HMC PPC (chair Wave-4 Sprint B) — completes CRM evidence base.
8. Manuscript W4-W6 (F.1) — abstract + Introduction + senior-review buffer.
9. `split_protocol=lodo` headline rerun: A.4 lands T2 but a v13 cycle is
   needed to publish 11-fold mean±std (A.4 risk note).
10. Real `LatentComputeScheduler` wired to VAE pretrain (M6), enabling
    parallel multiseed runs for A.5.

---

## §6 Push readiness

**Safe to push now (audit reports, documentation only):**
`docs/V12_REFACTOR_AUDIT/wave5/category_{A..H}_*.md` and
`WAVE5_VERDICT.md` (this file). They are documentation, no source change,
and accurately describe the current on-disk state including the predict-mean
floor result, the +19 % baseline drift, and the t(11;14) safety flag — they
**are** the correction artifact.

**NOT safe to push (any source patch without per-PR verification):** all A-G
inline patches in worker reports were authored under sandbox no-source rules
and have not been applied or tested. The W5.F README replacement is
paste-ready but should pass senior-author review against `RUNS.md` rows in
the parent checkout. The W5.E `tests/test_clinical_safety.py` imports seven
clinical modules; until the mix-in is applied to all seven the test will
fail.

**Ordered checklist for code-only push (per-patch PRs):**

1. **PR-1 (B.1)** Land `data/{loaders,preprocessors,splits}.py`. Acceptance:
   `pytest -x tests/test_leakage.py` passes.
2. **PR-2 (B.3, B.6, B.4, B.2)** pyproject auto-discover; rename ablation +
   delete duplicate; replace fabricated-AUROC fall-through; v0.1
   `reproduce.sh`. Acceptance: hash-test on byte-identical files passes.
3. **PR-3 (B.5)** CI workflow + fabrication sentinel. Acceptance: green run.
4. **PR-4 (E.2 + E.1)** `_safety.py` + tests + 7 mix-ins + DTR env-gate +
   delete `__main__`. Acceptance: `tests/test_clinical_safety.py` passes;
   `import resistancemap.clinical` without ack env-var raises.
5. **PR-5 (D.4)** TileEncoder gate + AST test. Acceptance: AST test passes;
   no `paper/` table flows through `ImagingRadiomicsIntegrator`.
6. **PR-6 (A.1 + D.2)** Remove sigmoid; rewrite docstring + DSM unit test;
   regenerate `baseline_comparison.json`. Acceptance: regenerated JSON
   committed; `--strict-baselines` (A.2) passes on a fresh run.
7. **PR-7 (F.2 + F.3)** Replace `README.md` with W5.F body. Acceptance:
   `scripts/check_docs_consistent.py` passes against PR-6's JSON.
8. **PR-8 (G.1 Option A)** `verification/audit_log.py` + `_make_result`
   integration + de-promised docstrings. Acceptance: smoke run produces
   `logs/agent_audit_log.jsonl` with one record per agent.

PR order: PR-1 → PR-2 → PR-3 → PR-4 → PR-5 → PR-6 → PR-7 → PR-8. PR-7
must follow PR-6 so README's recomputed numbers match the JSON committed.

---

## §7 Open questions

1. **Path mismatch for A.1.** The user-supplied delta cites
   `resistancemap/training/predictor.py:341`; W5.A §A.1 cites
   `resistancemap/landscape/predictor.py:341`; chair spot-check finds the
   sigmoid at `landscape/predictor.py:340-341` and `training/predictor.py`
   does not exist. **W5.A is correct; the delta has a path typo. Discard the
   delta's path-correction language.**
2. **Whether the missing W5.H files (M1/M2/M3) come from a different
   snapshot.** Chair cannot resolve from this worktree. The original
   audit-ticket author should re-anchor against an explicit branch+commit;
   if no v6/v10/v11/v12 HEAD contains them, formally retract from BLOCKER
   tally (W5.H header).
3. **Whether W5.F's README sandbox-block was environmental or intentional.**
   Chair has Write permission in this session. If a policy-level pre-commit
   hook blocks README writes pending senior-author review, queue PR-7 as
   draft.
4. **RUNS.md row enumeration target.** RUNS.md is tracked in the parent
   checkout but absent from this worktree (`find . -iname 'RUNS*'` → 0).
   Sprint 5 / T1114 / MOFA-vs-v11.5 / Geneformer F2-leak / paired-C-index
   outcomes are real per project memory. Recommend folding into README per
   W5.H §M8 (CI grep-check already proposed) rather than a separate
   `RELEASE_NOTES.md`.
5. **D.5 PCA-substitution Wave-3 doc.** W5.D §D.5 references
   `wave3/pca_substitution.md` as TODO; the user's memory says Wave 3 already
   addressed this but the file does not exist in this worktree. Verify
   against `wave3/WAVE3_VERDICT.md` content before D.5 docstring rewrite
   references the path.
6. **A.5 baseline-comparison schema.** A.5 patch needs per-drug MSE vector in
   `baseline_comparison.json`; current schema only has aggregate `test_mse`.
   The exact location of `scripts/run_baselines_real.py` was not verified
   by W5.A — open a sub-issue under PR-6.
7. **F.1 senior-author review buffer.** W5.F §F.1 estimates 1.5 PW for W6;
   chair cannot tighten without senior-author availability data.

**End of Wave-5 chair verdict.**
