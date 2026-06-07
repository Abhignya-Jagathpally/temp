# ResistanceMap v20 — Integrated Code-Change Plan

**Chair:** integrating reviewer. **Branch:** v20. **Date:** 2026-06-07.
**Companion:** `COMPARISON.md` (verified present in this directory).

This is the de-duplicated, prioritized, file-level plan merged from four specialist
reviews (dag, harmonize_bug, survival, governance, repr). Items already committed
per CONTEXT (Phase 0 bugs, Phases 1.2–6 loaders/models, the four committed DAG fixes)
are **dropped**. Overclaiming items are demoted or struck. No ResistanceMap metric is
asserted beyond CONTEXT: the only owned numbers are the GDC MMRF-COMMPASS clinical-only
OS baselines (ElasticNet-Cox 0.664 [0.590,0.739], CoxPH 0.646 [0.578,0.710], RSF 0.557,
GBM 0.555; N=994/191 events; 2 features, age-dominated) and the BeatAML matrix shape
(510×60,483, no ex-vivo AUC). Everything else is engineering-status, not a result.

Each item: **dimension** · **grounding finding** · **runnable-now vs data/dep-blocked**.

Verified against on-disk files this session:
- `scripts/mortfm_harmonize_identifiers.py` (L44–53, L94–109): empty-`feature_genes` → `rows=[]` → `pd.DataFrame([])` has no `tier` column → `mapping["tier"]` at L109 raises `KeyError: 'tier'`. The "STRING-alias path lacks tier" framing is a mislabel; both deep reads converged on the empty-DataFrame cause.
- `dags/mortfm_evidence_pipeline.py` (L142, L201–202): `acquire_public >> harmonize_ids`; then `harmonize_ids >> [ingest_string, ingest_uniprot, ingest_reactome, ingest_chembl] >> build_bio_graph` — confirmed inversion (harmonize reads `protein_nodes.csv` + `string_aliases.parquet` that those ingests produce).
- `resistancemap/mortfm/longitudinal/clinical_features.py:171` `encode_longitudinal_labs()` exists; default LENS path uses `encode_for_lens` (L120, 5 legacy features). Expanded labs are NOT in the default path.

---

## P0 — Correctness blockers (no run is trustworthy until these land)

### P0-1 — Fix ISSUE 6: harmonize KeyError `'tier'` (fail-closed, not crash)
- **Dimension:** harmonize / honesty-gate.
- **File:** `scripts/mortfm_harmonize_identifiers.py` L106–109.
- **Grounding:** When all feature parquets are missing/skipped (L47–48 warn-and-skip), `feature_genes` is empty, the L95 loop never appends, so `rows=[]` and `pd.DataFrame([])` has no `tier` column; L109 `int((mapping["tier"] > 0).sum())` raises `KeyError`. This is reachable on a clean run because of P0-2 (ingests run after harmonize).
- **Change:** Construct with explicit schema and fail loud before the count:
  ```python
  mapping = pd.DataFrame(rows, columns=["feature_gene", "uniprot_id", "tier", "source"])
  if mapping.empty:
      # write a fail-closed gate artifact, then non-zero exit
      _write_report(args.report, coverage=0.0, n_mapped=0,
                    gate_pass=False, error="no feature genes resolved; "
                    "ensure ingest_uniprot + ingest_string + RNA/BeatAML ingests "
                    "run BEFORE harmonize (see P0-2 / ISSUE 3)")
      return 1
  ```
  Keep the existing `return 1` gate-fail semantics; do NOT emit a misleading `0.0` coverage with exit 0.
- **Status:** Runnable now. Low risk (additive guard; no happy-path change). Surfaces, not masks, the real upstream dependency.

### P0-2 — Fix ISSUE 3: reorder bio-knowledge ingests + build_biological_graph BEFORE harmonize
- **Dimension:** dag.
- **File:** `dags/mortfm_evidence_pipeline.py` L142, L201, L202 (and verify L224, L259–260, L318, L337, L388).
- **Grounding:** `harmonize_ids >> [ingest_string, ingest_uniprot, ...]` (L201) inverts the real edge: `mortfm_harmonize_identifiers.py:56,67` read `protein_nodes.csv` (produced by `ingest_uniprot`) and `string_aliases.parquet` (produced by `ingest_string`).
- **Change:**
  - Delete L201 edge `harmonize_ids >> [ingest_string, ingest_uniprot, ingest_reactome, ingest_chembl]`.
  - Add `acquire_public >> [ingest_string, ingest_uniprot, ingest_reactome, ingest_chembl]`.
  - Add `[ingest_string, ingest_uniprot] >> harmonize_ids` (harmonize strictly needs only string_aliases + protein_nodes; do not force it behind `build_bio_graph`).
  - Keep L202 `[ingest_string, ingest_uniprot, ingest_reactome, ingest_chembl] >> build_bio_graph`.
  - Re-verify no cycle (`acquire_public` is already upstream of `harmonize_ids`, so acyclic) and that `compute_esm2` (L337) and the `harmonize_ids >> {prepare_mmrf, ingest_geo_sc, download_gdc_open, download_geo_scrna, ingest_beataml}` edges still resolve — they do, since harmonize still runs, just later.
- **Status:** Runnable now (DAG-topology edit). Low/structural risk; Airflow-graph cycle re-check is the only verification gate.

### P0-3 — Guard harmonize input reads with actionable FileNotFoundError
- **Dimension:** harmonize.
- **File:** `scripts/mortfm_harmonize_identifiers.py` L56, L67/76.
- **Grounding:** `pd.read_csv(args.uniprot_nodes)` (L56) and `pd.read_parquet(args.string_aliases)` (L67) have no existence guard, unlike the feature loop (L47–48). Standalone runs via `scripts/mortfm/01_harmonize_identifiers.py` (runpy shim) die with a bare pandas traceback.
- **Change:** Add `Path(...).exists()` checks raising `FileNotFoundError("run mortfm_ingest_uniprot.py / mortfm_ingest_string.py first")`, matching the project's established error convention.
- **Status:** Runnable now. Low risk; pure error-message improvement.

### P0-4 — Wire `encode_longitudinal_labs` into the default model path (the central thesis signal)
- **Dimension:** survival / repr (the lab-first contribution itself).
- **Files:** `resistancemap/mortfm/longitudinal/clinical_features.py:171` (`encode_longitudinal_labs`, exists);
  `scripts/mortfm/10_run_baselines.py` (`clinical_cols` detector ~L1155 recognizes only the 5 legacy columns);
  the canonical LENS feature assembly that currently calls `encode_for_lens` (5 features).
- **Grounding:** CONTEXT marks the expanded lab panel + `encode_longitudinal_labs()` as ADDITIVE-ONLY and NOT wired into the default path; the canonical model still defaults to the original 5 clinical features. The ONLY survival numbers we own (clinical-only OS) use 2 features (age+gender). The lab-first thesis is presently un-benchmarked.
- **Change (additive, presence-guarded):**
  - Broaden `clinical_cols` detection in `10_run_baselines.py` to include the `D_LAB_*` columns when present.
  - Add an explicit `clinical_labs_cox` benchmark arm so the expanded panel is actually fit and reported, distinct from the `clinical_only` arm.
  - Leave the two trajectory models (`pangea_landmark_cox`, `ferle_lstm_crbm`) honestly skipping on cross-sectional data — do not force them.
  - Until a real lab-panel result exists, any report line must label it engineering-status (see P1-3).
- **Status:** Code runnable now; **producing a number is data-blocked** (requires the longitudinal lab table the MMRF/Virtual-Lab path emits — not on disk). Guard for column presence so cross-sectional runs do not break.

---

## P1 — Validity & honesty (required before any "beats clinical-only" or scientific claim)

### P1-1 — Add nested CV to the survival cascade (single split inflates/destabilizes C-index)
- **Dimension:** survival. **Files:** `scripts/run_first_results.py` (single `train_test_split`, L324); `scripts/mortfm/10_run_baselines.py` (one fixed `split` column).
- **Grounding:** zero `KFold`/nested-CV in either file; with 191 events a single 30% split (~57 test events) makes the 0.557–0.664 spread fragile, and EN-Cox `l1_ratio`/`alpha` are fixed with no inner-loop selection — so "EN beats RSF" is currently a single-draw artifact. The owned CI (0.664 [0.590,0.739]) is wide.
- **Change:** Outer `GroupKFold` (group=`submitter_id`) + inner fold for any tuned hyperparameter; report mean ± across-fold SD; add `--cv-folds` (default 5×3) to `10_run_baselines.py`.
- **Status:** Runnable now on the GDC clinical cohort (it exists). Medium compute risk.

### P1-2 — Replace sigmoid-proxy IBS with true IPCW IBS or `None`
- **Dimension:** survival / governance. **File:** `scripts/mortfm/10_run_baselines.py` L781–818.
- **Grounding:** `_integrated_brier_score` sigmoid-maps risk and ignores censoring — its own docstring says "Simplified IBS approximation." Reporting it under the column "IBS" mislabels a non-calibration scalar, exactly what the claim-gate harness exists to prevent.
- **Change:** For models exposing survival curves (sksurv Coxnet with `fit_baseline_model=True`, RSF, GBM), compute `sksurv.metrics.integrated_brier_score` on a shared grid. For constant/linear predictors emit `ibs=None, ibs_method="none"` — never the proxy.
- **Status:** Runnable now (dep-gated on sksurv importability — see Open Q1). Low–medium risk.

### P1-3 — Wire the claim-gate layer into the human-facing report (governance is dead code)
- **Dimension:** governance. **Files:** `scripts/generate_report.py` (no import of `resistancemap.governance`); `resistancemap/governance/{claim_gates.py, endpoint_validator.py}`.
- **Grounding:** `endpoint_validator.py:106–111` already encodes "OS is NOT a resistance endpoint." But `generate_report.py` never calls it, so nothing prevents drift into resistance/trajectory language. The only owned result is clinical-only OS → `survival_head_technical_validation` grade only. PK-SSM/scVI sections assert capability with no gate (`generate_report.py` ~L61–97).
- **Change:**
  - In `generate_report.py`, build a `RunEvidence`, call `run_gates(...)` + `validate_endpoint_semantics(endpoint_name="overall_survival", ...)`, and gate every "proves" line on `claim_levels_allowed`; print `blocking_reasons` verbatim.
  - Split the proven list into **ENGINEERING (runs, untrained)** vs **SCIENTIFIC (endpoint-validated)**; PK-SSM/scVI/expanded-lab go in ENGINEERING until a gate-passing endpoint exists.
  - Read `logs/mortfm/claim_gate_report.json` if present; if absent print "all scientific claims UNVERIFIED this run" (fail-closed).
  - Emit one standing line: `MAX SCIENTIFIC CLAIM THIS RUN: <gate level>` plus SOTA context cited as external/published (PANGEA-SMM C≈0.79 N≈2344; Ferle AUROC≈0.78@3mo N≈875) — never as ours.
- **Status:** Runnable now (governance modules + the OS baseline artifact exist). Low risk; degrade-to-UNVERIFIED on missing config.

### P1-4 — Pre-register the "beats-clinical-only" bar with a paired significance test
- **Dimension:** survival / governance. **File:** `scripts/mortfm/10_run_baselines.py` (`_build_claim_comparison` ~L1422; `CLAIM_LEVEL_BASELINE_REQUIREMENTS` ~L700).
- **Grounding:** today the harness compares bare point means (`must_beat_max_ci` from a mean, no test). The thesis is PARITY, not a win; the clinical-only CI [0.590,0.739] is wide — overlapping it must not count as "beats."
- **Change:** Add `compute_beats_bar()` requiring (a) candidate mean C-index > strongest must-beat mean AND (b) a paired bootstrap CI on the C-index *difference* excluding 0; emit boolean `beats_clinical_only` + CI. Use a C-index-specific wrapper, not the squared-residual `paired_bootstrap_delta`.
- **Status:** Runnable now (on GDC cohort). Low risk.

### P1-5 — Short-circuit the MMRF→LENS subgraph so the open-access baseline cascade still runs
- **Dimension:** dag / survival. **File:** `dags/mortfm_evidence_pipeline.py` L388 (MMRF chain), L449 (`audit_leakage >> run_baselines`), L455 (`download_gdc_open >> gate_revalidate`).
- **Grounding:** CONTEXT HARD BLOCK: no MMRF outcome data. `prepare_mmrf` (L388) will `FileNotFoundError`, cascading to `run_baselines` (only wired off `audit_leakage`, L449). So the one part with real numbers (GDC clinical-only OS) cannot fire in open-access-only mode; `download_gdc_open` is NOT wired to `run_baselines`.
- **Change:** Add a `ShortCircuitOperator` (mirroring the committed `check_cellline` pattern) over the longitudinal/LENS/survival subgraph keyed on MMRF dataset+split-manifest presence; add `download_gdc_open >> run_baselines`; set `run_baselines` `trigger_rule=TriggerRule.NONE_FAILED` so a skipped MMRF chain does not skip baselines. Do NOT fabricate MMRF inputs.
- **Status:** Runnable now (DAG edit). Medium risk — verify `run_baselines` does not fire before the GDC split manifest exists (Open Q2/Q3).

---

## P2 — Coverage, legibility, robustness (not blocking)

### P2-1 — Add real RNA-seq features to `run_on_gdc_rnaseq` (currently age+gender despite its name)
- **Dimension:** survival / repr. **File:** `scripts/run_first_results.py:453`.
- **Grounding:** the function named "rnaseq" only builds `X=[age_scaled,is_male]`; the committed `gdc_open_loader.py` (`query_rnaseq_manifest/build_expression_matrix/merge_expression_clinical`) is never called. The script's own "Next Steps" (L552) requests STAR-Counts as features.
- **Change:** Behind `--with-rnaseq` + on-disk cache, assemble STAR-Counts → log1p → top-N variance genes / PCA → join clinical; add as a feature arm separate from `clinical_only`.
- **Status:** Data/dep-blocked (large download; sksurv import — Open Q1). Medium risk; keep offline clinical path intact.

### P2-2 — Fix bootstrap RNG reuse + event-count guards in `run_first_results.py`
- **Dimension:** survival. **File:** `scripts/run_first_results.py` L416–421.
- **Grounding:** RNG re-created with the same `seed` inside the per-model loop → identical resample indices across models (incidental, not declared-paired); `<5 events` skip silently drops draws (N<200). C-index reported even when test events tiny.
- **Change:** Seed RNG once (or per `(model,b)`); record `n_effective_boot`; emit `status="insufficient_events"` when test events `< k` (e.g. 10), mirroring existing skip rows.
- **Status:** Runnable now. Low risk.

### P2-3 — Integrate the representation chain via the committed modules (stop the parallel/fork)
- **Dimension:** repr. **Files:** `scripts/run_scvi_integration.py`, `scripts/run_pkssm_forward.py`; `resistancemap/models/{scvi_encoder.py, pk_observation_decoder.py, mechanism_classifier.py}`; `pk_state_space.py`.
- **Grounding:** committed `scvi_encoder`/`pk_observation_decoder` are referenced only in docstrings; runners inline their own gene lists and use a parallel `pk_state_space.PKSSM` whose `MechanismClassifier` duplicates/conflicts with the committed one. `run_pkssm_forward.py:118` uses `torch.randn` fallback and its JSON `note` admits "Mechanism assignments are random initialization"; L131 fabricates a time axis by `.expand(-1,6,-1)`. This is a parameter-count demo, not the thesis chain.
- **Change:** New `scripts/run_repr_chain.py`: `SCVIEncoder → ChromatinODE (20 reader/writer genes already align) → GraphEnergyResistanceSDE+HittingTime (via existing LatentToGraphProjector) → PKObservationDecoder → committed MechanismClassifier (8 classes + abstention)`. Demote `pk_state_space.PKSSM` to a baseline. Reconcile the duplicate `MechanismClassifier` to the committed 8-label set.
- **Status:** Dep-blocked (scvi-tools + scikit-misc not installed) and partially data-blocked (no paired scRNA→serum-lab supervision). Medium–high risk (SDE `d_graph` wiring — Open Q5).

### P2-4 — Declare a real `scbio` extra incl. `scikit-misc`
- **Dimension:** repr / dep. **File:** `pyproject.toml` `[project.optional-dependencies]`.
- **Grounding:** `scvi_encoder.py:48` docstring points at extras group `scbio` that does not exist; `run_scvi_integration.py:120` `highly_variable_genes(flavor="seurat_v3")` needs skmisc. Both imports fail today.
- **Change:** Add `scbio = ["scvi-tools>=1.1,<1.5","scikit-misc>=0.3","scanpy>=1.9.5,<1.11","anndata>=0.10"]`.
- **Status:** Runnable now (manifest edit); actual install is the dep-block. Low risk; avoid scanpy pin collision.

### P2-5 — PK decoder fixes + honest validation emitter + docstring de-overclaim
- **Dimension:** repr / honesty. **Files:** `resistancemap/models/pk_observation_decoder.py` L80–89 (`predict_after` broadcast bug; docstring L8,14–16); new `scripts/run_pk_validation_gse136337.py` (consumed by committed `scripts/mortfm/11_validate_open_access.py` L174–186).
- **Grounding:** `predict_after` `dt.reshape(-1,1)` mis-broadcasts per-biomarker Δt; the decoder docstring claims "validated against GSE136337" but no committed code emits the `pred_{lab}` CSV the harness needs.
- **Change:** Guard the broadcast (`unsqueeze` 1-D, assert trailing dim ∈ {1,n}); add an emitter that writes `sample_id,pred_B2M,pred_Albumin,pred_LDH` (train on scRNA proxy, evaluate held-out — no leakage); rewrite docstring to "trained on scRNA proxy; evaluated by held-out Spearman on GSE136337; MAE scale-relative."
- **Status:** Broadcast fix runnable now; validation emitter data/dep-blocked (decoder untrained; scvi dep). Medium risk (unit scaling — report Spearman as the honest metric).

### P2-6 — DAG/hygiene: lambda callables, harmonize empty-feature guard, allow_exit1 doc
- **Dimension:** dag / harmonize. **Files:** `dags/mortfm_evidence_pipeline.py` L357–375 (lambda callables for `build_snapshots`/`build_temporal_pairs`); `scripts/mortfm_harmonize_identifiers.py` L44–53 (handled by P0-1); `acquire_public allow_exit1` doc note for ingest tasks.
- **Change:** Convert the two `lambda` callables to `python_callable=_run_script, op_args=[...]` for executor-safety/consistency; add `doc_md` on `ingest_string`/`ingest_uniprot` stating they are the real STRING/UniProt presence gate after the P0-2 reorder.
- **Status:** Runnable now. Low risk.

---

## Dropped / demoted from specialist proposals (chair filter)
- **harmonize → call `identifier_mapping.load_identifier_maps()` refactor** (repr/harmonize P1 suggestion): demoted out of P0/P1. The inline Tier-1/5 logic is correct when inputs exist; refactor is scope-creep under the lab-first pivot. Revisit only if omics harmonization is retained.
- **Tier-5 STRING-alias correctness gap** (string_aliases is UniProt-filtered → Tier-5 recovers ~nothing): real but P2-at-most and likely irrelevant under lab-first; not scheduled here.
- **`RunEvidence.feature_space/n_features` field add** (governance P0-3): folded into P1-3/P0-4 reporting rather than a standalone breaking dataclass change; defaulted field only if needed.
- **PKSSM competing-risk `_survival_nll` honesty fix** (repr P1.4): folded into P2-3 (demote PKSSM to baseline); do not assert `n_competing_risks=2` it doesn't deliver.

---

## Open questions (block specific items, listed once)
1. Is **sksurv** importable in the target env (scvi-tools + scikit-misc confirmed absent)? Gates P1-2, P2-1.
2. What exact files (dataset path + split-manifest filename) does the Spark MMRF path emit that `10_run_baselines.py` consumes? Needed for the P1-5 ShortCircuit `exists()` check.
3. Does `download_gdc_open` (`run_first_results.py`) write the split manifest `10_run_baselines.py` expects, or only `results/v20_first_baselines/`? Determines whether the P1-5 added edge suffices.
4. Canonical endpoint — OS (what the 0.664 is on) or IMWG-PFS (committed Phase 0)? OS-only forbids resistance-emergence claims per the endpoint registry; affects P1-3/P1-4 denominator.
5. scVI latent (30-d) → SDE `d_graph` bridge: confirm `LatentToGraphProjector` and the `d_graph` the SDE was constructed with (Bug B19). Gates P2-3.

---

## Uniqueness brief
ResistanceMap v20's defensible niche is **not** discrimination performance — on the only owned
endpoint it is expected to reach **parity, not a win** (clinical-only OS C-index 0.664 [0.590,0.739],
N=994/191, age-dominated; no trained lab-first or omics model result exists yet). What no cited
comparator does, and what this plan operationalizes, is the *measurement model itself*: routine labs
are treated as a half-life-filtered, assay-noisy projection of a latent resistance state, and a
**PK-constrained observation decoder inverts that measurement** (`pk_observation_decoder.py`) rather
than fitting labs at face value (contrast Ferle's LSTM+CRBM, mmSYGNAL, PANGEA-SMM). Relapse is cast as
**basin-escape on a Waddington-SDE with hitting-time as the hazard**, a mechanism classifier
decomposes risk into named MM programs with an explicit abstention class, and a **claim-gate harness
mechanically refuses unearned claims** (governance layer wired in P1-3; OS capped at
`survival_head_technical_validation`). The contribution is therefore measurement-inversion +
legibility + falsification discipline — a *how-we-know* claim, audited by the same governance code that
blocks calling an OS model a resistance predictor. This brief asserts no metric beyond CONTEXT and is
side-by-side defensible only as a capability contrast, not a performance win.
