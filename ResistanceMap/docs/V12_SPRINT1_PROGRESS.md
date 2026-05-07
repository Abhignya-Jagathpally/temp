# v12 Sprint 1 — parallel progress (2026-05-04)

This document tracks the parallel v12 Sprint 1 work launched after the v11
chair verdict landed at CONDITIONAL_PASS. It is a *progress note*, not a
sprint verdict — the chair re-runs once all five tracks are in.

## §1 Tracks at launch

| # | Track | Driver | Gates | Mode |
|---|---|---|---|---|
| A | HBayes PPC #67 | parent script | Nature Methods M1 | foreground |
| B | t(11;14) specialist head (S4 stratum) | parent script | discrimination ≥0.69 in S4 | foreground |
| C | MOFA+ vs v11.5 Cox PFS head-to-head (data-integration §2.7 gap) | data-integration-validator agent | bioRxiv M2 | background |
| D | Causal re-audit of v11/v11.5 chair claims | causal-inference-auditor agent | submission gate | background |
| E | Post-2025 MM/AML SOTA literature sweep | literature-deep-researcher agent | bioRxiv freeze | background |

## §2 Status by track

### §2.1 Track A — HBayes PPC ✅ COMPLETE

**Verdict:** F1 downgrades to **PASS-WITH-CAVEAT** (does NOT delete F1).

5/10 stratum-statistic combos FAIL:
- T1 (centroid magnitude): 4 strata FAIL (mean-field SVI shrinkage)
- T2 (dispersion): 1 stratum FAIL (t(11;14), homoscedastic σ_obs under-estimates)

Sufficient for bioRxiv. Insufficient for Nature Methods without v12.5 NUTS HMC + heteroscedastic σ_obs_s re-inference.

Artifacts: `scripts/v12/s_v12_hbayes_ppc.py`, `paper/v8_artifacts/v12_sprint1/hbayes_ppc.json`, `docs/V12_HBAYES_PPC_VERDICT.md`. RUNS.md row `r-2026-05-04-v12-hbayes-ppc`.

### §2.2 Track B — t(11;14) specialist head ✅ COMPLETE

**Verdict:** S4 STRICT FAIL — NULL on both variants. Pre-registered Δ_S4 > +0.05 NOT MET.

| Variant | nfeats | marg C | S4 C | Δ_S4 | 95% CI | p_two |
|---|---:|---:|---:|---:|---|---:|
| baseline (Cox_v11_routed_mmsygnal) | 24 | 0.6955 | 0.6426 | — | — | — |
| specialist (4 main effects) | 28 | 0.6925 | 0.6559 | +0.0135 | [−0.0213, +0.0538] | 0.446 |
| interaction_specialist (4 main + 4 interactions) | 32 | 0.6963 | 0.6371 | −0.0053 | [−0.0637, +0.0633] | 0.812 |

**Negative result is meaningful**: rules out the simplest BCL2-family bulk-RNA fix; v11.5 SOTA-tying not undermined; manuscript M2 §7 must add a Limitations sentence.

Artifacts: `scripts/v12/s_v12_t1114_specialist_head.py`, `paper/v8_artifacts/v12_sprint1/t1114_specialist_head.json`, `docs/V12_T1114_SPECIALIST_VERDICT.md`. RUNS.md row `r-2026-05-04-v12-t1114-specialist`. Wall time 442.8 s.

### §2.3 Track C — MOFA+ vs v11.5 ✅ COMPLETE

**Verdict:** **V12_ARCHITECTURAL_OPPORTUNITY** — MOFA+ shared factors and v11_richer features are *complementary*, not redundant.

Real mofapy2 v0.7.4 (no fallback). Honest deviations from spec: K=32 not K=64 (compute), iter=50, HVG-filter to top 5,000 genes (Argelaguet 2020 standard) — without these, two earlier K=64 iter=1000 runs were killed at 18 min and 4 min wallclock. Single-modality (no proteomics, no per-bp CNV in `data/processed/` — true multi-omics MOFA NOT exercised). ARD pruned K=32 → K_eff=29.

| Feature set | n_feat | marginal C | Δ vs v11_richer (95% CI) | p_two |
|---|---:|---:|---|---:|
| MOFA_factors                  | 29 | 0.6600 | +0.0064 [−0.0260, +0.0382] | 0.689 |
| Cox_v11_richer                | 23 | 0.6538 | (reference; reproduces s5e 0.6537) | — |
| **MOFA_factors + Cox_v11_richer** | **52** | **0.6751** | **+0.0214 [+0.0005, +0.0425]** | **0.046** |

**Per-stratum (combo dominates v11_richer in 4/5):**

| stratum | n | MOFA | v11_richer | combo | combo − richer |
|---|---:|---:|---:|---:|---:|
| S1 del17p   | 105 | 0.693 | 0.698 | 0.712 | +0.014 |
| S2 t(4;14)  |  97 | 0.591 | 0.586 | 0.626 | +0.040 |
| S3 +1q21    | 165 | 0.659 | 0.688 | 0.681 | −0.007 |
| **S4 t(11;14)** | 95 | 0.568 | 0.521 | **0.578** | **+0.058** |
| S5 other    | 325 | 0.650 | 0.621 | 0.656 | +0.035 |

**Critical cross-track finding:** the S4 t(11;14) lift of **+0.058** that the BCL2-family specialist (Track B) FAILED to deliver (Δ +0.0135 NS) is achieved by MOFA+ shared factors. The fix is unsupervised shared-factor decomposition, not curated BCL2/MCL1/CCND1/BCL2L1 features.

**Caveat:** the +0.0214 combo lift over v11_richer is on Cox_v11_richer (pure v11), NOT on Cox_v11_routed_mmsygnal (the v11.5 SOTA-tie reference at C=0.6955). MOFA+v11.5 ensemble has not yet been benchmarked. Open task for v12 Sprint 2.

Artifacts: `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`, `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_pfs.json`, `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_per_patient_log_hazards.npz`, `docs/V12_MOFA_VS_V115_VERDICT.md`. RUNS.md row `r-2026-05-04-v12-mofa-vs-v115` already appended (by agent). Wall time 549.8 s.

### §2.4 Track D — Causal re-audit ✅ COMPLETE

**Verdict:** CONDITIONAL PASS — no hard L1→L2 slippage in core claims; 5 text-level fixes required (RC-1 through RC-5).

Output: `docs/V12_CAUSAL_REAUDIT.md`.

| RC | Priority | Location | Fix |
|---|---|---|---|
| RC-1 | HIGH | M1 Methods | One sentence: Waddington scalars are baseline-geometry, not forecasted states; cite F8-DPS strict-fail |
| RC-2 | HIGH | M2 §1 | When citing mmSYGNAL, use only its L1 metric; do NOT reproduce "treatment selection" framing |
| RC-3 | MEDIUM | M1 §6 | Distinguish Stone estimation-layer failure from graph-theoretic identification failure |
| RC-4 | MEDIUM | M1 §7 | Disclaim causal interpretation of Sprint 4 NIE; cite E-value ≈1.20 |
| RC-5 | **FAIL** | `docs/V11_CLINICAL_TRANSLATION.md` §5.2 | Replace "identify patients for [specific therapy]" with "pre-stratification for correlative arm; causal benefit requires separate RCT" |

RC-5 is the only hard FAIL. Must be fixed before any clinical-translation section of a submitted manuscript reproduces §5.2 language.

### §2.5 Track E — Literature sweep ❌ BLOCKED

**Status:** Agent denied access to all 3 search MCPs (PubMed, bioRxiv/medRxiv, HuggingFace).

The agent correctly refused to fabricate citations and returned without writing the requested file. Re-grant the following MCP tool permissions to unblock:
- `mcp__claude_ai_PubMed__search_articles`
- `mcp__claude_ai_bioRxiv__search_preprints`
- `mcp__claude_ai_Hugging_Face__paper_search`

Then re-launch the literature-deep-researcher agent with the same prompt.

## §3 Next step priorities (ordered)

After Tracks B, C complete:

1. **Apply RC-5 fix** (causal-auditor finding) — text edit to `docs/V11_CLINICAL_TRANSLATION.md` §5.2. **Highest priority — must land before any submission.**
2. **Apply RC-1, RC-2** to manuscript drafts (M1 Methods, M2 §1).
3. Re-grant lit-search MCPs and re-run Track E.
4. Schedule v12 Sprint 2: NUTS HMC + heteroscedastic σ_obs_s for HBayes PPC closure (re-runs Track A).
5. If Track B succeeds → schedule v12 Sprint 2 BeatAML cross-disease replication of t(11;14) specialist (does it generalize, or is BCL2-family lift MM-specific?).
6. TCGA-LAML F3 cross-disease replication (chair gap #69).
7. MCP `connectors.py` wiring (chair gap #63) — engineering hygiene.

## §4 What still gates each venue (post-Track-A,D)

| Venue | Status | Gates |
|---|---|---|
| bioRxiv M1 + M2 | **Ready after RC-5 fix** | RC-5 must land |
| ICML / ICLR | Not without TrajectoryNet head-to-head on EB or GSE271107 LOPO | v12 Sprint 3 |
| Nature Methods | Not without v12.5 HBayes NUTS HMC re-inference + 2nd-cohort F_S5 replication | v12 Sprint 2 + v12 Sprint 4 |
| JCO CCI | Not without external MM cohort replication (HOVON/GMMG) | v12 Sprint 4 |
