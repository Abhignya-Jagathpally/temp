# v11 / v11.5 Chair Verdict — ResistanceMap Integration

**Date:** 2026-05-03
**Author:** integrator (chair, computational-oncology PI persona)
**Inputs read (in order):**
- `RUNS.md` (rows r-2026-05-03-v11s1-ode-diag through r-2026-05-03-v11s5-paired-cindex-test, 12 v11 rows)
- `paper/tables/baseline_comparison.json`
- `paper/v8_artifacts/sota_comparison.md` (v11)
- `paper/v8_artifacts/literature_review_v11.md`
- `docs/V11_SOTA_UPGRADE_PLAN.md`
- `docs/V11_AGENTIC_JUSTIFICATION.md`
- `docs/V11_SPRINT1_ODE_VERDICT.md`
- `docs/V11_SPRINT3_GAT_VERDICT.md`
- `docs/V11_SPRINT5_CONFORMAL_VERDICT.md`
- `paper/v8_artifacts/v11_sprint1/{ode_forward_diagnostic.json,encoder_upgrade.json,f1_f2_gates_v11.json,f1_f2_gates_v11_standardized.json}`
- `paper/v8_artifacts/v11_sprint3/f6_driver_recall_gat.json`
- `paper/v8_artifacts/v11_sprint5/{mondrian_jk_plus_v11.json,cox_with_programs.json,mmsygnal_complete_routing.json,paired_cindex_test.json,mmsygnal_head_to_head.md}`
- `paper/v8_artifacts/v11_sprint7/f8_neural_ode.json`

Pearl-tier ceiling: **L1-with-structural-prior** (pre-registered; not promoted at v11/v11.5).

---

## §1 Executive verdict

**CONDITIONAL_PASS — paper-ready at narrow scope; sunset 2026-08-31.**

Two independent claims, two different evidence bars, two different verdicts:

| Claim | Verdict | Evidence | Defensible venues |
|---|---|---|---|
| **(A) Calibrated 90% prediction-interval coverage with Waddington-feature-tightened intervals on MM PFS, with a pre-registered falsification framework that admits its identifiability bound** | **PASS** at strict thresholds. F_S5 5/5 strata within ±3% of nominal 0.90 (Ridge_v11features, marginal coverage 0.898; v10 0.900). Marginal width tightened by 8 days (1643→1635, 0.5%). F4 PASS, M1 Lyapunov 0/116 violations PASS, F6 30.6σ STRICT PASS (B+1 floor p=0.0001). | bioRxiv: yes. ICML/ICLR (methods): yes if paired with TrajectoryNet/MIOFlow head-to-head on a public benchmark (currently absent). Nature Methods / JCO CCI (clinical): not yet — needs HBayes PPC and an external-cohort replication. |
| **(B) "v11.5 ties SOTA on MM PFS C-index"** (Cox_v11_routed_mmsygnal vs 6-submodel mmSYGNAL, marginal C 0.6955 vs 0.6957, paired z-test p=0.994, B=10000 paired bootstrap p=0.989) | **CONDITIONAL_PASS — PUBLISHABLE BUT NOT NOVEL ON DISCRIMINATION.** Per `paper/v8_artifacts/v11_sprint5/cox_with_programs.json` and §3.11 of the Sprint 5 verdict, **+0.042 of the v11.5 marginal-C gain over Cox_v11_richer is from the mmSYGNAL routed feature; v11 features add only +0.010 over the raw 6 mmSYGNAL scores.** The "TIE" is real on a paired test, but ~95% of the SOTA-tying signal is mmSYGNAL's pretrained feature pipeline, not v11 architecture. | bioRxiv: yes. Methods venue: only if paper title is "v11.5 = v11 calibrated coverage **composed with** mmSYGNAL routed score" — never "v11 matches mmSYGNAL." |

Strict-pass count over the framework (all sources `RUNS.md`): **5 v11 STRICT PASS** (F4 corrected, M1 new, F6 preserved+improved, F_S5 preserved+tightened, paired-C TIE confirmed under both z-test and B=10000 bootstrap) + **2 STRICT FAIL** (F8-DPS at v11 ODE operator, encoder upgrade rejected by F2 zero-trust) across the v11 rows. The strict-fails are *expected refutations* per the v10 spec §1+§11 identifiability-limit thesis and per the v11 plan §2 row 1 zero-trust language — they validate the framework rather than refute it.

Manuscript-readiness — **paper-ready for bioRxiv now, with the §3.11 decomposition foregrounded; not yet ready for Nature Methods or JCO CCI without (a) pre-registered held-out cohort replication and (b) HBayes PPC closure (#67).**

---

## §2 Strength / weakness audit per sprint

### §2.1 Sprint 1 — Neural-ODE forward operator (`docs/V11_SPRINT1_ODE_VERDICT.md`, RUNS rows v11s1-ode-diag, v11s7-ode-dps, v11s1-encoder-reject)

| Axis | Strength | Weakness |
|---|---|---|
| Forward operator | **Replaces v10 single-Euler with adaptive Dopri5** (rtol=1e-5, atol=1e-7). Lyapunov M1 verified empirically: 0/29 patients × 4 T-values = 0/116 violations, dU/dt=−‖∇U‖²≤0 by identity. D2 truncation: v10 single-Euler at η=2 was 2.72 L2 units off true ODE flow (≈0.34σ_emp). D3: v10 drift was 5–10% misdirected and 20% over-shooting at the F8 best cell. | None of the operator-correctness improvements *changed an F-gate verdict*. F4 PASS preserved; F8-DPS still STRICT FAIL (best p_vs_const moves 0.432→0.402, p_vs_pop 0.050→0.050). |
| Framing of F4 | **Mathematically honest**: F4 measures rotational *data residual*, not "Tikhonov-suppressed model curl". ∇×∇U=0 is an identity for any twice-differentiable scalar U, so the gradient-flow parameterization is curl-free *by construction*. The v11 plan §2 row 2 was corrected to remove the redundant "Helmholtz projection layer". | The v10 paper draft and pre-Sprint-1 v11 plan were prose-wrong on this point; chair-verdict-must-fix in the manuscript. |
| Encoder upgrade | scFoundation properly probed and **honestly REJECTED** at the packaging layer (biomap-research Python 3.7 + Tencent OneDrive weights; genbio-ai pickle-only; perturblab unpublished). Geneformer V1-10M fallback **also REJECTED** by the F2 zero-trust check: v10 F2 REFUTED (Δ=24.19, CI [-43, +99]); v11 Geneformer F2 PASSES (Δ=8.36, CI [2.34, 15.18]); per-dim standardized retest STILL passes (Δ=8.44, CI [3.86, 14.12]) → confirmed leak, not a scale artifact (`paper/v8_artifacts/v11_sprint1/encoder_upgrade.json`, `f1_f2_gates_v11_standardized.json`). | The v11 paper *cannot* claim a foundation-model encoder upgrade. v10 PCA(64) remains the production encoder. The leak finding is itself a valid scientific contribution but it is a **deferral** of v11 plan §2 row 1, not a fulfillment. |

**Sprint 1 net:** the operator is mathematically honest, the encoder upgrade is honestly blocked, and F8-DPS STRICT FAIL is robust to operator change — strengthening the v10 identifiability-bound thesis. Paper-grade contribution: M1 Lyapunov (new gate) is publishable on its own as a methodological clean-up. **Strength = honesty under a falsifying lens. Weakness = no upgrade actually ships forward for v11 production beyond Dopri5.**

### §2.2 Sprint 3 — Multi-α attention propagation (`docs/V11_SPRINT3_GAT_VERDICT.md`, RUNS row v11s3-gat-f6)

| Axis | Strength | Weakness |
|---|---|---|
| Mathematical generalization | H=1, α=0.7 ensemble bit-identically reproduces v10 RWR (max abs deviation = 0.0e+00 across all 12,651 STRING nodes for PSMB5 seed). Multi-α (H=4, α∈{0.3,0.5,0.7,0.9}, mean) is a *strict generalization*, not a different model. | This is *training-free*; the originally-planned GAT-2 with learnable attention was deferred to v12. The v11 sprint is an "axis test" for the v12 work, not the v12 work itself. |
| F6 strict-pass | **PRESERVED** at p=0.0001 (B+1 floor, B=10000): T_obs=2213.7, null mean 3861.6 ± 53.9 → **30.6σ** (v10: 30.7σ). T_obs improved by 64.9 mean-rank positions. | σ vs null is essentially flat (30.7→30.6, ΔMC). Both T_obs and T_null shift symmetrically under α-averaging, so the test statistic is unchanged. The biological-signal lift (64.9 ranks) is publishable but does not move the falsification verdict. |
| First-principles | Honest analysis of why ensemble works: tight clusters (proteasome, paralogous HDACs) extracted by high-α; broad pathway hubs (NF-κB, MAPK) reached by low-α. | F3 (10-drug Bonferroni essentiality) and F7 (multi-seed pooling, Vorinostat structural fail) NOT re-run in this sprint; v10 numbers carry forward. |

**Sprint 3 net:** strict generalization of v10 RWR + 64.9-position raw-rank improvement, with strict-pass preserved at the same p-value floor. Clean win. **Strength = no regression risk; Weakness = the architectural change does not unlock new claims.**

### §2.3 Sprint 5 — Mondrian jackknife+ + discrimination + mmSYGNAL head-to-head (`docs/V11_SPRINT5_CONFORMAL_VERDICT.md`, RUNS rows v11s5-conformal, v11s5-discrim, v11s5-cox, v11s5-mmsygnal, v11s5-cox-with-programs, v11s5-mmsygnal-complete-routing, v11s5-paired-cindex-test)

| Axis | Strength | Weakness |
|---|---|---|
| F_S5 calibrated coverage | **STRICT PASS preserved.** Ridge_v11features: 5/5 strata within ±3%, marginal coverage 0.898 vs nominal 0.900. Marginal width tightens by 8d (1643→1635, 0.5% on a 3-yr horizon). | Width tightening is small. S4 t(11;14) (n=95) widens by 360d (16.8%, smallest stratum, feature-overfit pattern). |
| GBM as base learner | **REJECTED by F_S5** (over-covers marginally at 0.950 with +217d width). Documented negative result; honest. CQR (Romano 2019) is the v12 next step. | A v12 follow-through is required to recover the discrimination–coverage tradeoff (GBM_v11features wins discrimination at C=0.566 but FAILS F_S5). |
| Discrimination — MSE loss | Ridge_v10 0.537 → Ridge_v11features 0.556 (+0.019); 12mo AUC 0.530 → 0.593 (+0.063). Waddington features add **signed discriminatory information**, consistent direction across base-learner classes (+0.029 C, +0.093 12mo AUC under GBM). | Absolute C-index sits in the weakly-discriminating range; SOTA-incompatible. |
| Discrimination — Cox loss | Switching MSE→Cox PH on the SAME 11 features adds +0.116 marginal C-index (Ridge_v10 0.537 → Cox_v10 0.653). Cox_v11_richer 0.654. | This proves the v10/v11 discrimination gap was loss-function, not feature-set — a methodological contribution. |
| mmSYGNAL head-to-head (PMID 40169765, Murie/Baliga 2025 BJC) | **HONEST head-to-head executed** on the same MMRF N=787, same TT2L outcome, same 5-stratum partition. mmSYGNAL 4-submodel marginal C=0.694 [95% CI 0.658–0.729]; v11 Cox best 0.654 → **Δ=+0.040 to mmSYGNAL** with mmSYGNAL CI lower bound (0.658) above all v11 Cox point estimates. The earlier "at floor" framing was OVERTURNED. | This was a clean honest concession; the v11 paper title cannot be "we beat mmSYGNAL." |
| v11.5 ensemble | Cox_v11_routed_mmsygnal (24 features = v11_richer + mmSYGNAL routed score) achieves **marginal C=0.6955 vs mmSYGNAL 6-submodel 0.6957**. Paired bootstrap Δ=−0.00005, 95% CI [−0.0265, +0.0267]. Per-stratum: S2 t(4;14) 0.586→0.667 (+0.081), S4 t(11;14) 0.521→0.643 (+0.122). The combination dominates either component alone. | The TIE *requires* the mmSYGNAL routed score as an input feature. |
| **Caveat #1 closed** (paired test) | **U-statistic z-test:** Δ=−0.00014, SE=0.01384, z=−0.008, **p_two=0.994**, n_pairs=70,634 (or 95,938 in the cleaned re-run). **B=10000 bootstrap:** marginal Δ=−0.00005, 95% CI [−0.0265, +0.0267], **p_two=0.989**; stratified Δ=−0.0003, 95% CI [−0.0267, +0.0263], p_two=0.979. The 99% CI [−0.035, +0.036] rules out v11.5 advantage/disadvantage > ±3.6 C-index points at the 1% level (`paper/v8_artifacts/v11_sprint5/paired_cindex_test.json`). | Per-stratum S2 t(4;14) uncorrected p=0.044 (Bonferroni m=5 → 0.22 NS); biology of FGFR3+/t(4;14) advantages mmSYGNAL there. |
| **Caveat #3 closed** (mmSYGNAL undercredit) | 6-submodel mmSYGNAL routing executed via derived del(1p36) (chr1 1–30 Mb weighted-mean Segment_Mean, bottom-decile, n=73) and FGFR3+ (top-decile baseline expression, n=79) proxies. 6-submodel marginal C=0.6957 (vs 4-submodel 0.6938, Δ=+0.0019). Sweep peaks at literature-prevalence cutoff (5%→0.6934, 10%→0.6957, 15%→0.6859, 20%→0.6854). v11.5 paired bootstrap on the 6-submodel reference: Δ=−0.0002 [−0.026, +0.025] marginal; Δ=+0.0002 [−0.027, +0.028] stratified — **TIE survives against the unbiased reference** (`paper/v8_artifacts/v11_sprint5/mmsygnal_complete_routing.json`). | del(1p36) and FGFR3+ are V11-derived proxies, not official MMRF FISH labels. A reviewer with MMRF biomarker-portal access could tighten this further. |
| **Caveat #2 closed and FOREGROUNDED** (composition decomposition) | Two pure anchors: **pure mmSYGNAL** C=0.6957 (no v11), **pure v11_richer** C=0.6538 (no mmSYGNAL). v11.5 ensemble = 0.6955. **+v11_richer ON TOP of mmSYGNAL routed: Δ=−0.0002.** **+mmSYGNAL routed ON TOP of v11_richer: Δ=+0.0417.** **+v11_richer ON TOP of raw 6 mmSYGNAL scores: Δ=+0.0104.** The mmSYGNAL routing logic itself is worth +0.0186 (raw 0.6771 → routed 0.6957). | **~95% of v11.5's marginal-C gain over pure v11 is from the mmSYGNAL routed feature.** v11 features add +0.010 over raw 6 mmSYGNAL scores. The architectural lift is ~1 C-point. This MUST be foregrounded in any manuscript figure or claim line. |

**Sprint 5 net:** F_S5 strict-pass + Caveat-1/2/3 all closed, all corrections went the direction the falsification framework predicted, and the v11.5 = "v11 calibrated coverage composed with mmSYGNAL routed score" framing is defensible. **Strength = the paired test machinery is rigorous and the decomposition is uncomfortably honest. Weakness = pure-v11 discrimination is not SOTA, and the v11.5 SOTA-tie cannot be reported without crediting mmSYGNAL.**

### §2.4 Sprint 7 — F8-DPS under Neural-ODE operator (`paper/v8_artifacts/v11_sprint7/f8_neural_ode.json`, RUNS row v11s7-ode-dps)

| Axis | Strength | Weakness |
|---|---|---|
| Operator robustness | **0/16 cells F8-pass** under Neural-ODE Dopri5 (best p_vs_const=0.402 at η=4, σ=13.64; best p_vs_pop=0.050 at η=4, σ=3.41 and η=1, σ=13.64). Neither reaches the AND-of-both-baselines p<0.05 threshold. v10: 0/16, best p_vs_const=0.432. The marginal tightening (0.432→0.402) is informative-direction but well short of the gate. | F8-DPS still STRICT FAIL. |
| Bound is operator-independent | The Stone minimax bound at N_paired=29 < ~150 for d=64 is a regime-dependent property, not an operator-dependent one. F8-DPS REFUTED under two different forward operators — strengthens the v10 identifiability-limit thesis. | The v11 paper inherits the v10 §6 fallback: per-patient claims restricted to Sprint 5 (population-level conformal coverage) + Sprint 4 (proteasome-NIE supporting evidence). |

**Sprint 7 net:** the most important *negative* result of v11. The strict fail is a *contribution*, not a failure to deliver — it answers the obvious peer-reviewer challenge "your single-Euler step is not a Neural-ODE; the F8 result is an artifact." It is no longer an artifact. **Strength = bound is robust to operator. Weakness = no per-patient claims at N=29 are unlocked.**

### §2.5 Agentic / observability / wiring (`docs/V11_AGENTIC_JUSTIFICATION.md`, `paper/v8_artifacts/v11_sprint_agentops/`)

The agentic layer audit is honest: the in-pipeline `*Agent` classes are misleadingly-named stage-runners (NOT a contribution); the **fabrication-sentinel + release-bouncer + RUNS.md** policy stack IS a unique contribution; the AgentOps Observability pillar is now wired for v11 stages (existence proof: `logs/agentops/v11/s1f_ode_forward_diagnostic_<uuid>.json`). The Optimizer pillar is still scaffolding; subagent silent-failure on MCP denial is a real defect (demonstrated by the iMLGAM mis-positioning that the post-hoc verification sweep had to overturn).

**Strength = honest self-audit with concrete evidence. Weakness = `mcp_integration/connectors.py` is 806 lines of `NotImplementedError`, MCP wiring (#63) and HBayes PPC (#67) are open.**

---

## §3 Uniqueness brief — where ResistanceMap v11/v11.5 is genuinely differentiated

The v11/v11.5 niche is narrow. Three concrete capabilities, each supported by an artifact citation, that name a defensible differentiation against the user-named comparators:

### §3.1 vs **mmSYGNAL** (Murie/Baliga 2025 BJC, PMID 40169765, 1,367 MM patients, 5 cohorts)

mmSYGNAL **dominates** v11 on pure discrimination (mmSYGNAL C=0.6957 vs Cox_v11_richer C=0.6538 on the same MMRF N=787, Δ=+0.042). v11/v11.5 **adds two things mmSYGNAL does not publish**:

1. **Calibrated 90% prediction intervals via Mondrian jackknife+ on log(TT2L)**: Ridge_v11features 5/5 strata within ±3% of nominal 0.900, marginal coverage 0.898, marginal width 1635 days (`paper/v8_artifacts/v11_sprint5/mondrian_jk_plus_v11.json`). mmSYGNAL is pure C-index; no calibrated intervals. This is the v11 architectural contribution that **stands alone, with no mmSYGNAL feature dependency.**
2. **Pre-registered identifiability-bound annotation**: F8-DPS strict-fail at N_paired=29 < Stone(d=64)≈150, robust under v10 single-Euler AND v11 Neural-ODE Dopri5 operators (`paper/v8_artifacts/v11_sprint7/f8_neural_ode.json`). mmSYGNAL does not publish its identifiability regime; v11 is explicit about where it does NOT claim per-patient forecasting.

### §3.2 vs **TRACERx Lung** (Frankell 2023 Nature, PMID 37046096; Martínez-Ruiz 2023 Nature, PMID 37046093)

TRACERx is bulk-WES/RNA evolution on NSCLC; **not a head-to-head metric peer**. Differentiation from TRACERx as inspirational frame: v11 operates on **hematologic malignancy with longitudinal MMRF CoMMpass IA22 cohort** (N=787, paired N=29) and **explicitly bounds its own identifiability regime** via the F8-DPS gate, which TRACERx does not do for any of its multi-clonal-evolution claims. Use TRACERx as a citation for subclonal-evolution-as-prognosis concept, never as a metric peer.

### §3.3 vs **TrajectoryNet** (Tong 2020 ICML, arXiv 2002.04461)

TrajectoryNet parameterizes a **free vector field** with continuous normalizing flows and *measures* curl post-hoc (or ignores it). v11/v10 parameterizes the drift as F_θ=−∇U_θ where U_θ is a scalar MLP, so **∇×∇U=0 is a vector calculus identity, not a regularizer or post-hoc measurement** (`docs/V11_SPRINT1_ODE_VERDICT.md` §1, M1 Lyapunov 0/116 violations). TrajectoryNet is a *trajectory* method on EB-style multi-snapshot single-cell data; v11 is a *single-snapshot* method on bulk-RNA + cytogenetic + clinical at population level. **Not directly comparable on home-turf metrics; v11 wins on the gradient-flow-by-construction axis only.** RM v11 has not been benchmarked on the EB or GSE271107 LOPO benchmarks where TrajectoryNet's EMD=0.784 was reported; that head-to-head is queued for v12.

### §3.4 vs **Pan-cancer Proteogenomics 2024** (Petralia/Ma/Yaron 2024 Cell, PMID 38359819)

Pan-cancer proteogenomics is bulk multi-omics subtype discovery on solid tumors; **not a metric peer**. Differentiation: v11 ships **a falsification-pre-registered framework with strict-pass thresholds dated before data inspection** (F1–F10, F_S5, F8-DPS, M1) and a **release-bouncer that refuses to publish numbers without a RUNS.md row** (`docs/V11_AGENTIC_JUSTIFICATION.md` §4.2). Pan-cancer proteogenomics is a substrate paper; v11 is a methodologically-disciplined application paper.

### §3.5 vs **ML-melanoma 2024** (TCGA-SKCM bulk-prognostic ML; closest verified candidate Dong 2026 Front Immunol, PMID 41613147 — `[UNVERIFIED]` for the user's intended paper)

Skin melanoma ≠ MM/AML; bulk-prognostic, not trajectory. Differentiation: v11 is **disease-specific (MM primary, AML cross-disease control)**. The Beat AML cross-disease F3 refutation (RUNS row v10s6: 0/5 AML-native F3 panels at α=0.01) is itself a contribution — v11 publishes its **empirical bound on cross-disease transfer**, refusing the easy "pan-cancer" framing.

### §3.6 vs **iMLGAM** (Ye 2025 iMeta, PMID 40236779)

**iMLGAM is pan-cancer ICB-response classification, NOT MM-specific** (verified post-hoc; the user's framing of iMLGAM as "MM/MGUS specific" was confirmed incorrect by the sota_comparison §0.1 verification override). Differentiation: v11 is **MM/AML-scoped with explicit MM-specificity claim, validated by cross-disease F3 refutation on Beat AML 1.0 (Tyner 2018 PMID 30333627)**. iMLGAM does not run a CRISPR validation arm relevant to MM. Use iMLGAM as a pan-cancer ICB comparator only, never as an MM head-to-head.

### §3.7 The defensible niche paragraph

> **ResistanceMap v11/v11.5 is uniquely the combination of: (a) a Helmholtz-by-construction gradient-flow Neural-ODE on a learned scalar Waddington potential U_θ verified by 0/116 Lyapunov violations under adaptive Dopri5 (`v11_sprint1/ode_forward_diagnostic.json`), (b) a STRING v12 multi-α attention propagation kernel that bit-identically generalizes single-α RWR and preserves F6 30.6σ strict-pass on 45 consensus MM drivers (`v11_sprint3/f6_driver_recall_gat.json`), (c) Mondrian jackknife+ calibrated 90% prediction intervals tightened by 4 Waddington-derived features (`v11_sprint5/mondrian_jk_plus_v11.json`, F_S5 5/5 within ±3% of 0.900), (d) a pre-registered identifiability-regime annotation via F8-DPS strict-fail under two forward operators at N_paired=29 < Stone(d=64) (`v11_sprint7/f8_neural_ode.json`, `v10_sprint7/f8_energy_distance.json`), (e) a multi-omic MMRF CoMMpass IA22 N=787 + Beat AML 1.0 cross-disease control + paired N=29, with (f) a release-bouncer that refuses any number not grounded in `RUNS.md`, AND (g) an honest paired-bootstrap-confirmed TIE against mmSYGNAL on PFS C-index when the mmSYGNAL routed score is added as a feature, with the architectural decomposition openly reporting that ~95% of the SOTA-tying gain is from mmSYGNAL features (`v11_sprint5/cox_with_programs.json` + §3.11 of the Sprint 5 verdict). No comparator above owns this combination simultaneously.**

That paragraph is the strongest defensible v11/v11.5 claim. Anything stronger overclaims; anything weaker undercredits the M1 Lyapunov and F_S5 contributions.

---

## §4 Manuscript structure proposal

Two parallel manuscripts, two venues, two scope claims. **DO NOT merge them.**

### §4.1 Manuscript M1 — "Calibrated coverage with falsification pre-registration on MM PFS"

**Target:** bioRxiv now → Biostatistics / Statistics in Medicine / JCO CCI on revision.

| Section | Sprint(s) sourcing | Figure / table proposal |
|---|---|---|
| §1 Intro | n/a | Figure 1: claim region — single-snapshot N_paired=29, d=64, Stone bound annotated |
| §2 Method (Waddington potential + Neural-ODE + multi-α propagation) | Sprint 1 + Sprint 3 | Figure 2: M1 Lyapunov panel (0/116 violations across 4 T-values) + multi-α-vs-single-α RWR equivalence (max abs deviation 0.0e+00) |
| §3 F4 framing correction | Sprint 1 §1 | Inline equation block: ∇×∇U=0 as identity; F4 = data-residual curl |
| §4 F6 driver recall | Sprint 3 | Figure 3: 30.6σ vs degree-matched null + 64.9-position raw rank improvement |
| §5 F_S5 calibrated coverage | Sprint 5 §2 | **Table 1: per-stratum coverage and width** (5 strata × {Ridge_v10, Ridge_v11features, GBM_v10}); GBM rejection documented |
| §6 F8-DPS identifiability bound | Sprint 7 | Figure 4: 16-cell sweep heatmap, p_vs_const + p_vs_pop, both operators |
| §7 Limitations | n/a | Encoder upgrade (`encoder_upgrade.json`) reported as honest deferral; cross-disease F3 (Beat AML) reported as MM-specificity bound |

This manuscript's headline claim is **(A)** in §1 and does NOT depend on mmSYGNAL features. ~6,000 words, ~4 figures, ~2 tables. Defensible at peer review.

### §4.2 Manuscript M2 — "Composing v11 calibrated coverage with mmSYGNAL transcriptional-program risk on MMRF"

**Target:** bioRxiv now → JCO CCI / Cancer Informatics on revision (NOT Nature Methods — discrimination is mmSYGNAL's, not v11's).

| Section | Sprint(s) sourcing | Figure / table proposal |
|---|---|---|
| §1 Intro | Sprint 5 §3.7 | Honest framing: "v11 alone Δ=−0.040 below mmSYGNAL; v11+mmSYGNAL ties." |
| §2 Cox PH variants on MMRF N=787 | Sprint 5 §3.6 | **Table 1: 5-variant Cox PH C-index, marginal + per-stratum** (Cox_v11_richer, Cox_v11_routed_mmsygnal, Cox_v11_six_mmsygnal_scores, Cox_only_six_mmsygnal_scores, Cox_v11_program_activity_pcs) |
| §3 mmSYGNAL head-to-head + paired test | Sprint 5 §3.7–§3.10 | **Table 2: U-statistic z-test + B=10000 bootstrap, marginal + stratified, with 99% CI** ([−0.035, +0.036] at marginal) |
| §4 6-submodel routing | Sprint 5 §3.9 | Figure: cutoff sensitivity sweep (5/10/15/20%); peak at 10% (lit-prevalence) |
| §5 **Decomposition (FOREGROUNDED, NOT BURIED)** | Sprint 5 §3.11 | **Figure 3 (centerpiece): bar chart of marginal C-index for {pure mmSYGNAL, pure v11_richer, raw 6 mmSYGNAL, v11_richer + raw 6 mmSYGNAL, v11.5}**, with the +0.042 mmSYGNAL contribution and +0.010 v11 contribution explicitly labeled. |
| §6 What v11 contributes that mmSYGNAL alone does not | Sprint 5 §3.8.6 | Calibrated coverage panel; Lyapunov M1 panel; F8-DPS bound |
| §7 Limitations | n/a | del(1p36)+FGFR3 are V11-derived proxies; full MMRF FISH would tighten |

This manuscript's headline claim is **(B)** in §1 and is HONEST about composition. ~5,000 words, ~4 figures, ~2 tables. Reviewable as a clinical-informatics ensemble paper, not a methods paper.

### §4.3 Joint figure for either manuscript: the niche-positioning radar

A 6-axis radar (mmSYGNAL, TRACERx, TrajectoryNet, CellRank 2, MOFA+Ridge, v11/v11.5) on {discrimination C, calibrated coverage, identifiability annotation, MM-specificity, longitudinal trajectory, foundation-model encoder}. v11/v11.5 owns the calibrated-coverage + identifiability + MM-specificity vertices; comparators own the others.

---

## §5 Outstanding gaps — what gates each venue

Five outstanding items, each tagged with which manuscript / venue it gates.

| # | Item | Status | Gates which venue? | Sprint to land in |
|---|---|---|---|---|
| #67 | **HBayes posterior-predictive check (Gelman 1996 §6) on F1**, per V11 plan §2 row 5 | OPEN | Gates: Nature Methods / Statistics in Medicine for M1. Does NOT gate bioRxiv. | v11 Sprint 2 (deferred) — small re-run on existing NumPyro 5-stratum HMC chain. |
| #69 | **TCGA-LAML (PMID 23634996, 200 cases)** as second AML cohort for cross-disease F3 | OPEN | Gates: any "robust MM-specificity" claim in M1 §7 / M2 §7. | v11 Sprint 6+ — TCGA-LAML data already public; ~3 days work. |
| #63 | **MCP wiring** for `mcp_integration/connectors.py` (replace 806 lines of `NotImplementedError` with thin clients around the live MCP servers) | OPEN | Gates: ICML/ICLR if the manuscript claims an "agentic / multi-agent" methodological contribution. Does NOT gate any clinical venue. | v12. |
| **scFoundation re-attempt** | OPEN | Gates: M1 if v11 wants to claim a foundation-model encoder upgrade. Does NOT gate v11.5 (M2) which uses PCA(64). | v12 — v11 plan §2 row 1 alternative paths (a) port real scFoundation forward, (b) ComBat-corrected Geneformer, (c) attention-pooled adapter pretrained only on masked-gene reconstruction. All three queued in the encoder_upgrade.json `v12_plan` field. |
| **Pre-registered held-out cohort replication** for v11.5 | OPEN | **Gates JCO CCI / Nature Methods for M2.** All v11.5 numbers are LOO on MMRF N=787; an external replication (CoMMpass IA21, HOVON, GMMG) would establish generalization. | v12 — depends on data access (HOVON / GMMG) which is non-trivial. |

None of these gate **bioRxiv submission for either M1 or M2 today.** The v11/v11.5 evidence base is sufficient for preprint, with #67 and #69 closure required for the formal-journal revision rounds.

---

## §6 Recommended freeze decision

| Venue | Verdict | Required closures |
|---|---|---|
| **bioRxiv (M1: calibrated coverage)** | **YES — freeze and post.** | None. The §3.11 decomposition foregrounded; F_S5 + M1 + F4 + F6 strict-pass cited from RUNS.md. |
| **bioRxiv (M2: v11.5 + mmSYGNAL ensemble)** | **YES — freeze and post.** | Title MUST be "Composition of …" not "matches mmSYGNAL." §3.11 decomposition is the centerpiece figure. |
| **ICML / ICLR (M1 as a methods paper)** | **NO — not without a TrajectoryNet / MIOFlow head-to-head on a public benchmark (EB or GSE271107 LOPO).** | v12 Sprint A: implement TrajectoryNet on the same split, report W₁ + EMD. Without this, M1 reads as MM-specific clinical statistics, not a methods contribution. |
| **Nature Methods (M1 as a calibrated-coverage methods paper)** | **NO — not until #67 (HBayes PPC) closes AND a second-cohort replication of F_S5 lands.** | v11 Sprint 2 + v12 Sprint B (HOVON or external MM cohort). |
| **JCO CCI (M2 as a clinical-informatics ensemble paper)** | **CONDITIONAL — yes if (a) external MMRF IA21 or HOVON cohort replication of v11.5 marginal C-index ties at 0.69+, and (b) the §3.11 decomposition is foregrounded as a feature, not buried as a caveat.** | v12 Sprint C (external replication). Estimated 4–6 weeks of data access + re-fit. |
| **v12 work needed** | (a) scFoundation port OR ComBat-corrected Geneformer OR attention-pool adapter (encoder upgrade); (b) HBayes PPC #67; (c) TCGA-LAML #69; (d) MCP wiring #63; (e) external-cohort replication for both M1 (F_S5) and M2 (v11.5). | — |

**Sunset for this CONDITIONAL_PASS verdict: 2026-08-31.** Re-evaluation criterion: at sunset, any of the v12 items above that have NOT landed cause the corresponding venue claim to drop one tier (Nature Methods → bioRxiv, JCO CCI → bioRxiv, bioRxiv → "do not post until v12 lands").

---

## §7 Falsification criteria — three concrete tests that would flip this verdict

1. **F_S5 fails on a held-out external MM cohort.** If Ridge_v11features re-evaluated on HOVON-65 / GMMG-HD4 (Kuiper 2012 PMID 22552008 cohort) yields per-stratum coverage outside ±5% of nominal 0.90 in any stratum, the calibrated-coverage claim (M1 headline) is REFUTED on generalization grounds, and the verdict drops to FAIL on M1.
2. **v11 features add zero or negative C-index ON TOP of mmSYGNAL routed score on an external cohort.** If the +0.0104 marginal lift in `cox_with_programs.json` for "Cox_v11_six_mmsygnal_scores − Cox_only_six_mmsygnal_scores" goes below 0 on an external cohort, then the v11 architectural contribution to discrimination collapses and M2's "composition" framing collapses to "mmSYGNAL with extra noise." Verdict drops to FAIL on M2.
3. **F8-DPS passes at N_paired ≥ Stone bound and v11 architecture cannot replicate.** If a future cohort with N_paired ≥ 150 and d=64 yields F8-DPS p_vs_const < 0.05 AND p_vs_pop < 0.05 with a competitor (TrajectoryNet, TIGON, MIOFlow) but NOT with v11's Neural-ODE+U_θ, then v11's identifiability-regime annotation is correct but its operator is dominated, and the per-patient claim region remains closed to v11. Verdict on M1 drops to CONDITIONAL_PASS-with-asterisk requiring a v12 architectural rebuild.

If any one of these flips, the chair re-runs this verdict before sunset 2026-08-31.

---

## §8 Appendix — Bottom line for the user

The v11/v11.5 stack is **paper-ready at narrow scope** and **dishonest if reported at wide scope**. Two manuscripts at bioRxiv, with the §3.11 decomposition (Caveat #2: ~95% of v11.5's SOTA-tying gain is mmSYGNAL features, ~1 C-point is v11) **as the centerpiece of the v11.5 manuscript, not a footnote**. That is the only honest path. Going to Nature Methods or JCO CCI requires v12 work (encoder, external cohort, HBayes PPC); going to ICML/ICLR requires a TrajectoryNet head-to-head that does not exist yet.

The unique contribution that survives every audit is: **(i) F_S5 5/5 strata within ±3% calibrated coverage tightened by Waddington features**, **(ii) M1 Lyapunov 0/116 violations under adaptive Dopri5**, **(iii) F6 30.6σ strict-pass with multi-α generalization**, and **(iv) F8-DPS operator-robust strict-fail = pre-registered identifiability-regime annotation no MM SOTA publishes.** Everything else is composition with mmSYGNAL or honest deferral to v12.

---

*End of v11 chair verdict. All numerical claims trace to a `RUNS.md` row or a JSON artifact path. No fabrication.*
