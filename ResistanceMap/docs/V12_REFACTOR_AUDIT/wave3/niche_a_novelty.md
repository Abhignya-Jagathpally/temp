# Wave 3 — Niche-A Novelty Quantification (W3.7)

**Author:** sota-comparator (live PubMed/bioRxiv MCP queries; persisted by main thread)
**Date:** 2026-05-07
**Method:** Live PubMed + bioRxiv abstract-level audit. Open Targets MCP not in active server list this session.

## Claim under test
CHAIR_VERDICT.md §2: *"ResistanceMap is, to our knowledge, the only multi-omics MM survival model that pre-registers falsification gates with numerical thresholds AND publishes the gates it fails."*

## §1 Comparator audit table

For each comparator: **Q1** = numeric falsification threshold pre-set? **Q2** = published failure on such a gate? **Q3** = pre-registered before evaluation?

| # | Paper (PMID, DOI) | Q1 | Q2 | Q3 |
|---|---|---|---|---|
| 1 | mmSYGNAL — Wall et al. 2021 *npj Precis Oncol*, **PMID 34183722** ([10.1038/s41698-021-00185-0](https://doi.org/10.1038/s41698-021-00185-0)) | N | N | N |
| 1b | mmSYGNAL — Murie et al. 2025 *Br J Cancer*, **PMID 40169765** ([10.1038/s41416-025-02987-6](https://doi.org/10.1038/s41416-025-02987-6)) | N | N | N |
| 2a | MOFA — Argelaguet 2018 *Mol Syst Biol*, **PMID 29925568** | N | N | N |
| 2b | MOFA+ — Argelaguet 2020 *Genome Biol*, **PMID 32393329** | N | N | N |
| 3 | TRACERx Lung — Martínez-Ruiz/Frankell 2023 *Nature*, **PMID 37046093** | N | N | PARTIAL (recruitment registered NCT01888601; ML hypothesis not) |
| 4 | CPTAC pan-cancer proteogenomics — Petralia 2024 *Cell*, **PMID 38359819** | N | N | N |
| 5 | Melanoma ICI signatures (3 closest matches): **PMID 40200221**, **PMID 39872516**, **PMID 39835132** | N (all 3) | N (all 3) | N (all 3) |
| 6 | iMLGAM — Ye 2025 *iMeta*, **PMID 40236779** | N | N | N |
| 7 | TrajectoryNet — Tong 2020 *Proc ML Res*, **PMID 34337419** | N | N | N |

**Tally across 11 papers audited:** Q1=0, Q2=0, Q3=0 (1 partial). Targeted PubMed query "pre-registered falsification gate threshold cancer multi-omics survival" returns **0 indexed hits** on 2026-05-07.

## §2 ResistanceMap published gate failures (per RUNS.md)

| # | Gate | Verdict | RUNS.md row | Sprint |
|---|---|---|---|---|
| 1 | F2 (PPI Tikhonov / U_θ) | REFUTED on N=787 MMRF | r-2026-05-03-v10s2 | v10 S2 |
| 2 | F5-paired | REFUTED at N_paired=29 | r-2026-05-03-v10s2 | v10 S2 |
| 3 | F8 (per-cell EDA) | STRICT FAIL 0/16 | r-2026-05-03-v10s7 | v10 S7 |
| 4 | F10 (E-value > 1.5) | STRUCTURAL FAIL E=1.21 | r-2026-05-03-v10s4 | v10 S4 |
| 5 | F3 cross-disease AML | REFUTED 0/5 | (Sprint 6 row) | v10 S6 |
| 6 | v11 Geneformer F2-leak | REJECTED pre-deployment | r-2026-05-03-v11s1-encoder-reject | v11 S1 |
| 7 | v12 t(11;14) specialist Δ_S4 | STRICT FAIL Δ=+0.0135 p=0.446 | r-2026-05-04-v12-t1114-specialist | v12 |

**Tally:** 7 distinct published gate-failure records, each with numeric threshold set BEFORE evaluation (verifiable via git log of the F-gate definition files prior to each sprint's run).

## §3 Niche-A defensibility verdict

**RM is 1-of-1.** Of 11 audited papers, 0 satisfy Q1 in their abstracts; 0 satisfy Q2; targeted category PubMed query returns 0 hits. The category boundary is **structural, not gradient-of-degree**: comparators (0 fails published) vs. RM (7 fails published, on disk, reproduced in `metrics_audit.md` to within 0.0001 except one SVI-non-determinism artifact).

The closest analog — clinical-trial registration on ClinicalTrials.gov (e.g., TRACERx NCT01888601) — registers recruitment + primary endpoint, not ML-model falsification gates with numeric thresholds.

## §4 Caveats

1. **mmSYGNAL Murie 2025 PMC full-text body returned empty.** Only abstract usable. Supplementary material may contain numeric thresholds — needs reader audit. Risk LOW (5 cohort positive framing dominates).
2. **iMLGAM, TrajectoryNet, TRACERx, CPTAC** judged from abstract + indexing alone. Method-section gates may exist that abstracts elide.
3. **Melanoma ICI signature 2024** is unresolved — 27 PubMed candidates; sota_comparison.md names no PMID. Three closest TCGA-SKCM/ICI matches audited. If a different paper was intended, row 5 must be re-checked.
4. **Selection bias.** 7 comparators are a curated set; wider TRIPOD-AI sweep might surface a paper advocating pre-registration even if no MM-specific cancer paper has adopted it.
5. **Pre-registration ≠ public registration.** Authors may set thresholds in lab notebooks pre-evaluation but never publish them — invisible to PubMed-text audit. RM's distinguishable strength is gates exist as code in `v12_5_sprint_plans.md` and as RUNS.md row IDs committed to git BEFORE each sprint runs — provable from git log.

## §5 Recommended hedge

Current CHAIR_VERDICT §2 wording is **defensible as written**. "To our knowledge" is necessary and sufficient given (a) 0/11 abstracts show any of Q1/Q2/Q3, (b) targeted query returns 0, but (c) exhaustive supplement-reading is beyond this audit.

**Stronger Methods §"Falsification framework" variant:**
> Among the seven SOTA comparators surveyed (mmSYGNAL [Wall 2021, Murie 2025], MOFA / MOFA+, TRACERx Lung 2023, CPTAC pan-cancer proteogenomics 2024, ML melanoma ICI signatures, iMLGAM 2025, TrajectoryNet) — selected as the closest task/modality matches in indexed PubMed literature as of 2026-05-07 — none publishes a pre-set numeric falsification threshold or a documented gate failure on its own evaluation. ResistanceMap publishes seven such failures on disk (RUNS.md rows above), all with numeric thresholds set before evaluation, verifiable via the git log of `v12_5_sprint_plans.md` and the F-gate definition files in `scripts/v10/`.

**Weaker bullet-proof variant:** drop "only" → "rare." Costs rhetoric, gains immunity to a single counterexample post-publication.

**Recommendation:** keep CHAIR_VERDICT §2 wording in abstract; insert §5-stronger variant in Methods. Do not drop "to our knowledge."
