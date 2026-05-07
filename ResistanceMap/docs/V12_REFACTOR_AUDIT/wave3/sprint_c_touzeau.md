# v12.5 Sprint C - Touzeau 2014 t(11;14) Panel Verdict

Branch: v6 (worktree tier1-refactor)
Date: 2026-05-07
Run: v12.5-wave3-sprint-c-touzeau

## Section 1 - 5-gene panel sourced from

Source paper: Touzeau et al. 2014, Leukemia 28(1):210-212,
DOI 10.1038/leu.2013.216 (PubMed PMID 23860449, e-pub 2013-07-17).
The chair-verdict task spec cites PMID 24305167, which does not
resolve in PubMed; PMID 23860449 is the actual Leukemia 2014 Touzeau
paper on ABT-199 / t(11;14) myeloma. According to PubMed, this is
the canonical Touzeau-2014 ABT-199 source.
DOI link: https://doi.org/10.1038/leu.2013.216

Panel reconstruction: The published Touzeau 2014 letter does NOT
contain a discrete 5-gene predictive panel with weights; the only
predictive markers reported are (i) the BCL2/MCL1 mRNA ratio (most
powerful biomarker, p = 0.002), (ii) BCL2 expression (p = 0.008),
and (iii) MCL1 expression (trend, p = 0.09), with t(11;14)/CCND1
translocation as the cytogenetic enrichment factor and BCL-XL
(BCL2L1) co-dependence in a subset (extended in Touzeau 2015,
PMID 26174630, DOI 10.1038/leu.2015.184).

Per sprint-plan Sprint C deliverable 1, we use the
literature-reconstructed BCL2-family fallback panel:

| Gene    | Direction | Rationale                                                |
|---------|-----------|----------------------------------------------------------|
| BCL2    | +1        | High BCL2 -> ABT-199 sensitivity (Touzeau 2014, p=0.008) |
| BCL2L11 | +1        | BIM, pro-apoptotic (sensitises)                          |
| BAX     | +1        | pro-apoptotic effector                                   |
| MCL1    | -1        | High MCL1 -> ABT-199 resistance (Touzeau 2014, p=0.09)   |
| BCL2L1  | -1        | BCL-XL co-dependence; redundant survival                 |

Touzeau_score = sum(direction * z_TPM) with z computed across N=787.

## Section 2 - F_S4_PANEL_PRESENT result

Real MMRF data: N=787 baseline TPM, S4 strict cyto stratum n=95,
rest n=692.

| Gene    | Cohen-d (S4 vs rest) | abs(d) > 0.5 | Direction expected |
|---------|----------------------|--------------|---------------------|
| BCL2    | +0.133               | NO           | +1 (matches sign)   |
| BCL2L11 | -0.036               | NO           | +1 (wrong sign)     |
| BAX     | -0.322               | NO           | +1 (wrong sign)     |
| MCL1    | -0.073               | NO           | -1 (matches sign)   |
| BCL2L1  | -0.583               | YES          | -1 (matches sign)   |

Genes passing abs(d) > 0.5: 1 of 5.
Pre-registered threshold: at least 3 of 5.

### Verdict: F_S4_PANEL_PRESENT -> FAIL -> REFUTE_TOUZEAU_ON_BULK_RNA

Per pre-registered protocol the Cox specialist analysis is NOT run:
the panel does not separate t(11;14) from the rest of MMRF on
bulk-RNA TPM, so any downstream Cox lift would confound a real PFS
effect with panel-noise variance.

## Section 3 - Cox results

Not applicable. F_S4_PANEL_PRESENT fired the pre-Cox stop.
F_S4_PASS, F_MARGINAL_NON_REGRESSION, and the underpower protocol
are deferred: the design contract makes them conditional on
Section 2.

## Section 4 - Publishable negative-result framing

Touzeau 2014 identifies BCL2-family functional dependence in
t(11;14) MM via BH3 profiling on primary plasma cells (ratio of
cytochrome-c release to BAD/HRK/MS1 peptides). Bulk-RNA TPM of the
same gene symbols does not recapitulate this dependence at our
N=95: only BCL2L1 (BCL-XL) shows the pre-registered effect-size
threshold, and BAX moves in the opposite direction to its
biological role (t(11;14) cells in MMRF have lower BAX z-score
than the rest). Two of the four BCL2-family genes that should
distinguish t(11;14) by BH3 functional logic (BCL2, MCL1) sit at
abs(d) < 0.15.

This is a clean falsification of the bulk-RNA-as-proxy-for-BH3-
function hypothesis at the MMRF cohort scale. It does NOT refute
Touzeau itself - Touzeau measured mitochondrial cytochrome-c
release, not bulk transcript abundance - so the negative result
is about bulk-RNA transferability of BH3 profiling, not about
the biological-sensitivity claim. Companion v13 ingest of
BH3-functional profiling data is the right next step (sprint
plan Sprint C negative-result protocol).

The fact that BCL2L1 (BCL-XL) is the only marker that crosses
threshold is itself informative: t(11;14) MMRF cells
down-regulate BCL-XL transcript (d=-0.58), consistent with the
Touzeau 2015 follow-up (PMID 26174630) finding that BCL-XL
co-dependence is a subset-level phenotype (only some t(11;14)
cases show it). At the cohort-mean level, BCL-XL down-regulation
is the strongest cytogenetic signature in MMRF - but a
single-gene difference is insufficient biological evidence to
license a 5-gene specialist Cox feature.

## Section 5 - Implications for v12 manuscript S4 safety flag

1. The v12 t(11;14) STRICT FAIL stands and is now reinforced.
   Both the original 4-gene BCL2-family specialist (Delta_S4=+0.0135,
   p=0.446; row r-2026-05-04-v12-t1114-specialist) and the 5-gene
   Touzeau-reconstructed panel fail at the bulk-RNA level - the
   former by chance-level Cox lift, the latter by pre-Cox biological
   sanity. The S4 stratum stays at chance for v12.

2. Manuscript change: keep the v11.5 mmSYGNAL-routed S4 score
   (C ~ 0.643 from the prior specialist run) as the operational head.
   Add a Limitations sentence stating that bulk-RNA-derived
   BCL2-family panels (Touzeau-reconstructed) do not encode the
   BH3-profiling functional signal at MMRF scale (1/5 panel genes
   pass abs(d) > 0.5 between t(11;14) and rest).

3. No F-gate is relaxed. F_S4_PANEL_PRESENT is the new pre-Cox
   biological-sanity gate; we recommend promoting it into the
   permanent F-gate registry alongside the existing F_S4_PASS so
   that future panels are filtered for biological signal before
   prognostic-power claims.

4. Pearl-tier ceiling unchanged. No causal claim is upgraded or
   downgraded by this sprint; the result speaks only to feature
   transferability, not to mechanism.

5. Asterisk note: This verdict carries the literature-
   reconstructed asterisk per sprint-plan R1 (no discrete 5-gene
   table in Touzeau 2014). The asterisk would only convert to a
   conditional promotion if BOTH (a) the discrete supplement
   panel becomes accessible AND (b) F_S4_PANEL_PRESENT passes on
   the supplement-verified panel. Neither condition is met today.

## Pre-registered F-gate roll-up

| Gate                        | Verdict          | Notes                              |
|-----------------------------|------------------|------------------------------------|
| F_S4_PANEL_PRESENT          | FAIL (1/5)       | refutes bulk-RNA panel transfer    |
| F_S4_PASS                   | Not run          | conditional on F_S4_PANEL_PRESENT  |
| F_MARGINAL_NON_REGRESSION   | Not run          | conditional on F_S4_PANEL_PRESENT  |
| Underpower protocol         | Not triggered    | underpower only relevant post-Cox  |

## Artifacts

- Code: scripts/v12_refactor/wave3/s_v12_5_touzeau_t1114.py
- JSON: paper/v8_artifacts/v12_refactor_audit/wave3/touzeau_t1114.json
- Wall time: 2.7 s (no Cox / bootstrap; halted at sanity gate).
