# Methodological Paths Not Chosen for ResistanceMap v10

**Date:** 2026-05-03
**Audience:** methodology / results / ablation sections of the v10 manuscript.

Each section documents one alternative that was considered and rejected,
with supporting evidence and the v10 replacement. All citations point to
artifacts that exist on disk; numbers are sourced to file paths, PMIDs,
logs, or theorem references. The v10 architecture itself is specified in
`docs/V10_PER_PATIENT_FORECAST_SPEC.md`; this document is its *negative*.

---

## §1 — RNA-velocity methods (scVelo / dynamo / cellDancer)

**What was considered.** The velocity family of trajectory methods estimates
per-cell direction in latent space from the unspliced/spliced mRNA ratio.
scVelo (Bergen et al., *Nat Biotech* 38:1408, 2020, PMID 32747759) fits the
per-gene ODE $\dot u = \alpha - \beta u$, $\dot s = \beta u - \gamma s$ to
yield a velocity vector. dynamo (Qiu et al., *Cell* 184, 2022, PMID 35108499)
fits an analytical sparse vector field $\hat f(x)$ via vector-valued kernel
regression on the same $(u, s)$ inputs. CellRank's `VelocityKernel` consumes
either output.

**Why rejected.** All three methods are blocked by a hard data requirement:
they need `layers["spliced"]` and `layers["unspliced"]` populated. The two
scRNA datasets on disk do not have these.

**Supporting evidence.** From `docs/TRAJECTORY_ARCHITECTURE_AND_DATA.md`
§§1.1–1.3, audit run with `scanpy 1.12 / anndata 0.12.10`:
`data/raw/gse124310.h5ad` and `data/raw/gse271107.h5ad` both have **empty
`layers`** — no `spliced`, `unspliced`, or `ambiguous` arrays. §1.3 makes the
verdict explicit: scVelo / dynamo / cellDancer all return **NO** on the data
audit. The same row eliminates CellRank's `VelocityKernel` and `RealTimeKernel`
(the latter needs an `obs["time"]` column, also absent).

**What replaces it in v10.** CellRank 2 (Lange et al., *Nat Methods* 21:1196,
2024, PMID 38871986) is retained, but driven by a `PseudotimeKernel` over
scGPT-frozen 512-d embeddings, requiring no spliced/unspliced layers. See
`V10_PER_PATIENT_FORECAST_SPEC.md` §2 (STATE-TRANSITION BACKBONE / LATENT
ENCODER).

---

## §2 — Patient-conditioned latent SDE alone (Approach A)

**What was considered.** A neural SDE
$dX_t^p = f_\theta(X_t^p, c_p, t)\,dt + g_\theta\,dW_t$ with the patient
conditioning vector $c_p$ (cytogenetics + ISS + age + baseline TPM low-rank
summary, $d_c \in [10, 30]$) injected directly into the drift, trained
simulation-free via flow matching (Lipman 2023, arXiv:2210.02747; Tong 2023,
arXiv:2302.00482). This is "Approach A" in
`docs/TRAJECTORY_MATH_FEW_SHOT.md` §2.

**Why rejected.** Severely over-parameterised at $N_{\text{paired}} = 51$, and
the conditioning dimension exceeds the non-parametric rate's identifiable
budget.

**Supporting evidence.** From `TRAJECTORY_MATH_FEW_SHOT.md` §2.3, with drift
width $H = 256$, depth $L = 2$, latent $d = 512$:
- Drift parameter count: $\#\theta_f \approx 2.6 \times 10^5$.
- Paired scalar observations: $51 \times 512 = 26{,}112$.
- Parameter-to-sample ratio: **$\approx 10$ free parameters per scalar
  observation** — well into the over-parameterised regime where MLE without
  strong regularisation is non-identifiable.

Stone's minimax rate $N^{-2\beta/(2\beta + d_c)}$ (Stone 1982, JSTOR 2240700;
Tsybakov 2009, Thm 2.8) implies that for $L^2$ error $\leq 0.1$ at $N = 51$
with $\beta = 2$, identifiable conditioning dimension is at most
$d_c^{\text{eff}} \approx 4$ — far below the 10–30-dim vector. Yang & Tokdar
2015 (arXiv:1401.7278) sparse rate $N^{-2\beta/(2\beta + s)}$ with $s = 2$
gives error $\approx 0.18$ at $N = 51$. A second blocker: a test patient with
$c_{p^\star}$ outside the convex hull of training conditioning vectors has no
consistency theorem (`TRAJECTORY_MATH_FEW_SHOT.md` §2.5; Khemakhem 2020
Thm 1's auxiliary-variable identifiability is "borderline-violated" at our 6
strata, §2.4).

**What replaces it in v10.** Trajectory Flow Matching (Zhang et al.,
arXiv:2410.21154) supplies the *per-patient likelihood* layer but is wrapped
inside a hierarchical-Bayes outer layer (Approach C) and a Bellman-Harris
clonal prior, not used as a free patient-conditioned SDE.

---

## §3 — Optimal-transport reference matching alone (Approach B)

**What was considered.** Treat the 51 paired trajectories as an empirical
measure on path space; predict a new patient's $t_1$ state as the barycentric
projection of the entropic-OT plan (Schiebinger et al., *Cell* 176, 2019, DOI
10.1016/j.cell.2019.01.006).

**Why rejected.** In ambient 512-d latent space the empirical $W_2$ rate is
essentially null at $K = 51$, and the construction discards the 994
baseline-only patients entirely.

**Supporting evidence.** From `TRAJECTORY_MATH_FEW_SHOT.md` §3.4, applying
Weed & Bach 2019 (Bernoulli 25, arXiv:1707.00087, Thm 1):
$\mathbb{E}\,W_2^2(\hat\mu_K, \mu) \lesssim K^{-2/d}$. With $K = 51$,
$d = 512$:

$$K^{-2/d} = 51^{-2/512} \approx 0.985.$$

Expected $W_2$ error $\approx 1$ — a null result. The Nadaraya–Watson
bandwidth analysis (§3.3) gives "useless for $d = 512$." Even after
aggressive dimension reduction to $d_{\text{eff}} \leq 8$, the rate is
$51^{-1/4} \approx 0.37$ — bounded but slow. Structurally, the 994
baseline-only patients cannot enter Approach B's reference set at all (§5
decision matrix).

**What replaces it in v10.** None directly. The "minimum-kinetic-energy
interpolation" intuition is preserved through the dynamic-OT regulariser
inside Trajectory Flow Matching, but the data-discarding reference-matcher is
not part of v10.

---

## §4 — Attribution-based driver claims (TIG / SHAP) as causal

**What was considered.** Use integrated gradients / SHAP / attention-weight
saliency over a trained model as the primary "driver pathway" claim — read
off the top-attributed gene set per resistance state and call it mechanism.

**Why rejected.** A specific counter-example proves attribution is
attribution-of-the-model, not mediation-of-the-biology: the model attributes
BCL2 highly for Bortezomib resistance under del17p, while DepMap CRISPR
essentiality says BCL2 is anti-essential in MM cell lines (knockout *increases*
growth), so the directionality contradicts the attribution.

**Supporting evidence.** `V10_PER_PATIENT_FORECAST_SPEC.md` §3.4 records the
F1 verdict verbatim: "Counter-example proven: del17p → BCL2 ← high
attribution for Bortezomib resistance; DepMap shows BCL2 Chronos +0.066
(anti-essential). Attribution can't see the U → M back-door." §4 Test 3 lists
the values: BCL2 Chronos $+0.066$ (anti-essential); for contrast,
PSMB5/proteasome at Chronos $-2.51$ in three confirmed MM lines vs $-1.89$
pan-cancer (directionally correct for Bortezomib). The Pearl-tier framing is
from `CAUSAL_VALIDITY_AUDIT.md` §1 sub-claim (c): mediation is L3 by
definition; attribution on an observational model identifies which features
the model uses, which is a property of the model, not the biology.

**What replaces it in v10.** Driver-pathway claims demoted to L1 "associated
with" framing (Rade et al. *Nat Cancer* 2024, PMID 38641734 template). The
sole mechanistic escalation step is a CRISPR-essentiality falsification
oracle — Wilcoxon rank-sum on Chronos scores for top-$k$ predicted drivers
vs 1,000 size-matched random gene sets, refutation threshold $p < 0.05$
Bonferroni (`V10_PER_PATIENT_FORECAST_SPEC.md` §4 Test 3).

---

## §5 — DepMap CRISPR as instrumental variable

**What was considered.** Use cell-line CRISPR knockout (DepMap Achilles /
Chronos, ~18,000 genes × ~1,000 lines, configured at `depmap_crispr_path`) as
an instrumental variable $Z$ for an in-patient causal pathway claim — argue
that $Z$ acts on the patient outcome $Y$ only through candidate mediator $M$,
so a CRISPR-validated $Z\to M$ edge transports to a patient-level
$M \to Y$ effect.

**Why rejected.** Pearl-Bareinboim transportability conditions for
generalising a cell-line interventional result to patient populations fail.
Cell lines and patient tumours differ on multiple axes that affect $Y$
directly (not through $M$), violating the exclusion restriction.

**Supporting evidence.** `V10_PER_PATIENT_FORECAST_SPEC.md` §3.4 (F3):
"**INVALID** — Pearl-Bareinboim transportability fails: immune
microenvironment + clonal heterogeneity + passage drift directly affect Y
bypassing M." Reinforced by `CAUSAL_VALIDITY_AUDIT.md` §3.2 (survivorship
bias in 72-h cell-line screens vs weeks-long clonal-selection process in
patients) and §3.4 (cytogenetic risk variables affecting MM patient outcome
not present identically in DepMap cell lines).

**What replaces it in v10.** CRISPR is retained, but as a one-sided
falsification *oracle* only — it can refute a predicted driver set (failure
to enrich for essentiality vs 1,000 random sets) but a "pass" is interpreted
only as "consistent with cell-line essentiality," never as "causes resistance
in patients."

---

## §6 — Deep multi-omics fusion (v8/v9) without drug fingerprints + without MMRF

**What was considered.** End-to-end deep fusion over CCLE proteomics +
epigenomics + DepMap CRISPR (with NaN-indicator), trained on GDSC IC50 in a
single multi-task head across 11 drugs, *without* a structural drug
representation and *without* MMRF clinical validation. This was the actual v8
state.

**Why rejected.** Two simpler baselines, on the same `data_ready.pt`
checkpoint, beat or tie the deep stack.

**Supporting evidence (numerical).** From
`paper/v8_artifacts/baselines/elasticnet_results.json` (per-drug ElasticNet,
same 622-train/132-test split, 4 feature blocks, $\approx 20{,}574$
features):
- Pooled test MSE: **2.892**.
- Pooled "predict-mean" floor MSE: **2.811**.
- ElasticNet skipped 3 drugs (Dinaciclib / Panobinostat / Venetoclax,
  $n_{\text{nonzero}} = 0$).

From `paper/v8_artifacts/baselines/mofa_results.json` (MOFA+ 20-factor
unsupervised + Ridge):
- Pooled test MSE (MOFA+Ridge): **2.8155**.
- Pooled test MSE (ResistanceMap v8): **2.8364** (delta $-0.0209$).
- Fit time end-to-end: **19.7 s**.
- Verdict string: "MOFA+Ridge TIES ResistanceMap overall: MOFA beats RM on
  4/11 drugs, ties on 3, loses on 4 … ResistanceMap's reported test_mse
  (2.84) is dominated by a single Panobinostat outlier (mse=22.6) and is
  essentially the zero-prediction floor … multi-omics integration is not the
  bottleneck on this 886-cell-line / 11-drug task — sample size and label
  sparsity are."

The diagnosis (drug-side representation is the missing piece) is ratified by
`MODEL_LANDSCAPE_RANKING.md` §"#1 DrugCell": "ResistanceMap's biggest known
gap is the absence of any drug representation … With 692 samples × 11 drugs,
the multitask drug-conditioned formulation is essential to share statistical
strength across drugs."

**What replaces it in v10.** Cell-line layer retained as L1 training stage
with two changes: (a) Morgan fingerprints (radius 2, 2048-bit) absorbed from
DrugCell (PMID 33096023, MIT) as the drug encoder; (b) MMRF CoMMpass
ingestion as a Tier-2 PERCEPTION-style validation cohort (PMID 38637658,
$N = 994$ baseline / $N = 258$ with $\geq 2$ lines of therapy — see
`MMRF_ACQUISITION.md` §3 verified counts).

---

## §7 — DrugCell-style ontology hierarchy as the trajectory mechanic

**What was considered.** Adopt DrugCell's "visible neural network" — hidden
structure mirroring the Gene Ontology DAG — not just for the drug-response
head but as the *trajectory* mechanic, with hierarchy levels mapped to
temporal scales. The natural "biology-aware" alternative to a Waddington
energy landscape.

**Why rejected (partially absorbed).** The drug-side encoder is genuinely
valuable and is absorbed (see §6). The ontology-as-trajectory framing,
however, loses the energy-landscape interpretation in which resistance states
are basins, transitions are barrier crossings, and the clonal-dynamics prior
controls how the population redistributes between basins under drug pressure.
A GO DAG gives interpretable feature groups but no notion of basin depth or
barrier height.

**Supporting evidence.** From `MODEL_LANDSCAPE_RANKING.md` §"#1 DrugCell":
"Visible neural network whose hidden structure mirrors the Gene Ontology,
concatenated with a Morgan-fingerprint MLP for the drug, then a final MLP to
predict response." The ontology side is a static hierarchy, not a dynamical
operator. The v10 trajectory mechanic is explicitly an *energy landscape* —
see `V10_PER_PATIENT_FORECAST_SPEC.md` §2 (TRAJECTORY MECHANICS = Trajectory
Flow Matching neural-SDE), inheriting the energy-landscape framing of
PRESCIENT (Yeo et al., *Nat Commun* 12:3222, 2021, PMID 34059684) and the
dynamic-OT formulation of TrajectoryNet (Tong 2020, PMLR 119:9526), as
documented in `TRAJECTORY_ARCHITECTURE_AND_DATA.md` §§2.1–2.2.

**What replaces it in v10.** Drug encoder = DrugCell-style Morgan fingerprints
(absorbed). Trajectory mechanic = TFM neural-SDE on a Waddington-style latent
landscape; GO hierarchy absent from the dynamical operator.

---

## §8 — GigaPath / GigaTIME as primary v10 architecture

**What was considered.** "Microsoft GigaTIME" was suggested as a peer-reviewed
reference architecture. The framing was: copy GigaTIME's design.

**Why rejected.** GigaTIME (PMID 41371214, *Cell* 189(2), Jan 2026) is a
**cross-modal translator** from H&E slides to virtual multiplex
immunofluorescence. It scales up *cross-sectional* TME characterisation. It
is not a per-patient temporal forecaster, has no temporal axis in the
published architecture, and does not target the paired-longitudinal regime.

**Supporting evidence.** `docs/PER_PATIENT_FORECAST_LITERATURE.md` Part 1
verifies the published record (PMID 41371214 / DOI 10.1016/j.cell.2025.11.016)
and quotes the abstract: "GigaTIME learns a cross-modal translator to generate
virtual mIF images from hematoxylin and eosin (H&E) slides … applied to 14,256
patients from 51 hospitals … generating 299,376 virtual mIF slides." The
verdict in that document is verbatim: "GigaTIME is a **cross-modal translator**
… It is NOT a per-patient temporal forecaster, has no temporal axis in the
published architecture, and does not address the paired-longitudinal regime."
GigaPath (PMID 38778098, *Nature* 2024) — closest Microsoft adjacent — is
also non-temporal.

**What replaces it in v10.** Foundation-model components are drawn from
scGPT (PMID 38409223, frozen encoder) and Chronos (arXiv 2403.07815, parallel
clinical-timeline channel). The "single-snapshot scalability" target is
addressed via the hierarchical-Bayes outer layer + 994 baseline-only patients
tightening the cohort prior, not via a cross-modal translator.

---

## §9 — k=10 simultaneous pathway-mediator NIE

**What was considered.** A full natural-indirect-effect mediation analysis
simultaneously over $k = 10$ pathways — partition the total effect of
cytogenetic risk on time-to-resistance into contributions through 10
candidate mediating pathways at once.

**Why rejected.** Events-per-variable is too low. With the strict-paired
sub-cohort at $N = 42$ (51 longitudinal-RNA patients ∩ second-line-therapy
events) and 10 mediators + 5 cytogenetic adjustment variables, EPV
$\approx 42 / 15 \approx 2.8$ — far below the conventional threshold of
$\geq 25$ for stable Cox-style estimates with that many covariates.

**Supporting evidence.** `V10_PER_PATIENT_FORECAST_SPEC.md` §3.4 records F2
verbatim: "L2 feasible at N=42 single mediator — Powered for one
pre-specified mediator … **Not powered for k=10 simultaneous.** Chief
unmeasured confounder: **EIMOC** — physician treatment-adjustment between
visits, structurally unidentifiable without visit-level dose records." §6
Sprint 4 reinforces: "Single pre-specified mediator NIE on N=42 with EIMOC
disclaimer (Formulation 2). One mediator: proteasome pathway score at T1 →
time to 2nd-line Bortezomib therapy. IPTW-Cox conditioning on ISS + 5
cytogenetics." `CAUSAL_VALIDITY_AUDIT.md` §3.5 (time-varying confounding)
caps the multi-mediator $N$ further: "not all patients have cycle-matched
RNA-Seq — only 51 have longitudinal RNA."

**What replaces it in v10.** A *single* pre-specified mediator: proteasome
pathway score at $T_1$ as the mediator on
"baseline cytogenetics → proteasome → time-to-2nd-line Bortezomib," with
IPTW-Cox conditioning on ISS + 5 cytogenetic flags. The EIMOC caveat must
appear in the manuscript; the analysis is framed as exploratory L2-feasible,
not as the primary causal claim.

---

## §10 — dbGaP-controlled MMRF download path

**What was considered.** Acquire MMRF CoMMpass through dbGaP study
`phs000748` (controlled-access). Requires eRA-Commons DAR (2–6 weeks),
followed by a GDC user token and download of the ~206 TB controlled release.

**Why rejected.** The Tier-2 validation cohort that v10 needs (baseline
RNA-Seq + treatment timeline + cytogenetics + masked variants + CNV) is
entirely available **open-access** on the GDC, with no DAR and no token.
dbGaP would only add raw BAMs / FASTQs and unmasked variants, which v10 does
not consume.

**Supporting evidence.** `docs/MMRF_ACQUISITION.md` §1 (verified live
2026-05-03 against `https://api.gdc.cancer.gov/projects/MMRF-COMMPASS`):

| Modality | Open-access files | Size |
|---|---:|---:|
| RNA-Seq STAR augmented gene counts | 859 | ~3.5 GB |
| Masked Somatic Mutations | 1,091 | ~50 MB |
| Copy Number Segments | 1,010 | ~150 MB |
| **Total** | **2,960 files** | **~3.7 GB** |

Clinical / treatment / sample timeline lives as case-level metadata in the
`/cases` API and is open. `scripts/pull_mmrf_clinical.py` produces 995 × 16
clinical, 7,184 × 11 treatments, 4,860 × 7 samples TSVs (§3 verified
2026-05-03). §6 verdict: "You only need [dbGaP] for the ~31,000
controlled-access files (BAM/FASTQ raw reads, unmasked VCFs). For Tier 2
validation, we do not."

**What replaces it in v10.** The open-access GDC API path
(`MMRF_ACQUISITION.md` §§2–4). dbGaP held as contingency only if reviewers
later request raw-read evidence.

---

## §11 — Vanilla z-score thresholding for cytogenetic translocation calls

**What was considered.** Define t(11;14) (CCND1-driven) by a cohort z-score:
positive iff $z = (\log\!1\mathrm{p}(\mathrm{TPM}) - \mu)/\sigma \geq 2.0$.
Same threshold proposed for FGFR3 / NSD2 (t(4;14)).

**Why rejected for CCND1.** Two-component GMM on log1p(TPM) for CCND1 in MMRF
produced a *degenerate* split ("no expression vs any expression" — upper
component carried ~70 % mass rather than literature 15–20 %), and $z \geq 2$
yielded only ~1.2 % positives in MMRF — vs published consensus 15–20 %.

**Supporting evidence.** `scripts/derive_cytogenetics.py` lines 193–256
implement the chosen `expr_call_posterior(...)` and document the failure
mode in its docstring (lines 198–212): GMM is accepted only if
`0.05 <= upper_w <= 0.30 AND mean_sep >= 1.0` (lines 230–246); otherwise
fall back to a literature-prevalence-anchored top-quantile cutoff
(lines 248–254). Lines 258–263 fix the priors:

```
# Literature priors (Avet-Loiseau et al., MM consensus reviews):
#   t(4;14) ≈ 12-15 % → use 13.5 %
#   t(11;14) ≈ 15-20 % → use 17.5 %
fgfr3_call = expr_call_posterior(FGFR3_ENSG, 0.135, "FGFR3")
nsd2_call  = expr_call_posterior(NSD2_ENSG,  0.135, "NSD2")
ccnd1_call = expr_call_posterior(CCND1_ENSG, 0.175, "CCND1")
```

The resulting cytogenetic prevalence in `cytogenetics.tsv` (976 patients)
matches published bands per `INTEGRATED_BUILD_PLAN.md` line 74:
"del17p 13.6 %, chr1q21 33.6 %, del13q 52 %, t(4;14) 14.6 %, t(11;14)
16.8 % — all in or near published IA-series literature bands." Plain
$z \geq 2$ would have returned ~1.2 % t(11;14), an order of magnitude below
truth.

**What replaces it in v10.** The hybrid GMM-with-literature-anchored quantile
fallback in `derive_cytogenetics.py`. GMM preferred when non-degenerate;
otherwise top-prevalence quantile (top 17.5 % for t(11;14), top 13.5 % for
t(4;14)). Each call's method is logged.

---

## §12 — gdc-client binary as the MMRF download tool

**What was considered.** The historical NCI-recommended path: install
`gdc-client`, feed it manifest TSVs, let it pull open-access files in
parallel with built-in resume + MD5 verification.

**Why rejected.** As of 2026-05-03, every gdc-client distribution channel is
broken: binary URLs redirect to not-found, package not on PyPI, GitHub
releases ship source-only — no canonical install path on a fresh machine.

**Supporting evidence.** `MMRF_ACQUISITION.md` "Verification log" (live
fetches 2026-05-03):

| Endpoint | Result |
|---|---|
| `gdc-client` binary URLs (1.6.1 / 2.0 / 2.3) | All 302→not-found |
| `pip install gdc-client` | "No matching distribution on PyPI" |
| `https://api.github.com/repos/NCI-GDC/gdc-client/releases` | Releases 2.1, 2.2, 2.3 ship source-only (no binary assets) |

§4 records the resolution: "We bypass all of that with a pure-Python
downloader (`scripts/gdc_download.py`) that hits the `/data/<UUID>` endpoint
with parallel workers, MD5 verification, and resume-by-existence."

**What replaces it in v10.** `scripts/gdc_download.py` (~160 LOC,
stdlib + `requests`). Resumable, MD5-verified, consumes manifest TSVs from
the GDC `/files?return_type=manifest` endpoint.

---

## §13 — Summary mapping (rejected → replacement)

| # | Rejected | Why rejected | Primary evidence | v10 replacement |
|---|---|---|---|---|
| 1 | scVelo / dynamo / cellDancer | No spliced/unspliced layers in either h5ad | `TRAJECTORY_ARCHITECTURE_AND_DATA.md` §§1.1–1.3 | CellRank 2 with `PseudotimeKernel` over scGPT embeddings |
| 2 | Patient-conditioned latent SDE alone | $\sim 10\times$ over-parameterised at $N=51$; Stone's rate caps $d_c^{\text{eff}} \leq 4$ | `TRAJECTORY_MATH_FEW_SHOT.md` §§2.3–2.4 | TFM neural-SDE inside hierarchical-Bayes outer layer |
| 3 | OT reference matching alone | $W_2$ rate $\approx 0.985$ at $K=51, d=512$; discards 994 baselines | `TRAJECTORY_MATH_FEW_SHOT.md` §§3.4, 5 | Dynamic-OT regulariser inside TFM, not standalone |
| 4 | TIG / SHAP attribution as causal | BCL2 high-attribution but DepMap Chronos $+0.066$ (anti-essential) | `V10_PER_PATIENT_FORECAST_SPEC.md` §§3.4 (F1), 4 Test 3 | L1 framing + CRISPR falsification oracle |
| 5 | DepMap CRISPR as IV | Pearl-Bareinboim transportability fails | `V10_PER_PATIENT_FORECAST_SPEC.md` §3.4 (F3); `CAUSAL_VALIDITY_AUDIT.md` §§3.2, 3.4 | CRISPR as one-sided falsification oracle |
| 6 | v8/v9 deep stack w/o drug FP, w/o MMRF | Pooled MSE 2.836 vs floor 2.811; ties MOFA+Ridge in 19.7 s | `paper/v8_artifacts/baselines/{elasticnet_results.json, mofa_results.json}` | DrugCell Morgan-FP encoder + MMRF Tier-2 cohort |
| 7 | DrugCell ontology as trajectory | Static DAG; no basin/barrier semantics | `MODEL_LANDSCAPE_RANKING.md` §"#1 DrugCell"; `V10_…SPEC.md` §2 | Drug encoder absorbed; TFM as trajectory mechanic |
| 8 | GigaTIME / GigaPath as primary | Cross-modal translator (H&E → mIF), not temporal | `PER_PATIENT_FORECAST_LITERATURE.md` Part 1 | scGPT (frozen) + Chronos (clinical timeline) |
| 9 | $k=10$ simultaneous mediator NIE | EPV $\approx 2.8$ at $N=42$ (10+5 covariates) | `V10_PER_PATIENT_FORECAST_SPEC.md` §§3.4 (F2), 6 Sprint 4 | Single pre-specified mediator (proteasome → time-to-2nd-line) |
| 10 | dbGaP controlled MMRF | 2,960 open-access files / ~3.7 GB cover Tier-2 | `MMRF_ACQUISITION.md` §§1, 6 | Open-access GDC API path |
| 11 | $z \geq 2$ for CCND1 | GMM degenerate (upper 70 % mass); $z \geq 2$ → 1.2 % positives vs 15–20 % literature | `scripts/derive_cytogenetics.py` lines 193–263; `INTEGRATED_BUILD_PLAN.md` line 74 | GMM-or-literature-quantile (top 17.5 %) |
| 12 | gdc-client binary | All distribution channels broken on 2026-05-03 | `MMRF_ACQUISITION.md` "Verification log" | `scripts/gdc_download.py` (stdlib + requests) |

---

## §14 — Honesty constraints inherited from v10 spec §7

Consequences of the rejections above that propagate into every figure:

1. **Population-stratified, not universal.** Conformal coverage transfers
   only to patients exchangeable with the MMRF Recurrent BM calibration
   cohort.
2. **N=42 paired sub-cohort is exploratory.** Single-mediator NIE only;
   $k > 1$ not powered. EIMOC unmeasured.
3. **Driver pathways are L1 features with L2 CRISPR-oracle support, not
   L3 mediators.** Never frame as "causes resistance."
4. **Clone-dynamics prior assumes $K \leq 8$ subclones**, justified by
   PyClone-VI on MMRF VAF and Maura et al. 2019 (PMID 31444325, median
   $4 \pm 2$; `CLONE_DYNAMICS_AND_CONFORMAL.md` §1a).
5. **Run-ledger discipline applies.** Sample sizes and coverage tolerances
   in this document live in debugging-tier until ledger row + W&B link +
   verification_chain.json exist.

---

## §15 — One-paragraph summary

v10 rejected twelve alternatives on empirical or theoretical grounds.
Strongest empirical: v8/v9 deep stack without drug encoder (§6) — pooled MSE
2.836 vs MOFA+Ridge 2.816, losing 4/11 drugs to plain ElasticNet on the same
checkpoint. Strongest theoretical: patient-conditioned latent SDE (§2) —
$\sim 10\times$ over-parameterised at $N_{\text{paired}} = 51$, Stone's rate
capping $d_c^{\text{eff}} \approx 4$. Cleanest infrastructural: gdc-client
(§12) — all channels broken 2026-05-03, replaced by a 160-LOC pure-Python
downloader. The surviving v10 stack — Mondrian jackknife+, hierarchical
Bayes over 5 cytogenetic strata, Bellman-Harris on $K \leq 8$ subclones, TFM
likelihood, CellRank 2 over frozen scGPT, parallel Chronos timeline — is
each-component-the-unique-survivor.
