# MORT-FM — Stated Limitations and Claim Guardrails

> Read this **before** quoting any MORT-FM number in a paper, abstract,
> grant, slide, or PR.

## 1. As of this commit, MORT-FM is a code release.

The `v15-mort-fm` branch landed the architecture, data contract, encoders,
fusion, dynamics, landscape, heads, losses, trainer, configs, smoke test,
and 38-test pytest suite. It has **not been trained** on a real MMRF /
Beat-AML / paired single-cell longitudinal cohort.

Therefore the following claims are **NOT permitted** until a real training
run completes:

* "MORT-FM predicts time-to-resistance with C-index X"
* "MORT-FM identifies the resistance pathway"
* "MORT-FM matches/beats PRESCIENT/scNODE/CellRank"
* "MORT-FM is calibrated"
* "MORT-FM generalises to held-out patients"

What you **can** claim from this commit alone:

* "MORT-FM is an implemented architecture that composes a multi-omic
  foundation encoder, a graph-conditioned Neural SDE on a learned
  Waddington potential, and seven prediction heads."
* "The data contract enforces patient-disjoint splits and rejects batches
  missing the supervision a loss requires."
* "Smoke test passes on a 10-patient structural cohort in ~3 seconds on CPU."

## 2. Data-availability constraints

The following modalities are **wired in code** but **require external data**
to actually train:

| Modality                | Loader status   | Data required                                  |
|-------------------------|------------------|------------------------------------------------|
| scRNA-seq               | Implemented      | GSE271107, GSE124310, or MMRF scRNA            |
| scATAC-seq              | Implemented      | Not currently in repo                          |
| DNA methylation         | Implemented      | Not currently in repo                          |
| Histone PTMs (HDAC marks)| Implemented      | Not currently in repo                          |
| Proteomics              | Implemented      | CPTAC MM, CCLE proteomics                      |
| Phosphoproteomics       | Implemented      | Not currently in repo                          |
| MMRF clinical outcomes  | Implemented      | dbGaP / Synapse-gated download                 |
| Drug-target edges       | Implemented      | DrugBank / DGIdb dumps                         |
| CRISPR perturbations    | Implemented      | DepMap CRISPRGeneEffect.csv                    |
| Perturb-seq deltas      | Implemented      | Public Perturb-seq atlases                     |

The model's behaviour on this commit is *structurally correct* — every
forward + backward pass works, every checkpoint round-trips — but no
biological signal has been learned because no biological data has been
trained on.

## 3. Identifiability ceiling

The v10 sprint history (see `memory/project_v10_sprint7_outcome.md`) showed
that with `N_paired = 29` patients, the model could not distinguish
`ED_prior ≈ ED_const` — an identifiability failure expected by the spec
itself. MORT-FM's `MORTFMConfig.min_paired_n_for_longitudinal_claim`
defaults to **100**, and `min_patient_n_for_survival_claim` defaults to
**200**, *to keep us above that floor*. The Claim Critic Agent reads these
fields and refuses to emit longitudinal / survival claims when the actual
training cohort is below them.

## 4. SOTA-comparison constraints

Per the v12 audit (`memory/project_v12_refactor_audit_outcome.md`),
ResistanceMap v12 *tied* baselines on PFS C-index and was *below* v11.5.
**MORT-FM has not been compared to PRESCIENT / scNODE / CellRank / MultiVI
yet** on real data. Any comparison published before head-to-head training on
the same cohort is unsupported.

## 5. Counterfactual head — predictive, not causal

`CounterfactualInterventionHead.rank_interventions` returns
*predictions under counterfactual inputs*, not actual causal effects. For
publication-grade claims of "this intervention will delay resistance" we
need at minimum:

1. CRISPR or drug-screen oracle data for the candidate nodes.
2. Agreement between predicted and observed rankings on held-out
   perturbations.

The `counterfactual_consistency_loss` in `mortfm_losses` is the mechanism
for adding that check, but it has not been validated against any oracle yet.
See the causal-inference-auditor agent.

## 6. Pathway-attribution head — predictions, not mechanisms

`PathwayRouteHead` outputs sigmoid scores over a fixed protein/edge set.
These scores are *correlational explanations of the model's behaviour*,
not validated biological mechanisms. Validation requires:

* Reactome / KEGG overlap precision on known resistance signatures.
* CRISPR knockout agreement for top-ranked nodes.
* Drug-target recovery for the patient's actual drug.

These checks are wired in `resistancemap/evaluation/mortfm/` (placeholder
modules); they have not been populated with real validators yet.

## 7. What to do before any paper claim

1. Acquire MMRF CoMMpass via Synapse.
2. Run `scripts/mortfm_prepare_mmrf.py` (TODO) to build snapshots + outcomes.
3. Train Stage C (cell-line drug-response pretrain) on CCLE/GDSC.
4. Train Stage D (patient adapt) + F (survival) on MMRF.
5. Run full evaluation suite (`evaluation/mortfm/`).
6. Compare against PRESCIENT, scNODE, CellRank, MultiVI on the same cohort.
7. Run the Claim Critic Agent on every numeric claim in the write-up.
