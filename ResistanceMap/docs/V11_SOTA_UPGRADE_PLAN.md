# v11 SOTA Upgrade Plan — Multi-Omic Foundation Model for Epigenetic Drug-Resistance Trajectories in Hematologic Malignancies

**Date authored:** 2026-05-03
**Author:** Architect (post-v10 closeout)
**Status:** Plan only — no code or doc claims migrate to README/ARCHITECTURE until each stage produces a `RUNS.md` row.

---

## §0 First principle: what v11 is, and what it is NOT

v11 is **NOT** a re-launch of per-patient temporal forecasting at N_paired=29.
The v10 spec §1 + §11 abstract pre-declared, and S7 F8 strict-fail empirically
confirmed, that the Stone minimax bound caps identifiable drift dimension at
≈4 for N_p≤50. **v11 cannot manufacture identifiability the data does not
support.** Any architecture (Neural-ODE-on-Waddington, scGPT fine-tune, GNN-
augmented PPI propagation) that *appears* to break this bound at N_paired=29
is overfitting and must be flagged by the existing falsification gates.

v11 IS a **regime-aware foundation-model upgrade** that:

1. **Preserves verbatim** the v10 falsification framework (F1–F10, F_S5, F8-DPS) — these are the unique contribution and are NOT touched. The pre-registered REFUTED outcomes (F2 post-leak, F5-paired, F10 E-value, F8 DPS LOO) remain REFUTED in v11; the upgrade is allowed to *strengthen* refutations or *narrow* the regime where the prior is informative, never to *flip* a refuted gate by changing the test.
2. **Upgrades the prior** from a hand-MLP U_θ trained on N=787 cross-sectional baselines to a foundation-model prior fine-tuned on a curated MM/AML cohort (scFoundation/Geneformer + LoRA), then projected back into the U_θ landscape via DSM. Expected metric impact: F4 curl fraction tightens (already 0.0151), F6 driver-recall lift (currently 30.7σ at p=0.0001 — already strict-pass, but T_obs/T_null gap may widen by 5-15%).
3. **Adds a Neural-ODE-on-Waddington** integrator (`torchdiffeq.odeint` over -∇U_θ) so that the forward operator is mathematically a flow on the learned potential, rather than the current single Euler step. F4 (Helmholtz curl) is *theorem-equivalent* to "the integrator is gradient-flow", so this upgrade makes the F4 measurement honest rather than approximate.
4. **Wraps PPI propagation** in a graph-attention (GAT-2) layer over the same STRING v12 full-channel graph, providing a learnable diffusion kernel that generalizes RWR. F3 strict-pass (6/10 both panels) and F6 strict-pass (30σ) must replicate at p≤ current strict thresholds; the GAT version is rejected if the seed→target signal is *worse* on either oracle. (Zero-trust gate.)
5. **Wires AgentOps three-pillar telemetry** into every stage of the pipeline so observability/evaluation/optimization metrics are emitted for every run automatically, not generated post-hoc.
6. **Ships a multi-agent orchestration layer** that uses the existing specialized subagents (sota-comparator, literature-deep-researcher, causal-inference-auditor, …) and MCP servers (PubMed, ChEMBL, Open Targets, ClinicalTrials.gov, bioRxiv, HuggingFace) for evidence retrieval, grounding, and SOTA benchmarking — replacing the `connectors.py` stubs with real MCP clients.

The net deliverable is the same v10 paper **with a stronger prior and an honest forward operator**, plus a fully traced/evaluated/optimizable pipeline. No claim escalates from L1-with-structural-prior to L2-interventional.

---

## §1 What stays UNTOUCHED (the unique contribution)

| Component | Where | Why untouched |
|---|---|---|
| Pre-registration of falsification gates F1–F10, F_S5, F8-DPS | `docs/V10_FOUNDATION_MODEL_PAPER_SPEC.md` §9 | Pre-declared in dated spec; mutating these is p-hacking. |
| Bounded-claims rule (no number in README without RUNS.md row) | `RUNS.md` §"Rule" + `scripts/check_docs_consistent.py` | Release-bouncer enforced; CI gate. |
| Refuted-gate verdicts (F2, F5-paired, F10, F8-DPS) | `docs/V10_SPRINT{2,4,7}_VERDICT.md` | These ARE the central scientific findings (identifiability limits + L1 ceiling). |
| Mondrian per-stratum conformal coverage (Sprint 5) | `scripts/v10/s5_mondrian_jackknife_plus.py` | Already strict-pass exceeding spec; no architectural reason to touch. |
| Cross-disease F3 refutation on Beat AML (Sprint 6) | `scripts/v10/s6b_v2_aml_native.py` | The refutation IS the empirical bound on cross-disease transfer. |
| Sprint orchestrator semantic stage names | `resistancemap/v10_runner.py` | Stable contract; v11 stages get appended, v10 names preserved. |
| Lange-Hansen Cox NIE focused-mediator pipeline (Sprint 4) | `scripts/v10/s4d_*.py`, `s4e_v2_focused_neg_control.py` | F8/F9 strict-pass + F10 structural-fail is the primary mediation finding. |

**Rule:** if v11 work flips any of these to a different outcome, that is an
audit-failure event and triggers a `fabrication-sentinel` review before a
single number is released.

---

## §2 Component-by-component upgrade table (with first-principles justification)

Each row lists: (a) current implementation, (b) v11 upgrade, (c) first-
principles justification, (d) expected F-gate impact, (e) zero-trust check.

| # | Component | Current | v11 upgrade | First-principles justification | Expected gate impact | Zero-trust check |
|---|---|---|---|---|---|---|
| 1 | **Multi-omic encoder** | `s1c_encode_mmrf_z64.py`: PCA(64) on baseline log-TPM | scFoundation embedding + LoRA fine-tune on N=787 MM (frozen base, learn 64-dim adapter) — see HF `whxhx/scFoundation`-class | A foundation model trained on 50M cells captures gene-gene covariance not in PCA's linear subspace; LoRA adapter is rank-deficient → low overfit risk at N=787 | F4 (cleaner curl), F6 (T_obs↑) | Re-run F2 (994-baseline ablation): if F2 *passes* at v11 where it REFUTED at v10, the adapter is leaking train-on-test → reject and revert |
| 2 | **U_θ scalar potential** | `s1d_train_scalar_potential.py`: 64→256→256→128→1 MLP, DSM + PPI Tikhonov | Same DSM target on adapter latents. **NO projection layer needed** — see correction below | The previous "Helmholtz projection" framing was mathematically redundant: ∇ × ∇f = 0 is a vector calculus identity for any twice-differentiable scalar f, so the gradient-of-scalar parameterization is curl-free *by construction*, not by regularization. F4 measures the curl of (empirical drift − model drift), which is the rotational *data residual* — i.e., the part of the empirical flow that gradient-of-scalar cannot capture. The PPI Tikhonov term controls the *gradient's smoothness on the gene graph*, not its curl. **See `docs/V11_SPRINT1_ODE_VERDICT.md` §1 for the corrected framing.** | F4 stays PASS (0.0151), framing now honest (was "Tikhonov-suppressed curl", now "identity-curl-free model + data residual measurement") | If anyone "adds a Helmholtz projection layer" to U_θ they have introduced a redundant nn.Module and should be reverted |
| 3 | **Forward operator** | Single Euler step: z(t1)=z(t0)−η∇U_θ + σε | `torchdiffeq.odeint(method="dopri5", rtol=1e-5, atol=1e-7)` integrating ż = -∇U_θ(z) over t∈[0, η]. **LANDED 2026-05-03** as `resistancemap/landscape/neural_ode_flow.py`. | Single Euler is a 1st-order quadrature; v11 diagnostic measured the v10 operator at η=2 to be 2.72 L2 units off the true ODE flow (`paper/v8_artifacts/v11_sprint1/ode_forward_diagnostic.json` D2). Adaptive Dopri5 is order-5 with err ≤ rtol·‖z‖+atol. Lyapunov M1 (dU/dt = −‖∇U‖² ≤ 0) verified: 0/29 patients × 4 T-values = 0/116 violations. | F8-DPS expected-fail at N_p=29 (regime-bound, not operator-bound) — **CONFIRMED 2026-05-03**: v11 Neural-ODE re-run gives 0/16 F8 pass, best p_vs_const=0.402 (vs v10 0.432); F8 refutation is robust to operator choice | If F8-DPS passes at N_p=29 with the Neural-ODE operator, that contradicts Stone — audit, do not publish |
| 4 | **PPI propagation** | RWR (closed-form `(I − αW̃)⁻¹` solve) | GAT-2 layer over STRING with 1-hop attention, RWR as inductive bias (init heads to RWR row-stochastic kernel) | RWR = 1 / (1 − αP); a learned attention kernel A_ij = softmax(QK^T/√d) generalizes the row-stochastic transition while preserving the closed-form recovery (set Q=K=I, scaling = α). Provides *learnable* diffusion. | F3 ≥ current 6/10; F6 ≥ current 30.7σ (zero-trust strict bound) | If GAT does NOT match RWR baselines on ≥1 panel, revert. GAT must dominate, not tie. |
| 5 | **Hierarchical Bayes** | NumPyro 5-stratum HMC (Sprint 2 inner) | Same model; ADD posterior-predictive checks (Gelman 1996 §6) for each stratum before F1 | Bayesian p-values flag prior–posterior mismatch; without them F1 strict-pass is necessary but not sufficient | F1 stays pass; new sufficiency check | If PPC fails ≥1 stratum, F1 verdict downgrades from "pass" to "pass with caveat" — does NOT delete F1 |
| 6 | **Conformal layer** | jackknife+ Ridge base, Mondrian by 5 cyto strata (Sprint 5) | Replace Ridge with the v11 GAT+ODE forward as residual model; jackknife+ unchanged | Theorem (Barber et al. 2021): jackknife+ coverage guarantee is base-learner-agnostic. We may swap the base learner for tighter intervals as long as we don't change the conformal step. | F_S5 still strict-pass; expected interval width ↓20–40% (sharper but still calibrated) | If marginal coverage drops below 1−α−1.96·√(α(1−α)/n), revert |
| 7 | **Mediation NIE** | Cox PH single-mediator focused (M_seed3) | UNCHANGED. Add E-value tipping-point sensitivity *as a paper figure*, not a re-test | F10 structural-fail is biology, not algorithm; re-fitting won't change E=1.20 | F10 stays REFUTED | Re-running with v11 priors must reproduce F10 fail to within ±0.05 E-value |
| 8 | **DPS single-snapshot wrapper** | Sprint 7 Chung-style, η×σ sweep | UNCHANGED at N_p=29; the v11 paper claims this primitive is honest *only* at N_p ≥ Stone bound and computes the projected bound (≈150 for d=64) | F8-DPS STRICT FAIL is the v10 headline | F8 stays FAIL | If F8 passes, Stone bound is wrong or there's a leak |
| 9 | **Cross-disease scope** | Beat AML 1.0 (Tyner 2018), AML-native expression seeds REFUTED | Add LAML-TCGA (PMID 23634996, 200 cases, RNA-seq + mutation), reuse cross-disease F3 framework | Two AML cohorts independently REFUTING expression-driven F3 is stronger evidence than one | Cross-disease F3 still refuted; AUC across two cohorts | If TCGA-LAML gives p<0.01 cross-disease F3 pass, revisit MM-specificity claim |
| 10 | **Causal layer** | Pearl L1-with-structural-prior | UNCHANGED. v11 adds a `causal-inference-auditor` agent gate that runs before each release to verify no language has slipped from L1 to L2 | Pearl ladder is hierarchical; promotion requires intervention or counterfactual data we don't have | No metric change; doc/release governance only | Auditor must sign off before `git tag` |

---

## §3 Multi-agent architecture (parallelism, MCP servers, HuggingFace)

### §3.1 Agent task graph

Sprint upgrades parallelize across **independent omic streams** and **independent
falsification gates**. Stage names use the v10 semantic-stage convention and
add v11 prefixes only when the v10 verdict is preserved.

```
                                  ┌─ literature-deep-researcher ─┐
                                  ├─ sota-comparator ────────────┤  → SOTA table
                                  └─ evidence-curator ───────────┘    (parallel)

  data-integration-validator ──┐
                               ├─→ proteomics-pathway-validator ─→ trajectory-dynamics-specialist ─→ architecture-chair
  fabrication-sentinel ────────┤   (PPI + RWR/GAT validation)         (Neural-ODE Waddington)        (final verdict)
                               └─→ causal-inference-auditor ─→ clinical-translator
                                                              (release-bouncer gate)
```

The graph is partial-DAG: research agents run in parallel; validators serialize
on the artifacts they share. The chair runs LAST, after all sub-verdicts are
available.

### §3.2 MCP server wiring (replaces `connectors.py` stubs)

| Purpose | MCP server | Used by |
|---|---|---|
| Drug MoA, IC50, ADMET | `claude.ai_chembl` | proteomics-pathway-validator, evidence-curator |
| Trial endpoints, eligibility | `claude.ai_c-trials` | clinical-translator |
| Primary literature, citation backing | `claude.ai_PubMed` | every agent that produces a citable claim |
| Preprints (latest methods) | `claude.ai_bioRxiv` | sota-comparator, literature-deep-researcher |
| HuggingFace foundation models + datasets | `claude.ai_Hugging_Face` | encoder upgrade (component #1) |
| Target druggability, GWAS evidence | `claude.ai_Open_Targets` | proteomics-pathway-validator |

**Action:** delete the empty-array stubs in `resistancemap/mcp_integration/connectors.py` and replace with thin clients that wrap the MCP tool calls. Cache responses per-trace via `tracer.span("mcp_call")`.

### §3.3 HuggingFace integration

| Need | In-house vs HF | Justification |
|---|---|---|
| Single-cell / bulk RNA encoder | **HF** (scFoundation, Geneformer, scGPT) | Pre-trained on 50M+ cells; in-house 787-patient training cannot compete on representation quality |
| Protein sequence / structure | **HF** (ESM-2, ProtT5) | Same |
| MM/AML drug-response head | **In-house** | No public foundation model fits PRISM/CCLE/DepMap drug-cell-line matrix; LoRA on top of a frozen scFoundation backbone is the right size |
| Conformal layer | **In-house** | Tight coupling to Mondrian strata; no off-the-shelf library |
| U_θ landscape MLP | **In-house** | 64→1 head; trivial to train, not a foundation-model fit |

### §3.4 Worktrees

Each upgrade row in §2 gets an isolated git worktree (`git worktree add wt/v11-encoder`, `wt/v11-ode`, `wt/v11-gat`). The `architecture-chair` agent merges them only after each independently passes its zero-trust check.

### §3.5 Hooks

`fabrication-sentinel` already runs on edits to `resistancemap/evaluation/`, `resistancemap/data/`, `scripts/`, `paper/`. v11 adds:
- pre-edit hook on `resistancemap/agentops/` to emit a tracer event;
- pre-commit hook calling `release-bouncer` to verify no synthetic numbers leak.

---

## §4 AgentOps three-pillar wiring

### §4.1 Observability (already scaffolded in `resistancemap/agentops/tracer.py`)

| Metric | Where measured | Storage |
|---|---|---|
| End-to-end trace duration | `Tracer.start_trace()` / `Trace.end()` wraps every v11 stage | `logs/agentops/<trace_id>.jsonl` |
| Agent-to-agent handoff latency | `Tracer.trace_handoff(from_agent, to_agent)` instrumented at every subagent dispatch | same |
| Cost per request | LLM token counter (input+output × model price) per `Span` | `Span.metadata["cost_usd"]` |
| Tool execution latency | Per-MCP-call span around every connector method | `Span.duration_ms` |

### §4.2 Evaluation (already scaffolded in `resistancemap/agentops/evaluator.py`)

| Metric | Definition for v11 | Acceptance threshold |
|---|---|---|
| Task completion rate | (sprint stages exiting 0) / (sprint stages launched) | ≥ 0.95 |
| Guardrail violation rate | fabrication-sentinel + release-bouncer hits / total commits | 0 (any hit blocks release) |
| Factual accuracy rate | sample of 50 paper claims auto-checked against PubMed/ChEMBL via evidence-curator | ≥ 0.95 |
| Clinical appropriateness | clinical-translator review score on top-N drug recommendations | ≥ 0.90 |
| First-pass approval rate | v11 PRs merged without architect-chair revision | ≥ 0.70 |

### §4.3 Optimization (already scaffolded in `resistancemap/agentops/optimizer.py`)

| Metric | Definition | Target |
|---|---|---|
| Prompt token efficiency | (output_tokens) / (input_tokens) per agent task | ≥ 0.30 (i.e. ≤ 3× prompt expansion) |
| Retrieval precision @ k | (k MCP results actually cited in output) / k | k=5; precision ≥ 0.6 |
| Handoff success rate | (handoffs producing usable artifact) / (total handoffs) | ≥ 0.85 |
| Flow step efficiency | (stages PASS) / (stages launched in last 7 days) | ≥ 0.80 |
| Improvement velocity | (gate verdicts upgraded month-over-month) | ≥ 1 per release cycle |

---

## §5 SOTA benchmark protocol

v11 is not allowed to claim "SOTA" without:

1. A public benchmark dataset on which the comparator is evaluated.
2. The comparator's published metric on that dataset cited.
3. v11 evaluated on the same dataset with the same metric.
4. Statistical comparison (paired bootstrap or McNemar) reported.

### §5.1 Required comparators (parallel research surface)

| Comparator | Paper / DOI | Benchmark | Metric | Status |
|---|---|---|---|---|
| TrajectoryNet (Krishnaswamy) | Tong et al. 2020 ICML, arXiv 2002.04461 | EB scRNA / GSE271107 LOPO | 1-Wasserstein at t1 / EMD | re-implementation needed; published EB EMD = 0.784 |
| MIOFlow | Huguet et al. NeurIPS 2022, arXiv 2206.14928 | EB t=2 | W₁ = 25.744; MMD(G) = 0.061 | re-implementation needed (architecturally closest cousin) |
| CellOT (Bunne 2023) | PMID 37770709, [DOI](https://doi.org/10.1038/s41592-023-01969-x) | sci-Plex chromatin / 4i protein | Cell-pair W₂ | methodological cousin; not direct head-to-head |
| TIGON (Sha 2023) | PMID 38274364, [DOI](https://doi.org/10.1038/s42256-023-00763-w) | Time-series scRNA + growth | WFR distance, GRN recovery | run on GSE271107 if multi-snapshot available |
| WOT (Schiebinger 2019) | PMID 30712874, [DOI](https://doi.org/10.1016/j.cell.2019.01.006) | Pop-level lineage / Weinreb hematopoiesis | Pearson r=0.150, AUROC=0.599 | population-level baseline on GSE271107 |
| CellRank 2 (Weiler 2024) | PMID 38871986, [DOI](https://doi.org/10.1038/s41592-024-02303-9) | hematopoiesis / endoderm | Terminal-state recovery | use as fate-mapping post-processor of v11's drift |
| PRESCIENT (Yeo 2021) | PMID 34050150, DOI 10.1038/s41467-021-23518-w | Weinreb 2020 hematopoiesis | Pearson r=0.347, AUROC=0.692 | head-to-head on Weinreb if v11 ingests dataset |
| TRACERx Lung 2023 (Frankell) | PMID 37046096, [DOI](https://doi.org/10.1038/s41586-023-05783-5) | NSCLC subclonal selection | DFS hazard | citation framing only; not metric peer (lung, not MM) |
| **mmSYGNAL (Murie/Baliga 2025)** | **PMID 40169765, [DOI](https://doi.org/10.1038/s41416-025-02987-6)** | **MMRF CoMMpass + 4 cohorts (1,367 MM patients)** | **PFS C-index; 67-drug response concordance** | **PRIMARY MM ML head-to-head** — verified 2026-05-03 |
| RUSBoost MM dynamic biomarker (2026) | PMID 41814396 | 662 newly-diagnosed MM patients | F1=0.75 @ Cycle 4 vs R-ISS F1=0.32 | clinical-relevance bar (different metric) |
| MOFA+ Ridge baseline | (already on disk) | Same MMRF panel | Pooled MSE 2.8155 (vs RM 2.836) | DONE — RM ties / loses |
| ~~iMLGAM~~ (rejected) | PMID 40236779 (pan-cancer ICB-response, NOT MM) | — | — | **REJECTED** as MM comparator after 2026-05-03 verification |

This table reflects post-verification corrections (2026-05-03 PubMed sweep). See `paper/v8_artifacts/sota_comparison.md` §0.1–§0.2 for the override block. The previous iMLGAM-as-MM-comparator framing was incorrect and is replaced by mmSYGNAL.

### §5.2 Datasets

| Dataset | Use | Quality control |
|---|---|---|
| MMRF CoMMpass IA22 | Primary MM (already integrated) | Already validated by validate_data_files |
| Beat AML 1.0 (Tyner 2018) | Cross-disease (already integrated) | Already validated |
| TCGA-LAML | Second AML cohort (v11 add) | New: schema audit + duplicate-check before integration |
| DepMap 25Q1 Chronos | CRISPR oracle (already integrated) | Already validated |
| PRISM secondary screen | Drug oracle (already integrated) | Already validated |
| CCLE proteomics | Drug-cell-line oracle (already integrated) | Already validated |
| GTEx tissue baseline | Negative control for MM-specificity claim | New: pull and integrate |

---

## §6 Sequencing & complexity

### §6.1 Dependency-ordered stages

```
v11_encoder        → component #1   (HF scFoundation + LoRA)
v11_landscape_dsm  → components #2,#3 (Helmholtz projection + ODE forward)
v11_propagation    → component #4 (GAT over STRING)
v11_hbayes_ppc     → component #5 (PPC sufficiency check)
v11_conformal_gat  → component #6 (Mondrian + GAT base learner)
v11_xdisease_tcga  → component #9 (TCGA-LAML)
v11_evidence       → MCP grounding sweep across paper claims
v11_chair_verdict  → architecture-chair final integration
```

Each emits a stage log `logs/v11_e2e/<stage>.log` and a row in `RUNS.md`.

### §6.2 Time complexity (per stage, per run)

| Stage | Time complexity | Wall-clock estimate |
|---|---|---|
| v11_encoder | O(N·d_emb·L_layers) for LoRA forward; O(N·r·d) for adapter where r≪d | ~30 min on 1×A100 |
| v11_landscape_dsm | O(N·d_z²) for DSM + O(\|E\|·d_z) for PPI Tikhonov; ODE forward O(steps·d_z) | ~5 min CPU |
| v11_propagation | GAT: O(\|E\|·d_h·n_heads) per layer | ~10 min CPU |
| v11_hbayes_ppc | HMC: O(N·n_chains·n_warmup·grad_eval) | ~30 min CPU |
| v11_conformal_gat | jackknife+: O(N·model_train_cost) (here N=787·model) | ~5 min |
| v11_xdisease_tcga | O(N_AML·d) for projection | ~5 min |
| v11_evidence | O(n_claims·n_MCP) calls | ~15 min API |
| v11_chair_verdict | O(n_subagents·trace_size) | ~10 min |

### §6.3 Space complexity

| Component | Space | Storage |
|---|---|---|
| scFoundation base (frozen) | ~7 GB FP16 | HF cache |
| LoRA adapter | ~50 MB | checkpoint |
| U_θ MLP | <5 MB | checkpoint |
| GAT layer | ~100 MB (STRING 12651 × 64 × heads) | checkpoint |
| Tracer logs (per release) | ~50 MB JSONL | logs/ (gitignored) |

---

## §7 Zero-trust gates (MUST pass before merging)

1. `fabrication-sentinel` clean on the worktree diff.
2. `release-bouncer` confirms every new numeric claim has a `RUNS.md` row.
3. `data-integration-validator` re-runs F2 (994-baseline) to confirm leak-fix still holds.
4. `causal-inference-auditor` confirms no language has migrated from L1 to L2 in any artifact.
5. `architecture-chair` issues a uniqueness brief that names the niche where v11 is differentiated.

If any of (1)–(5) fail, the v11 stage rolls back to the v10 verdict.

---

## §8 Out of scope (explicitly)

- Claiming any v11 result breaks the Stone N_p=29 identifiability bound.
- Promoting any L1 mediation finding to L2 without a randomized experiment.
- Replacing the v10 Mondrian conformal Sprint 5 verdict (already exceeds spec).
- Synthetic data generation of any form (`fabrication-sentinel` enforced).
- Adding any "deep architecture" component without justifying it against the project-memory finding "RM ties MOFA+Ridge and loses to predict-mean on pooled MSE".

---

## §9 What ships at v11 release

- Same 7 strict-pass gates from v10, with sharper margins where the stronger prior helps.
- F4 trivially-pass-by-construction (Helmholtz projection); F8-DPS still REFUTED.
- A Mondrian conformal interval distribution at most 60% the v10 width on the same coverage.
- A SOTA comparison table covering ≥6 named comparators from §5.1.
- Full AgentOps trace logs for every stage (observability).
- Three-pillar metric report (evaluation + optimization).
- Architect-chair uniqueness brief.

**Pearl-tier ceiling at v11 release: still L1-with-structural-prior. Same as v10. No claim escalates.**
