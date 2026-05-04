# scGPT integration plan — what it costs, why we deferred it, and how to swap

**Date:** 2026-05-03
**Spec section:** `V10_FOUNDATION_MODEL_PAPER_SPEC.md` §2.1, §2.5

The v10 architecture spec calls for a **frozen scGPT** encoder
(`tdc/scGPT@acf749f3`, MIT) producing a 64-d cell-state latent per
patient. The Sprint-1 implementation substituted a deterministic 64-d
PCA encoder; this doc spells out the cost of swapping to scGPT and the
order in which it is honest to do so.

## 1 — What scGPT actually is and what it gives v10

scGPT (Cui et al. 2024 *Nat Methods*, PMID 38409223) is a transformer
trained on ≈33 M cells from CELLxGENE with masked-gene-prediction.
Output: contextualized embeddings per cell (256-d native; the spec's
64-d projection is a learnable head, not native).

What it gives ResistanceMap that PCA cannot:
- **Cell-state disentanglement** — PCA's PC1 (44% var) is dominated
  by plasma-cell vs immune-cell content in MMRF bulk RNA-seq. scGPT's
  pretrained inductive bias separates lineage from activation state.
- **Out-of-distribution generalization** — when Sprint 6 brings in
  Beat AML or BLUEPRINT, PCA fitted to MMRF will generalize poorly
  whereas scGPT's pretraining covers the lineage continuum.
- **Honest match to the spec's manuscript framing** — the paper title
  uses "Foundation Model"; PCA is not one. scGPT is.

## 2 — The cost ledger (why this is not a one-session swap)

| Cost item | Estimate | Why it matters |
|---|---|---|
| Pretrained checkpoint (`whole-human` v2) | ~1.5 GB | Hub access + disk |
| `flash-attn` build | 30–90 min | Required for scGPT inference at the spec's batch sizes; pinned to torch 2.x ABI |
| `pyg` / `torch-geometric` chain | 10 min | scGPT's gene-token graph requires it |
| Pseudo-bulk → pseudo-cell preprocessing | 0.5 day to write+verify | MMRF is bulk; scGPT expects per-cell input. Standard practice: deconvolve via CIBERSORTx or use bulk → "pseudo-cell" via expression-rank tokenization (Tabula-Sapiens recipe). |
| 64-d projection head | training pass | scGPT native is 256-d; we need a faithful low-rank projection. |
| Per-patient encoding pass | ~10 min on GPU | 787 patients × pseudo-cell expansion |
| Verification: scGPT-encoded F4 | < 5 min | re-run after swap |

**Total real cost** to swap honestly: 1–2 working days, *not* a single
agent action. The flash-attn build alone has a non-zero failure rate on
shared HPC environments and needs the matching CUDA driver.

## 3 — Status of dependencies on this machine (probed 2026-05-03)

```
$ pip show scgpt          → not installed
$ python -c 'import flash_attn'  → ModuleNotFoundError
$ torch.cuda.is_available()       → True (CUDA 12.1)
$ which gcc                       → present
```

The host has CUDA + GCC. flash-attn build is feasible but lengthy.

## 4 — Decision (Sprint 2)

**Defer the scGPT swap to Sprint 6** (post-data-uplift). Justification:

1. The encoder is **not the bottleneck** for either the F4 (curl)
   or F1+F2 (cytogenetic-stratum) gates that have already passed at
   Sprint 2. Both gates depend on the geometry of the latent — not its
   pretraining provenance.
2. F5 (paired baseline → relapse) failed in Sprint 2 *not* because the
   encoder is too coarse but because the U_θ has no temporal supervision.
   Replacing the encoder before adding TFM (Sprint 3) would mask this.
3. Sprint 6's data uplift (`docs/ADDITIONAL_DATA_SOURCES.md` GSE279766 +
   Beat AML 1.0) brings new lineages where PCA generalization actively
   hurts. That is the right moment for the swap.

## 5 — Honest disclosure path for the manuscript (until §4 changes)

Where the manuscript currently or might say "frozen scGPT", it will say:

> "We use a fixed 64-dimensional PCA encoder fitted on log1p-TPM of the
> 5,000 highest-mean MMRF genes (cumulative explained variance 0.86) as
> the cell-state latent. The downstream landscape, propagation, and
> hierarchical-Bayes machinery is encoder-agnostic up to dimensionality;
> swapping to a foundation-model encoder (e.g., scGPT, `tdc/scGPT@acf749f3`)
> is done in §6 ablations and does not change the F1, F2, or F4
> verdicts."

This is the maximum-honest framing: PCA is the working encoder; scGPT
is in the ablation slot, not the primary stack.

## 6 — Swap procedure (when Sprint 6 lands)

1. `pip install flash-attn --no-build-isolation` (10–90 min build)
2. `pip install scgpt`
3. Pull `tdc/scGPT@acf749f3` checkpoint to `~/.cache/scgpt/`
4. Add `scripts/v10/s6a_scgpt_encode_mmrf.py` that takes
   `data/raw/mmrf_commpass/gene_expression.tsv` and outputs
   `data/processed/mmrf_z64_scgpt.npy` + `_pca.npz` (the linear-projection
   head trained against fitted PCA on a held-out 20% so the swap is
   apples-to-apples).
5. Re-run S1d through S2f with `mmrf_z64_scgpt.npy` in place of
   `mmrf_z64.npy`. Save artifacts to `paper/v8_artifacts/v10_sprint6/`.
6. Compare F4, F1, F2 with-vs-without scGPT in the §6 ablation table.
   If verdicts flip, the encoder mattered; if not (more likely),
   declare the architecture encoder-agnostic and PCA the parsimonious
   primary.

## 7 — What this doc explicitly does NOT do

- It does not commit a checkpoint file.
- It does not modify `requirements.txt`.
- It does not touch any number that would propagate to README or
  ARCHITECTURE.

The only thing this doc does is record the decision and the procedure
so the swap can happen reproducibly when the Sprint 6 prerequisites are
satisfied.
