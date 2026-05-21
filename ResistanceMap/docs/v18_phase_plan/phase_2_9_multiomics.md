# v18 Phase Plan — Phase 2 (real multi-omic modality activation) + Phase 9 (foundation pretraining)

Branch: `v18-clean-canonical-mortfm`. Author: multi-omics integration audit, 2026-05-20.

Scope: bring ATAC / methylation / histone-PTM / phosphoproteomics from "schema-present, loader-stub, encoder-defined, never-trained" to "loaded → masked → encoded → reconstructed → fused". HARD CONSTRAINT: no synthetic data; missing modalities are **masked, never zero-filled**.

---

## PHASE 2 — Real multi-omic modality activation

### 2.1 Loader audit (what exists vs. what's wired)

`ResistanceMap/resistancemap/data/` already contains:

| Loader file | Present? | LOC | Notes |
| --- | --- | --- | --- |
| `scrna_loader.py` | yes | 207 | exercised by `mortfm_dataset.py` |
| `scatac_loader.py` | yes | 73 | stub — not consumed by `MORTBatch` |
| `methylation_loader.py` | yes | 78 | stub — no `ModalityTensor` emission |
| `histone_ptm_loader.py` | yes | 87 | stub |
| `phosphoproteomics_loader.py` | yes | 59 | stub |
| `proteomics_loader.py` | yes | 72 | wired |
| `cite_seq_loader.py` | yes | — | surface-protein loader |
| `mortfm_dataset.py` | yes | — | only consumes `rna`, `proteomics`, `clinical`, `drug` |
| **`feature_registry.py`** | **MISSING** | — | proposed below |
| **`epigenome_alignment.py`** | **MISSING** | — | required for ATAC↔gene mapping |

`configs/mortfm.yaml` lines 19–26 default `use_atac=false`, `use_methylation=false`, `use_histone_ptm=false`, `use_phosphoproteomics=false`. The encoder dims (lines 34–39) ARE declared, but the loader→encoder pipeline never executes for those modalities because (a) `mortfm_dataset.py` does not call the four stub loaders and (b) the trainer's Stage A recon loop (`trainer.py:236-246`) only runs RNA recon as a "representative".

### 2.2 Three config profiles to add under `ResistanceMap/configs/`

**`mortfm_minimal.yaml`** — smoke-test floor; matches today's default.
```yaml
# diff vs. mortfm.yaml
extends: mortfm.yaml
use_rna: true
use_proteomics: true
use_clinical: true
use_drug: true
# all others: false  (explicit; no inheritance ambiguity)
loaders:
  rna: resistancemap.data.scrna_loader:load_ccle_rna
  proteomics: resistancemap.data.proteomics_loader:load_ccle_protein
  clinical: resistancemap.data.clinical_outcome_loader:load_mmrf_clinical
  drug: resistancemap.data.drug_target_loader:load_gdsc_drug
```

**`mortfm_full_multiomic.yaml`** — all six omic modalities on.
```yaml
extends: mortfm.yaml
use_atac: true
use_methylation: true
use_histone_ptm: true
use_phosphoproteomics: true
# rna/proteomics/clinical/drug already true via base
loaders:
  atac: resistancemap.data.scatac_loader:load_encode_atac
  methylation: resistancemap.data.methylation_loader:load_tcga_450k
  histone_ptm: resistancemap.data.histone_ptm_loader:load_cptac_histone
  phosphoproteomics: resistancemap.data.phosphoproteomics_loader:load_cptac_phospho
modality_dropout_p: 0.30   # higher because more modalities available
feature_registry:
  enable: true
  gene_universe: hgnc_2024_03
  protein_universe: uniprot_human_reviewed_2024_03
  peak_universe: encode_cre_v3
  cpg_universe: illumina_450k
```

**`mortfm_patient_longitudinal.yaml`** — patient-paired, per-visit.
```yaml
extends: mortfm_full_multiomic.yaml
training_mode: longitudinal
pair_builder: resistancemap.data.trajectory_pair_builder:build_mmrf_pairs
min_paired_visits: 2
include_treatment_exposure: true
loaders:
  rna: resistancemap.data.mmrf_loader:load_mmrf_rna_per_visit
  proteomics: resistancemap.data.mmrf_loader:load_mmrf_protein_per_visit
  clinical: resistancemap.data.clinical_outcome_loader:load_mmrf_clinical_longitudinal
trajectory_epochs: 50
finetune_epochs: 25
```

### 2.3 New file — `resistancemap/data/feature_registry.py`

Single source of truth for cross-cohort feature identity. Without it, "ATAC peak chr1:1234-5678" in ENCODE has no relation to "promoter peak of MYC" in CPTAC.

```python
# resistancemap/data/feature_registry.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass(frozen=True)
class FeatureUniverse:
    name: str                    # "hgnc_2024_03", "uniprot_human_reviewed_2024_03", ...
    ids: Tuple[str, ...]          # canonical IDs, ordered
    index: Dict[str, int]         # id -> column index

class FeatureRegistry:
    """Cross-cohort feature ID harmonizer.

    Owns one FeatureUniverse per modality. The registry is loaded once at
    pipeline init and is stamped into every ModalityTensor via
    `feature_names`. It REFUSES to silently coerce unknown IDs: an
    unmapped feature must either (a) be dropped with a logged warning, or
    (b) raise StrictMappingError under strict mode.
    """

    def __init__(self, universes: Dict[str, FeatureUniverse], *, strict: bool = False):
        self.universes = universes
        self.strict = strict

    # ----- registration ------------------------------------------------
    def register_universe(self, modality: str, universe: FeatureUniverse) -> None: ...

    # ----- harmonization API ------------------------------------------
    def project_to_universe(
        self,
        modality: str,
        source_ids: Sequence[str],
        values: "torch.Tensor",        # (N, len(source_ids))
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return (values_in_universe_order, presence_mask).
        Columns absent in `source_ids` get mask=False, value=0 (the
        value is irrelevant because the encoder must gate on the mask)."""

    def align_pair(
        self, modality: str, ids_a: Sequence[str], ids_b: Sequence[str],
    ) -> Tuple[List[int], List[int]]:
        """Indices into ids_a / ids_b that share canonical IDs."""

    def cross_modality_link(
        self, src_modality: str, src_ids: Sequence[str],
        dst_modality: str,
    ) -> Dict[str, List[str]]:
        """Edges like ATAC-peak -> overlapping HGNC genes (via GENCODE),
        CpG -> nearest TSS gene, UniProt -> HGNC, phosphosite UniProt+pos
        -> parent UniProt."""
```

### 2.4 Loader standardization — emit `ModalityTensor`

`schemas.py` already defines `ModalityTensor` (lines 92–116 of `mortfm/schemas.py`). The four stub loaders must emit it instead of bare tensors. Canonical signature:

```python
# All four loaders adopt this signature:
def load_<source>(
    cohort_manifest: Path,
    registry: FeatureRegistry,
    *,
    sample_ids: Sequence[str],
) -> ModalityTensor:
    """Returns ModalityTensor(
        name=<modality_name>,
        values=<(N, D_universe) float tensor, in registry order>,
        feature_names=registry.universes[<modality>].ids,
        mask=<(N, D_universe) bool tensor; per-cell, per-feature observed>,
        batch_id=<(N,) long tensor>,
    )
    Rows where the entire modality is absent get mask=all-False and
    values=0. The trainer MUST gate the encoder by the row-level mask.
    """
```

Per-loader notes:

| Loader | Source | Universe | Mask semantics |
| --- | --- | --- | --- |
| `scatac_loader.py` | ENCODE / 10x scATAC | ENCODE cCRE v3 (~1M peaks; project to 50k variable peaks) | per-peak: observed iff peak overlaps a sequenced fragment in the cell's cluster pseudobulk |
| `methylation_loader.py` | TCGA Illumina 450k | Illumina 450k CpG IDs | per-CpG: observed iff detection p-value ≤ 0.01 |
| `histone_ptm_loader.py` | CPTAC histone-PTM panel | fixed 6-mark panel (H3K4me3/K9me3/K27me3/K27ac/K36me3/K79me3) | per-mark: observed iff peptide-level CV ≤ 0.30 |
| `phosphoproteomics_loader.py` | CPTAC phospho-TMT | UniProt accession + residue position (e.g. `P38398_S1524`) | per-site: observed iff intensity above noise floor AND parent protein observed in proteomics |

### 2.5 Encoder normalization — every encoder emits `d_token`

Confirmed by inspection of `resistancemap/models/encoders/`:

| Encoder file | exposes `d_token` | output shape | deviation |
| --- | --- | --- | --- |
| `rna_encoder.py` | yes (line 50) | `(N, d_token)` | clean |
| `atac_encoder.py` | yes (line 24) | `(N, d_token)` | clean |
| `methylation_encoder.py` | yes (line 24) | `(N, d_token)` | clean |
| `histone_ptm_encoder.py` | yes (line 32) | `(N, d_token)` | clean |
| `proteomic_encoder.py` | yes (line 26) | `(N, d_token)` | clean |
| `phosphoproteomic_encoder.py` | yes (line 26) | `(N, d_token)` | clean |
| `clinical_encoder.py` | yes (line 24) | `(N, d_token)` | clean |
| `drug_encoder.py` | yes (line 41) | `(N, d_token)` | clean |

No deviations. Every encoder uses the same `d_token` default and constructs a final `nn.Linear(in_dim, d_token)` projection. Phase 2 does NOT need any encoder-side refactor.

### 2.6 Tests to add / extend

**`tests/mortfm/test_schemas.py`** (extend; already exists). Add:
- `test_mortbatch_composes_all_eight_modalities` — build a `MORTBatch` with all eight `ModalityTensor` slots populated and assert `(N, d_token)` after a dry `ModalityTokenizer.forward(batch)` call.
- `test_missing_modality_masked_not_zero_filled` — set `batch.atac = None`, assert `modality_mask["atac"]` is all-False and that the tokenizer returns a zero token tagged with mask=0, NOT a learned non-zero token.

**`tests/mortfm/test_feature_registry.py`** (NEW). Assertions:
- `test_register_and_project_roundtrip` — register HGNC universe of 5000 genes; project a (10, 3000) tensor with overlap of 2500 → output is (10, 5000) with mask sum == 2500.
- `test_unknown_id_strict_raises` — strict mode raises `StrictMappingError` on unmapped CpG IDs.
- `test_atac_peak_to_gene_link` — `cross_modality_link("atac", peaks, "rna")` returns ≥1 gene for every peak overlapping a GENCODE promoter ±2kb.
- `test_phosphosite_parent_protein_consistency` — every phospho `feature_name` of form `<UNIPROT>_<RES><POS>` has a matching parent UniProt in the proteomics universe.

### 2.7 File-change table

| Path | Action | Reason |
| --- | --- | --- |
| `resistancemap/data/feature_registry.py` | NEW | cross-cohort ID harmonizer |
| `resistancemap/data/scatac_loader.py` | rewrite to emit `ModalityTensor` | currently stub |
| `resistancemap/data/methylation_loader.py` | rewrite to emit `ModalityTensor` | currently stub |
| `resistancemap/data/histone_ptm_loader.py` | rewrite to emit `ModalityTensor` | currently stub |
| `resistancemap/data/phosphoproteomics_loader.py` | rewrite to emit `ModalityTensor` | currently stub |
| `resistancemap/data/mortfm_dataset.py` | wire 4 new loaders behind config flags | currently only consumes RNA/prot/clin/drug |
| `configs/mortfm_minimal.yaml` | NEW | smoke profile |
| `configs/mortfm_full_multiomic.yaml` | NEW | all-modality profile |
| `configs/mortfm_patient_longitudinal.yaml` | NEW | MMRF per-visit profile |
| `tests/mortfm/test_schemas.py` | EXTEND | full-modality composition test |
| `tests/mortfm/test_feature_registry.py` | NEW | universe / link tests |

### 2.8 Phase 2 Definition of Done

1. All four config profiles load without raising.
2. `mortfm_full_multiomic.yaml` produces a `MORTBatch` where every `ModalityTensor` has shape consistent with `feature_registry.universes[modality].ids`.
3. `tests/mortfm/test_schemas.py::test_missing_modality_masked_not_zero_filled` PASSES.
4. `tests/mortfm/test_feature_registry.py` (4 cases) PASSES.
5. `python -m resistancemap.mortfm.canonical_trainer --config configs/mortfm_full_multiomic.yaml --stages A` runs one epoch on real MMRF + CPTAC + ENCODE data without `NaN`s and without raising `MissingSupervisionError` (Stage A is unsupervised).
6. No call site uses `torch.zeros_like(...)` to fake an absent modality — `grep -rn "zeros_like.*modality\|zeros_like.*atac\|zeros_like.*methylation" resistancemap/` returns empty.

---

## PHASE 9 — Foundation pretraining: real, not just token fusion

### 9.1 Audit of Stage A / Stage B today

`resistancemap/mortfm/trainer.py`:
- Line 132: `if stage == "A":  # modality reconstruction only` — selects the recon weights.
- Lines 234–246 (Stage A): only RNA recon runs. Comment on line 236 explicitly admits: *"Use RNA recon (NB) as a representative — this is a placeholder for the per-modality recon dispatch that a more thorough trainer would assemble."*
- Lines 247–256 (Stage B): only proteomics masked-token consistency runs, and the "target" is the same encoder re-applied to the same input — i.e. the loss measures fusion-token / encoder-token agreement, NOT cross-modal prediction. This is not a true masked-modality objective.
- Lines 257–263 (Stage B contrastive): aligns the first two modality embeddings only (`embs[0]` vs `embs[1]`), ignoring the other six.

There is NO `resistancemap/mortfm/foundation/` directory. The per-modality decoders are scattered inside the encoder files (e.g. `rna_encoder.py:76-78`, `proteomic_encoder.py:59`, `methylation_encoder.py:50-51`, `phosphoproteomic_encoder.py:42`, `histone_ptm_encoder.py:48`, `atac_encoder.py:45`) — usable, but the trainer never calls them except for RNA.

### 9.2 New module — `resistancemap/mortfm/foundation/decoders.py`

Centralise per-modality decoders behind a uniform interface so the trainer can dispatch them in a loop.

```python
# resistancemap/mortfm/foundation/decoders.py
from typing import Optional, Tuple
import torch, torch.nn as nn

class RNANBDecoder(nn.Module):
    """z -> (mu, theta) for negative binomial counts."""
    def __init__(self, d_token: int, n_genes: int): ...
    def forward(self, z: torch.Tensor, library_size: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]: ...

class ATACBernoulliDecoder(nn.Module):
    """z -> peak-open logits."""
    def __init__(self, d_token: int, n_peaks: int): ...
    def forward(self, z: torch.Tensor) -> torch.Tensor: ...   # logits (N, P)

class ProteomicsGaussianDecoder(nn.Module):
    """z -> (mu, log_sigma) for log-intensity."""
    def __init__(self, d_token: int, n_proteins: int): ...
    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]: ...

class MethylationBetaDecoder(nn.Module):
    """z -> (alpha, beta) Beta-distribution params over [0,1] beta-values.
    Falls back to Gaussian on logit(beta) if alpha+beta numerics misbehave."""
    def __init__(self, d_token: int, n_cpg: int): ...
    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]: ...

class HistonePTMGaussianDecoder(nn.Module):
    """z -> (mu, log_sigma) for the 6-mark vector."""
    def __init__(self, d_token: int, n_marks: int = 6): ...
    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]: ...

class PhosphoproteomicsGaussianDecoder(nn.Module):
    """z + site-mask -> (mu, log_sigma) for log phospho-intensity.
    Decoder MUST consume the site-mask so unobserved sites contribute 0
    to the loss (and 0 gradient back through the decoder weights)."""
    def __init__(self, d_token: int, n_sites: int): ...
    def forward(self, z: torch.Tensor, site_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

### 9.3 New module — `resistancemap/mortfm/foundation/masked_modality.py`

True modality dropout + cross-modal reconstruction. Today's trainer feeds the masked modality back to itself; this module predicts the held-out modality from the OTHERS.

```python
# resistancemap/mortfm/foundation/masked_modality.py
from typing import Dict, List
import torch, torch.nn as nn
from resistancemap.mortfm.schemas import MORTBatch, ModalityName

class MaskedModalityTask(nn.Module):
    """Cross-modal masked-modality pretraining.

    Procedure per batch:
      1. Sample a subset M of currently-observed modalities to mask
         (Bernoulli(p) per modality, with at least one kept).
      2. Run fusion on the surviving modalities.
      3. For each masked modality m in M, decode the fusion token at
         m's slot via decoders[m] and compute m's reconstruction loss
         vs. the WITHHELD true values.
    Cannot leak: the masking happens BEFORE the encoder forward pass,
    not after fusion.
    """
    def __init__(
        self,
        decoders: Dict[str, nn.Module],   # one of the classes from decoders.py per modality
        p_mask: float = 0.30,
        min_kept: int = 1,
    ): ...

    def forward(
        self,
        batch: MORTBatch,
        fusion: nn.Module,                # the MORT-FM fusion stack
    ) -> Dict[str, torch.Tensor]:
        """Returns {f"masked_{m}_from_others": loss_scalar, ...}.
        Empty dict if no observed modality could be masked (e.g.
        modality_mask sums to 1)."""
```

### 9.4 New dispatch — `resistancemap/training/canonical_mortfm_losses.py`

```python
# resistancemap/training/canonical_mortfm_losses.py
from typing import Dict
import torch
from resistancemap.mortfm.schemas import MORTBatch
from resistancemap.training.mortfm_losses import (
    reconstruction_loss, masked_modality_loss,
)

def reconstruction_loss_dispatch(
    out: Dict[str, "tuple|torch.Tensor"],   # decoder outputs keyed by modality
    batch: MORTBatch,
) -> Dict[str, torch.Tensor]:
    """Returns the eight per-modality + cross-modal loss terms:
      recon_rna                          (NB on raw counts)
      recon_atac                         (Bernoulli on open/closed)
      recon_methylation                  (Beta or Gaussian on logit-beta)
      recon_histone_ptm                  (Gaussian)
      recon_proteomics                   (Gaussian, presence-masked)
      recon_phosphoproteomics            (Gaussian, site-masked)
      masked_rna_from_epi_prot           (cross-modal: predict RNA from ATAC+Meth+Prot)
      masked_protein_from_rna_epi        (cross-modal: predict Prot from RNA+ATAC+Meth)
    Any term whose modality is missing for the entire batch is omitted
    from the returned dict (NOT zero-filled — the trainer sums what's
    present)."""
```

The eight terms map 1:1 to the eight rows of the loss-config block the trainer aggregates.

### 9.5 Patch plan — `resistancemap/mortfm/canonical_trainer.py` Stage A / Stage B

Current trainer (`mortfm/trainer.py:234-263`) is the legacy path; the v18 canonical trainer (per branch description) reads `forward()`'s LENS-dict surface. Patches:

| Stage | Today | Patch |
| --- | --- | --- |
| A | only RNA NB recon (line 244) | call `reconstruction_loss_dispatch(out, batch)` over the six per-modality terms; sum-with-weights only present terms |
| A | proteomics, methylation, histone, ATAC, phospho recon never run | run them via the decoders in `foundation/decoders.py`, each with its own distribution |
| B | masked-token self-consistency only (line 256) | replace with `MaskedModalityTask(decoders, p_mask=config.modality_dropout_p)(batch, fusion)`; sum `masked_*_from_others` losses |
| B | contrastive on first two modality embeddings only (line 261) | iterate over all observed-modality pairs and average pairwise InfoNCE, weighted by `modality_mask` row sums |

Stage A and Stage B must continue to use the `MissingSupervisionError` policy ONLY for supervised stages (E/F/G per `trainer.py:115-119`); Stage A/B are unsupervised foundation pretraining and may legitimately have batches where one modality is fully absent.

### 9.6 New tests

**`tests/mortfm/test_foundation_pretraining.py`** (NEW):
- `test_reconstruction_dispatch_returns_only_present_modalities` — build a `MORTBatch` with `atac=None` and assert `recon_atac` is NOT a key of the returned dict.
- `test_per_modality_decoder_shapes` — all six decoders produce the documented output shapes from a `(N, d_token)` z.
- `test_nb_decoder_gradients_flow_to_fusion` — backward on `recon_rna` produces non-zero `fusion.parameters()[0].grad`.
- `test_stage_a_runs_all_six_modalities` — run one batch of Stage A on a synthetic-shape but real-distribution sample (NO synthetic VALUES — use a held-out real subsample) and assert every enabled modality's recon term is present in `components` dict.

**`tests/mortfm/test_missing_modality.py`** (NEW):
- `test_masked_modality_task_does_not_use_held_out_input` — patch encoder of the masked modality to raise on forward; the task must still complete (proving the held-out modality is not fed into the encoder during fusion).
- `test_mask_zero_means_zero_gradient_to_that_encoder` — set `batch.modality_mask["histone_ptm"]` all-False; assert `histone_ptm_encoder.parameters()[0].grad` is `None` or all-zero after backward.
- `test_modality_dropout_keeps_at_least_one` — over 1000 sampled masks, `min(kept_count) >= 1`.
- `test_cross_modal_recon_loss_decreases_over_10_steps` — sanity: starting from random init on real CPTAC pilot data, `masked_protein_from_rna_epi` loss after 10 Adam steps is < loss at step 0.

### 9.7 File-change table — Phase 9

| Path | Action | Reason |
| --- | --- | --- |
| `resistancemap/mortfm/foundation/__init__.py` | NEW | package marker |
| `resistancemap/mortfm/foundation/decoders.py` | NEW | per-modality decoder classes |
| `resistancemap/mortfm/foundation/masked_modality.py` | NEW | true cross-modal masking task |
| `resistancemap/training/canonical_mortfm_losses.py` | NEW | 8-term dispatch |
| `resistancemap/mortfm/trainer.py` lines 234–263 | REPLACE | route through dispatch, not RNA-only |
| `resistancemap/mortfm/canonical_trainer.py` Stage A/B | PATCH | call dispatch on canonical `forward()` LENS dict |
| `tests/mortfm/test_foundation_pretraining.py` | NEW | 4 cases |
| `tests/mortfm/test_missing_modality.py` | NEW | 4 cases |

### 9.8 Phase 9 Definition of Done

1. `reconstruction_loss_dispatch` returns at most 8 keys; for any batch where modality m is fully absent, `recon_m` is NOT in the dict.
2. Stage A on `mortfm_full_multiomic.yaml` exhibits decreasing loss for ≥5 of the 6 per-modality recon terms over the first epoch on real data.
3. Stage B's `masked_*_from_others` terms are computed using ONLY the unmasked subset on the encoder side (verified by `test_masked_modality_task_does_not_use_held_out_input`).
4. `grep -n "placeholder" resistancemap/mortfm/trainer.py resistancemap/mortfm/canonical_trainer.py` returns no matches in the Stage A/B blocks.
5. The 8 cases in `test_foundation_pretraining.py` + `test_missing_modality.py` PASS.
6. `pretrain_epochs: 50` run on `mortfm_full_multiomic.yaml` reaches a held-out (patient-disjoint) Stage-B cross-modal recon NLL lower than RNA-autoencoder-only baseline — quantitative gate, recorded in `RUNS.md`.

---

## Cross-cutting risk table

| Risk | Modality | Likelihood | Severity | Mitigation |
| --- | --- | --- | --- | --- |
| ATAC peaks defined per-cohort don't align across ENCODE / CPTAC / 10x | ATAC | high | high | `FeatureRegistry` projects all cohorts onto ENCODE cCRE v3; peaks outside cCRE dropped with logged warning |
| ATAC peak ↔ gene mapping (peak-to-TSS ±2kb) is many-to-many | ATAC ↔ RNA | high | medium | `FeatureRegistry.cross_modality_link` returns lists, never single mappings; downstream attribution must aggregate |
| Phosphosite UniProt accession changes between releases (`P38398` → `P38398-2` isoform) | Phospho | medium | high | pin UniProt release in `feature_registry` config (`uniprot_human_reviewed_2024_03`); fail closed on unknown accession in strict mode |
| CpG → gene mapping is ambiguous (intergenic CpGs, promoter overlaps) | Methylation ↔ RNA | high | medium | use Illumina manifest's `UCSC_RefGene_Name`; multi-gene CpGs distributed proportionally to TSS distance |
| Histone-PTM 6-mark panel ≠ ChIP-seq broad/narrow peak space | Histone | medium | medium | treat histone-PTM as bulk per-sample vector only (encoder already does this); do NOT pretend it is per-locus |
| Proteomics ↔ phosphoproteomics dependency: phospho intensities only meaningful when parent protein observed | Prot ↔ Phospho | high | high | site-mask in `PhosphoproteomicsGaussianDecoder` is the AND of `phospho_observed` and `parent_protein_observed` |
| Modality dropout with `p=0.30` can mask the only label-bearing modality in small batches | All | medium | medium | `MaskedModalityTask.min_kept=1` plus reject-and-resample when masking would orphan supervision |
| Beta-distribution decoder for methylation diverges near β=0 or β=1 | Methylation | medium | low | fall back to Gaussian on logit(β) clamped at [-5, 5] |
| Per-modality recon losses have different natural scales (NB NLL ~ 1e2, Gaussian NLL ~ 1e0) | All | high | high | normalize each term by its first-epoch median before adding to total loss; record the normalizer in checkpoint metadata |
| Missing modality leakage via batch statistics (BatchNorm on an absent modality) | All | medium | high | encoders use LayerNorm not BatchNorm (verified); add a regression test in `test_missing_modality.py` |
| `feature_registry.py` becomes a hidden ground truth that nobody audits | All | medium | medium | universe content is content-addressed by SHA in config; `FeatureRegistry.__init__` logs the SHA at load time |
