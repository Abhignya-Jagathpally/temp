# MORT-FM Data Contract

Every object that flows through MORT-FM is one of seven typed primitives
defined in `resistancemap/mortfm/schemas.py`. This document is the
authoritative reference for those types — change them only by editing that
file (which will cascade through encoders, fusion, trainer, and tests).

## Primitives

### `ModalityTensor`
Wraps one modality of one batch with provenance.

| Field           | Type            | Meaning                                                |
|-----------------|-----------------|--------------------------------------------------------|
| `name`          | str             | One of `ModalityName` values                           |
| `values`        | Tensor (N,D) or (N,T,D) | Float feature values                          |
| `feature_names` | List[str]       | length-D gene/peak/protein identifiers                 |
| `mask`          | Tensor or None  | (N,D) bool, per-feature observed                       |
| `batch_id`      | Tensor or None  | (N,) long, technical batch / instrument / plate        |

**Invariants:** `len(feature_names) == values.shape[-1]`; mask, if present,
matches values shape.

### `DrugContext`
Structured drug record.

| Field                | Type              |
|----------------------|-------------------|
| `drug_name`          | str               |
| `drug_class`         | str or None       |
| `target_genes`       | List[str] (HGNC)  |
| `target_proteins`    | List[str] (UniProt)|
| `dose`               | float or None     |
| `start_time`         | float or None (months) |
| `end_time`           | float or None (months) |
| `combination_id`     | str or None       |
| `molecular_embedding`| Tensor or None    |

### `PatientCellSnapshot`
One patient (or one cell, or one pseudobulked sample) at one timepoint.
All modality fields are `Optional[ModalityTensor]` — missing modalities are
encoded as `None`, never as zero-tensors. The drug field is
`Optional[DrugContext]`.

### `ResistanceOutcome`
Patient-level supervision target. Carries:
* `event_time` (months) and `censored` (bool) for survival heads
* `resistance_label` (int) for the state head
* `future_state` (Tensor) for the trajectory head
* `drug_response` (Tensor) for the legacy IC50 head
* `pathway_labels` (List[str]) for weak supervision of the pathway head

Any field may be `None`; the trainer routes rows through only those heads
whose supervision is present.

### `TemporalTrainingPair`
The unit of training data:
```
TemporalTrainingPair(x_t, x_t_delta, outcome)
```

* `x_t_delta is None` is valid — the row contributes to survival / state
  losses but not the trajectory loss.
* `outcome.event_time is None` AND `x_t_delta is None` makes the pair
  unlabelled — usable only for masked-modality pre-training.

### `MORTBatch`
Canonical batch object consumed by the model. Built by
`resistancemap.data.mortfm_dataset.mort_collate` from a list of
`TemporalTrainingPair`. Has one optional tensor per modality plus a
`modality_mask: Dict[str, Tensor[N]]` — a per-row bool flag for each
modality saying whether it was observed.

**Validation:** `MORTBatch.validate_for_loss(loss_name)` raises if the
required supervision is missing. The trainer calls this before each loss to
fail fast.

### `MORTFMConfig`
Typed mirror of `configs/mortfm.yaml`. Used by `MORTFM`, `MORTFMTrainer`,
`MultiOmicFoundationFusion`, and the Claim Critic Agent.

## Construction patterns

### From raw .h5ad
```python
from resistancemap.data.scrna_loader import build_snapshots_from_h5ad
from resistancemap.data.clinical_outcome_loader import load_mortfm_outcomes
from resistancemap.data.trajectory_pair_builder import build_temporal_pairs

snapshots = build_snapshots_from_h5ad(
    "data/raw/gse271107.h5ad",
    patient_obs_col="patient_id",
    timepoint_obs_col="timepoint_months",
)
outcomes = load_mortfm_outcomes("data/raw/mmrf_commpass", endpoint="pfs")
pairs = build_temporal_pairs(
    snapshots, outcomes,
    allowed_time_gaps=[3.0, 6.0, 12.0], gap_tolerance=1.0,
)
```

### From scratch (programmatic, e.g. tests)
```python
import torch
from resistancemap.mortfm.schemas import (
    ModalityTensor, PatientCellSnapshot, ResistanceOutcome, DrugContext,
)
rna = ModalityTensor("rna", torch.poisson(torch.full((1, 200), 3.0)),
                      [f"g{i}" for i in range(200)])
snap = PatientCellSnapshot(
    patient_id="P1", sample_id="P1_t0", disease="MM", timepoint=0.0, rna=rna,
    drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01),
)
out = ResistanceOutcome(patient_id="P1", baseline_time=0.0,
                        event_time=12.0, censored=False, resistance_label=0)
```

## What this contract enforces

1. **No silent zero-imputation.** A missing modality is `None`, not a zero
   tensor. The collate function materialises a `modality_mask` so the fusion
   layer knows to ignore those rows.
2. **No silent censoring shortcut.** Survival losses validate
   `event_observed` upfront — they refuse to run on a batch that lacks it.
3. **No patient leakage.** Patient identity travels with every row. Split
   functions enforce disjointness; tests verify it.
4. **Deterministic subsampling.** When loaders truncate, they take the head
   of the list — no `np.random` in any loader (also satisfies the
   fabrication-sentinel hook).
