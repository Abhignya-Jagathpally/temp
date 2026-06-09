"""
resistancemap/mortfm/schemas.py
==============================
Canonical typed primitives for MORT-FM (Multi-Omic Resistance Trajectory
Foundation Model, v15).

This module is intentionally minimal-dependency: it imports ``torch`` and the
standard library only, so that any other MORT-FM module (loaders, encoders,
fusion, dynamics, heads, trainer, evaluation) can import these schemas without
pulling heavy single-cell or graph dependencies.

Design rules
------------
1. Every modality is wrapped in a :class:`ModalityTensor` so missing-modality
   masks travel with the data. A modality that is absent is encoded as
   ``snapshot.<mod_name> is None`` — *never* as a zero-tensor with a hidden
   mask, because silent zero-imputation has caused at least one published
   multi-omics integration to learn an "all-zero" shortcut.
2. Every batch carries a ``modality_mask`` dict so the model knows which
   modalities are real vs. missing for *each row*. Modality-dropout
   pretraining flips entries of this mask, but never invents data.
3. Patient and time indices are first-class. The static v14 pipeline can
   forget patient identity because all rows are independent cell lines; MORT-FM
   cannot, because (a) leakage tests require patient-disjoint splits and
   (b) temporal pairs require patient-grouped sampling.
4. Outcomes carry an explicit ``censored: bool`` flag. Survival heads consume
   this — anything that silently treats censoring as event-observed is a
   correctness bug.

The MORTBatch contract
----------------------
``MORTBatch`` is the *only* object the trainer feeds into the model. Every
training script ultimately produces a ``MORTBatch``; every loss function
consumes one. If a new modality is added to the model, it must be added here
first, because the encoder router uses these field names as keys.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover - torch is a hard dependency
    raise ImportError(
        "resistancemap.mortfm.schemas requires torch; install torch>=2.1"
    ) from exc


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------


class ModalityName(str, enum.Enum):
    """Canonical modality names. Used as keys in MORTBatch.modality_mask."""

    RNA = "rna"
    ATAC = "atac"
    METHYLATION = "methylation"
    HISTONE_PTM = "histone_ptm"
    PROTEOMICS = "proteomics"
    PHOSPHOPROTEOMICS = "phosphoproteomics"
    CLINICAL = "clinical"
    DRUG = "drug"


#: Order in which modalities are emitted by the foundation encoder. Stable
#: order is important for downstream interpretability — a transformer with
#: per-token attention maps that uses dict iteration order will produce
#: nondeterministic attribution figures otherwise.
MODALITY_ORDER: Tuple[ModalityName, ...] = (
    ModalityName.RNA,
    ModalityName.ATAC,
    ModalityName.METHYLATION,
    ModalityName.HISTONE_PTM,
    ModalityName.PROTEOMICS,
    ModalityName.PHOSPHOPROTEOMICS,
    ModalityName.CLINICAL,
    ModalityName.DRUG,
)


# ---------------------------------------------------------------------------
# Modality tensor
# ---------------------------------------------------------------------------


@dataclass
class ModalityTensor:
    """Wraps a single-modality feature tensor with its provenance.

    Attributes
    ----------
    name :
        Canonical modality name. Must match a value in :class:`ModalityName`.
    values :
        Float tensor of shape ``(N, D)`` for static features or ``(N, T, D)``
        for time-resolved features. ``N`` is the batch dimension (cells or
        patients depending on training stage); ``D`` is the feature dimension.
    feature_names :
        Length-``D`` list of feature identifiers (gene symbols, peak IDs,
        protein UniProt IDs, etc.). Required so that pathway-attribution
        outputs can be mapped back to biology.
    mask :
        Optional ``(N, D)`` bool tensor marking *per-feature* observed
        entries. ``mask[i, j] == False`` means feature ``j`` is missing for
        sample ``i``. None means everything observed.
    batch_id :
        Optional ``(N,)`` long tensor naming the technical batch (instrument,
        plate, study). Used by the batch-correction head; ``None`` disables
        it.
    """

    name: str
    values: Tensor
    feature_names: List[str]
    mask: Optional[Tensor] = None
    batch_id: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError(
                f"ModalityTensor.values must be a torch.Tensor, got {type(self.values)}"
            )
        if self.values.ndim not in (2, 3):
            raise ValueError(
                f"ModalityTensor.values must be 2-D (N,D) or 3-D (N,T,D); "
                f"got shape {tuple(self.values.shape)} for modality {self.name!r}"
            )
        feat_dim = self.values.shape[-1]
        if len(self.feature_names) != feat_dim:
            raise ValueError(
                f"ModalityTensor.feature_names has length {len(self.feature_names)} "
                f"but tensor feature dim is {feat_dim} for modality {self.name!r}"
            )
        if self.mask is not None and self.mask.shape != self.values.shape:
            raise ValueError(
                f"ModalityTensor.mask shape {tuple(self.mask.shape)} does not "
                f"match values shape {tuple(self.values.shape)}"
            )

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.values.shape[-1])

    def to(self, device: torch.device) -> "ModalityTensor":
        return ModalityTensor(
            name=self.name,
            values=self.values.to(device),
            feature_names=self.feature_names,
            mask=self.mask.to(device) if self.mask is not None else None,
            batch_id=self.batch_id.to(device) if self.batch_id is not None else None,
        )


# ---------------------------------------------------------------------------
# Drug context
# ---------------------------------------------------------------------------


@dataclass
class DrugContext:
    """Treatment-pressure descriptor consumed by the drug encoder.

    Notes
    -----
    The v14 ResistanceMap uses a fixed 11-drug GDSC list. MORT-FM treats drugs
    as *first-class structured objects* because (a) drug class and target
    proteins are essential for graph-conditioned dynamics, (b) combination
    therapy is the norm in MM standard of care, and (c) dose and timing
    matter for trajectory prediction.

    Attributes
    ----------
    drug_name :
        Canonical drug name, e.g. "Bortezomib".
    drug_class :
        Mechanism class, e.g. "proteasome_inhibitor", "IMiD", "HDAC_inhibitor".
    target_genes :
        HGNC gene symbols of primary targets (used to construct drug-target
        edges in the heterogeneous biological graph).
    target_proteins :
        UniProt IDs of primary protein targets. Disjoint from ``target_genes``
        because some compounds target non-coding products.
    dose :
        Dose in micromolar (cell-line context) or mg/m^2 (patient context).
        Caller must record which convention applies — MORT-FM does not unify.
    start_time, end_time :
        Treatment exposure window relative to the snapshot timepoint
        (``t=0`` is the snapshot). Negative values denote prior exposure.
    combination_id :
        Stable identifier for combination regimens (e.g. "VRd",
        "DaraVRd"). ``None`` means monotherapy.
    molecular_embedding :
        Optional pre-computed chemical embedding (Morgan FPs, MolFormer,
        ChemBERTa, etc.) of shape ``(d_chem,)``. ``None`` is allowed; the
        drug encoder will fall back to identity + class one-hot.
    """

    drug_name: str
    drug_class: Optional[str] = None
    target_genes: List[str] = field(default_factory=list)
    target_proteins: List[str] = field(default_factory=list)
    dose: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    combination_id: Optional[str] = None
    molecular_embedding: Optional[Tensor] = None

    def to(self, device: torch.device) -> "DrugContext":
        return DrugContext(
            drug_name=self.drug_name,
            drug_class=self.drug_class,
            target_genes=list(self.target_genes),
            target_proteins=list(self.target_proteins),
            dose=self.dose,
            start_time=self.start_time,
            end_time=self.end_time,
            combination_id=self.combination_id,
            molecular_embedding=(
                self.molecular_embedding.to(device)
                if self.molecular_embedding is not None
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class PatientCellSnapshot:
    """One patient (or one cell, or one pseudobulked sample) at one timepoint.

    A snapshot has no labels — it is purely the *input* side of MORT-FM. Labels
    live in :class:`ResistanceOutcome`; the two are bound together in
    :class:`TemporalTrainingPair`.
    """

    patient_id: str
    sample_id: str
    disease: str
    timepoint: Optional[float] = None  # months relative to baseline
    treatment_id: Optional[str] = None
    cell_id: Optional[str] = None
    rna: Optional[ModalityTensor] = None
    atac: Optional[ModalityTensor] = None
    methylation: Optional[ModalityTensor] = None
    histone_ptm: Optional[ModalityTensor] = None
    proteomics: Optional[ModalityTensor] = None
    phosphoproteomics: Optional[ModalityTensor] = None
    clinical: Optional[ModalityTensor] = None
    drug: Optional[DrugContext] = None

    def available_modalities(self) -> List[ModalityName]:
        present: List[ModalityName] = []
        for m in MODALITY_ORDER:
            if m == ModalityName.DRUG:
                if self.drug is not None:
                    present.append(m)
                continue
            tensor = getattr(self, m.value, None)
            if tensor is not None:
                present.append(m)
        return present


# ---------------------------------------------------------------------------
# Outcome (labels)
# ---------------------------------------------------------------------------


@dataclass
class ResistanceOutcome:
    """Patient-level outcome / supervision target.

    Attributes
    ----------
    patient_id :
        Must match a :class:`PatientCellSnapshot.patient_id`.
    baseline_time :
        Calendar timepoint of the corresponding input snapshot (months from
        a study-defined zero).
    event_time :
        Time-to-event (months) for the survival head. For PFS-style endpoints
        this is the time from ``baseline_time`` to progression / relapse /
        resistance emergence. ``None`` is only valid if the entire batch is
        being used for unlabelled pre-training.
    censored :
        ``True`` if no event was observed by ``event_time`` (right-censored).
        The Cox / discrete survival losses *require* this flag to be set
        correctly — silently treating censoring as event-observed bias the
        hazard estimate downward.
    resistance_label :
        Optional integer label naming a resistance phenotype class
        (sensitive=0, drug-tolerant persister=1, MRD-like=2, relapsed-resistant=3,
        drug-specific-resistant subtype>=4). The trainer maps these to logits
        via :class:`~resistancemap.models.heads.resistance_state_head.ResistanceStateHead`.
    future_state :
        Optional reference future molecular state tensor for the trajectory
        head's Wasserstein / MMD loss. Shape: ``(D_latent,)`` or ``(T, D_latent)``.
    drug_response :
        Optional continuous response tensor (IC50, AUC, log-fold). Kept for
        backwards compatibility with the v14 static drug-response benchmark.
    pathway_labels :
        Optional list of pathway IDs (Reactome, KEGG, custom MM-resistance
        signature) that are *known* to be active for this patient — used as
        weak supervision in the pathway-attribution loss.
    """

    patient_id: str
    baseline_time: float
    event_time: Optional[float] = None
    censored: bool = True
    resistance_label: Optional[int] = None
    future_state: Optional[Tensor] = None
    drug_response: Optional[Tensor] = None
    pathway_labels: Optional[List[str]] = None

    def has_survival_label(self) -> bool:
        return self.event_time is not None

    def has_future_state(self) -> bool:
        return self.future_state is not None


# ---------------------------------------------------------------------------
# Temporal pair
# ---------------------------------------------------------------------------


@dataclass
class TemporalTrainingPair:
    """A baseline snapshot, an optional follow-up snapshot, and the outcome.

    This is the only object the longitudinal trainer consumes. Building a
    list of :class:`TemporalTrainingPair` from raw data is the responsibility
    of :mod:`resistancemap.data.trajectory_pair_builder`.

    * ``x_t_delta is None`` is valid: it means no measured follow-up snapshot
      exists for this patient, but a survival label may still be present
      (right-censored or otherwise). The trainer routes these rows through
      the survival head only.
    * ``outcome.event_time is None`` AND ``x_t_delta is None`` makes the pair
      *unlabelled* — usable for masked-modality pre-training but not for any
      supervised head.
    """

    x_t: PatientCellSnapshot
    x_t_delta: Optional[PatientCellSnapshot]
    outcome: ResistanceOutcome

    def has_any_supervision(self) -> bool:
        return (
            self.x_t_delta is not None
            or self.outcome.has_survival_label()
            or self.outcome.resistance_label is not None
            or self.outcome.drug_response is not None
        )


# ---------------------------------------------------------------------------
# Canonical batch
# ---------------------------------------------------------------------------


@dataclass
class MORTBatch:
    """The canonical batch object consumed by the MORT-FM model.

    Every tensor field is ``Optional`` because MORT-FM is designed to train on
    *partially observed* multi-omics data. The corresponding key in
    ``modality_mask`` is a bool tensor of shape ``(N,)`` indicating which
    rows have that modality present (False = absent, model must impute or
    skip).

    Construction
    ------------
    The recommended pattern is::

        from resistancemap.data.mortfm_dataset import MORTFMDataset, mort_collate
        loader = torch.utils.data.DataLoader(
            MORTFMDataset(pairs),
            batch_size=64,
            collate_fn=mort_collate,
        )

    The collate function is responsible for stacking per-modality tensors,
    materialising ``modality_mask``, and validating shape consistency.
    """

    # Inputs (one tensor per modality, N x D each).
    rna: Optional[Tensor] = None
    atac: Optional[Tensor] = None
    methylation: Optional[Tensor] = None
    histone_ptm: Optional[Tensor] = None
    proteomics: Optional[Tensor] = None
    phosphoproteomics: Optional[Tensor] = None
    clinical: Optional[Tensor] = None
    drug: Optional[Tensor] = None  # already-encoded drug context (N x d_drug)

    # Per-row presence mask, one bool tensor (N,) per modality.
    modality_mask: Dict[str, Tensor] = field(default_factory=dict)

    # Provenance.
    patient_ids: List[str] = field(default_factory=list)
    cell_ids: Optional[List[str]] = None
    time: Optional[Tensor] = None  # (N,) float, months from baseline

    # Supervision (any subset can be None).
    future_state: Optional[Tensor] = None
    resistance_label: Optional[Tensor] = None
    event_time: Optional[Tensor] = None
    event_observed: Optional[Tensor] = None  # 1 = event, 0 = censored
    pathway_targets: Optional[Tensor] = None
    drug_response: Optional[Tensor] = None

    # --- v19 Phase 1: latent-target follow-up snapshot --------------------
    #
    # When the dataloader has paired follow-up snapshots (x_t and x_t_delta
    # on the same patient), the collate function constructs a SECOND
    # MORTBatch from the follow-up modality tensors and attaches it here.
    # The canonical trainer encodes this through the SAME ``forward_foundation``
    # call as the baseline, producing an encoder-consistent target latent.
    # NEVER populated from raw RNA; if no follow-up exists the field is None
    # and the trainer skips the trajectory term (consistent with the v18.2
    # strict-supervision policy reporting n_supervised=0).
    future_snapshot_batch: Optional["MORTBatch"] = None
    # (B,) float tensor of day-difference between baseline and follow-up.
    # NaN entries flag rows that have no follow-up. Used to normalise the
    # per-row latent L2 (short-horizon predictions should not dominate the
    # gradient).
    delta_t_days: Optional[Tensor] = None

    # Optional graph payload (PyG HeteroData is opaque to schemas).
    graph: Any = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        if self.patient_ids:
            return len(self.patient_ids)
        for name in ("rna", "atac", "methylation", "histone_ptm",
                     "proteomics", "phosphoproteomics", "clinical", "drug"):
            t = getattr(self, name)
            if t is not None:
                return int(t.shape[0])
        return 0

    def available_modalities(self) -> List[str]:
        return [m.value for m in MODALITY_ORDER if getattr(self, m.value) is not None]

    def to(self, device: torch.device) -> "MORTBatch":
        moved: Dict[str, Any] = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Tensor):
                moved[f.name] = val.to(device)
            elif isinstance(val, dict):
                moved[f.name] = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in val.items()
                }
            else:
                moved[f.name] = val
        return MORTBatch(**moved)

    def validate_for_loss(self, loss_name: str) -> None:
        """Raise a clear error if this batch lacks the supervision a loss needs.

        Called by losses at the start of forward to fail-fast instead of
        silently NaN-ing. Examples:
        * ``loss_name="survival"`` requires ``event_time`` and ``event_observed``.
        * ``loss_name="trajectory"`` requires ``future_state`` (or a follow-up
          batch which must be passed separately).
        """
        if loss_name == "survival":
            if self.event_time is None or self.event_observed is None:
                raise ValueError(
                    "Survival loss requires MORTBatch.event_time and "
                    "MORTBatch.event_observed; both are None. Refusing to fabricate."
                )
        elif loss_name == "trajectory":
            if self.future_state is None:
                raise ValueError(
                    "Trajectory loss requires MORTBatch.future_state. "
                    "If you only have a follow-up snapshot, encode it through "
                    "the foundation encoder and pass the latent as future_state."
                )
        elif loss_name == "resistance_state":
            if self.resistance_label is None:
                raise ValueError(
                    "Resistance-state loss requires MORTBatch.resistance_label."
                )
        elif loss_name == "drug_response":
            if self.drug_response is None:
                raise ValueError(
                    "Drug-response loss requires MORTBatch.drug_response."
                )
        elif loss_name == "latent_future_state":
            # v19 Phase 1 — canonical latent trajectory loss. Requires both
            # a re-encodable follow-up snapshot AND per-row Δt; if either
            # is missing the trainer must skip this batch (it should report
            # n_supervised=0 via the strict-supervision audit, NOT silently
            # train on a fabricated target).
            if self.future_snapshot_batch is None:
                raise ValueError(
                    "Latent future-state loss requires "
                    "MORTBatch.future_snapshot_batch (a MORTBatch holding the "
                    "follow-up snapshot's encoded inputs). Got None. "
                    "Refusing to fabricate a target from raw RNA."
                )
            if self.delta_t_days is None:
                raise ValueError(
                    "Latent future-state loss requires MORTBatch.delta_t_days "
                    "(per-row day-difference between baseline and follow-up). "
                    "Got None. Refusing to fabricate."
                )
        # Other losses (recon, masked-omics, contrastive) work without labels.


# ---------------------------------------------------------------------------
# Foundation latent state (encoder output)
# ---------------------------------------------------------------------------


@dataclass
class FoundationState:
    """Output of the foundation fusion encoder.

    Attributes
    ----------
    z0 :
        ``(N, d_latent)`` fused latent state. This is what the dynamics module
        evolves through time.
    modality_embeddings :
        Per-modality post-encoder tokens, ``{modality_name: (N, d_mod)}``. Kept
        so that contrastive cross-modal losses can operate directly on
        per-modality embeddings without re-running the encoder.
    attention_maps :
        ``{layer_name: (N, n_heads, n_tokens, n_tokens)}`` — optional, only
        populated when the fusion module is constructed with
        ``return_attention=True``.
    modality_gates :
        ``(N, n_modalities)`` soft gate of how much each modality contributed
        to ``z0``. Useful for the ablation / robustness reports.
    uncertainty :
        Optional ``(N, d_latent)`` epistemic uncertainty (e.g. ensemble std,
        evidential variance). ``None`` if the encoder is deterministic.
    """

    z0: Tensor
    modality_embeddings: Dict[str, Tensor] = field(default_factory=dict)
    attention_maps: Dict[str, Tensor] = field(default_factory=dict)
    modality_gates: Optional[Tensor] = None
    uncertainty: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# Trajectory prediction (dynamics + heads output)
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryPrediction:
    """Output of the dynamics + heads. The full inference payload for one batch.

    Attributes
    ----------
    z_path :
        ``(N, T, d_latent)`` deterministic mean trajectory.
    z_samples :
        Optional ``(S, N, T, d_latent)`` Monte-Carlo samples from the SDE.
    resistance_state_logits :
        ``(N, n_states)`` from :class:`ResistanceStateHead`.
    hazard :
        Optional ``(N, K)`` discrete-time hazard or scalar Cox risk.
    survival_curve :
        Optional ``(N, K)`` survival probability per time bin.
    trajectory_mean, trajectory_cov :
        Distribution parameters of the future-state head.
    pathway_protein_scores, pathway_edge_scores :
        ``(N, |V|)`` and ``(N, |E|)`` attribution scores over the biological
        graph. Used to extract the predicted resistance route.
    drug_specific_risk :
        ``(N, n_drugs)`` risk under each drug option.
    counterfactual_rankings :
        Optional list (length ``N``) of ranked intervention dicts.
    uncertainty :
        Optional dict bundling epistemic + aleatoric uncertainty per head.
    """

    z_path: Tensor
    z_samples: Optional[Tensor] = None
    resistance_state_logits: Optional[Tensor] = None
    hazard: Optional[Tensor] = None
    survival_curve: Optional[Tensor] = None
    trajectory_mean: Optional[Tensor] = None
    trajectory_cov: Optional[Tensor] = None
    pathway_protein_scores: Optional[Tensor] = None
    pathway_edge_scores: Optional[Tensor] = None
    drug_specific_risk: Optional[Tensor] = None
    counterfactual_rankings: Optional[List[List[Dict[str, Any]]]] = None
    uncertainty: Optional[Dict[str, Tensor]] = None


# ---------------------------------------------------------------------------
# Config dataclass mirroring configs/mortfm.yaml
# ---------------------------------------------------------------------------


@dataclass
class MORTFMConfig:
    """Typed mirror of ``configs/mortfm.yaml``. Used by trainer + agents.

    Only fields explicitly read by python code live here — extra YAML keys
    (e.g. comments, future-work placeholders) are tolerated by the loader but
    not exposed here.
    """

    # Identity
    model_name: str = "MORT-FM"
    version: str = "v15.0-dev"

    # Modality availability flags (turn off to skip an encoder cleanly).
    use_rna: bool = True
    use_atac: bool = True
    use_methylation: bool = False
    use_histone_ptm: bool = False
    use_proteomics: bool = True
    use_phosphoproteomics: bool = False
    use_clinical: bool = True
    use_drug: bool = True

    # Encoder dimensions.
    rna_input_dim: int = 5000
    atac_input_dim: int = 50000
    methylation_input_dim: int = 10000
    histone_ptm_input_dim: int = 6   # H3K27ac, H3K27me3, H3K4me3, H3K9me3, ac, me
    proteomics_input_dim: int = 8000
    phosphoproteomics_input_dim: int = 4000
    clinical_input_dim: int = 32
    drug_n_drugs: int = 32
    drug_n_classes: int = 12
    drug_embed_dim: int = 64

    # Foundation latent.
    d_latent: int = 128
    d_token: int = 64
    fusion_layers: int = 4
    fusion_heads: int = 4
    fusion_dropout: float = 0.1

    # Dynamics.
    dynamics_kind: str = "neural_sde"  # neural_ode | neural_sde | jump_sde
    integration_time: float = 12.0     # months
    n_time_grid: int = 25
    sde_n_mc_samples: int = 16
    use_graph_conditioning: bool = True

    # LENS conditioning widths consumed by GraphEnergyResistanceSDE.
    # d_clinical was previously hardcoded to 5 at three chokepoints
    # (SDE constructor, forward() zero-fill, trainer._clean_clinical),
    # which silently clipped the lab-first 21-feature vocabulary
    # (FULL_CLINICAL_FEATURE_NAMES) back to the baseline 5. It is now
    # config-driven: default 5 keeps existing checkpoints loadable; set
    # to 21 (and supply a visit-level lab table via
    # encode_for_lens(..., include_labs=True)) to consume the full panel.
    # NOTE: changing d_clinical changes the SDE drift's first Linear
    # in_features, so an SDE retrain is required — old checkpoints will
    # not load at a different width.
    d_clinical: int = 5
    d_drug: int = 8

    # Waddington landscape.
    use_potential: bool = True
    potential_hidden: int = 256
    n_attractors: int = 4

    # Survival head.
    survival_kind: str = "discrete"  # cox | discrete | competing_risks
    survival_n_bins: int = 12

    # Loss weights.
    lambda_recon: float = 1.0
    lambda_masked: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_drug: float = 1.0
    lambda_trajectory: float = 1.0
    lambda_survival: float = 1.0
    lambda_state: float = 0.5
    lambda_pathway: float = 0.3
    lambda_perturbation: float = 0.3
    lambda_landscape: float = 0.1
    lambda_counterfactual: float = 0.3
    lambda_calibration: float = 0.1

    # Training.
    batch_size: int = 64
    pretrain_epochs: int = 50
    finetune_epochs: int = 50
    trajectory_epochs: int = 25
    survival_epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-5
    mixed_precision: bool = True
    grad_clip: float = 1.0
    modality_dropout_p: float = 0.25
    seed: int = 42

    # Paths.
    checkpoint_dir: str = "checkpoints/mortfm"
    log_dir: str = "logs/mortfm"

    # Guardrails (Claim Critic Agent reads these).
    min_paired_n_for_longitudinal_claim: int = 100
    min_patient_n_for_survival_claim: int = 200
