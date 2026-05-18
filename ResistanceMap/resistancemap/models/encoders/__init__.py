"""Modality-specific encoders for MORT-FM.

Each encoder maps one raw modality (RNA counts, ATAC peaks, methylation beta
values, histone PTMs, proteomics, phosphoproteomics, clinical covariates,
drug context) into a fixed-width token tensor consumed by
:class:`~resistancemap.models.foundation_fusion.MultiOmicFoundationFusion`.

Design rules
------------
* Every encoder accepts a presence mask (``(N,)`` bool) and is responsible for
  zeroing its own output for absent rows. The fusion layer then re-weights
  via attention so zeroed rows do not poison the joint state.
* Every encoder exposes a ``reconstruct(z)`` method (a small decoder head)
  used by the modality reconstruction / masked-modality losses. ``reconstruct``
  is allowed to return None if the modality is not reconstructable (e.g.
  drug context is a categorical lookup, not a continuous reconstruction).
"""

from resistancemap.models.encoders.rna_encoder import RNAEncoder
from resistancemap.models.encoders.atac_encoder import ATACEncoder
from resistancemap.models.encoders.methylation_encoder import MethylationEncoder
from resistancemap.models.encoders.histone_ptm_encoder import HistonePTMEncoder
from resistancemap.models.encoders.proteomic_encoder import ProteomicEncoder
from resistancemap.models.encoders.phosphoproteomic_encoder import PhosphoproteomicEncoder
from resistancemap.models.encoders.clinical_encoder import ClinicalEncoder
from resistancemap.models.encoders.drug_encoder import DrugEncoder
from resistancemap.models.encoders.modality_tokenizer import ModalityTokenizer

__all__ = [
    "RNAEncoder",
    "ATACEncoder",
    "MethylationEncoder",
    "HistonePTMEncoder",
    "ProteomicEncoder",
    "PhosphoproteomicEncoder",
    "ClinicalEncoder",
    "DrugEncoder",
    "ModalityTokenizer",
]
