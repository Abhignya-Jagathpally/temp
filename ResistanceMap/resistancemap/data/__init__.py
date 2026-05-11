#"""Data loading and preprocessing for ResistanceMap."""
#from .mmrf_loader import MMRFCoMMpassDataset, collate_variable_length, HGNCMapper
#from .preprocessing import (
#    MultiOmicsPreprocessor, DataSplitter, RNASeqPreprocessor,
#    ProteomicsPreprocessor, EpigenomicsPreprocessor, PPINetworkLoader,
#    DrugResponseProcessor, FeatureSelector, MultiOmicsQC,
#)
"""resistancemap.data — single source of truth for dataset loading and splits.

This package was missing from the public repository as of commit 2026-05-07,
even though `resistancemap/main.py:333,337` and `tests/test_leakage.py:8`
already import from it. The minimal stubs here re-establish the import graph
so the pipeline is runnable from a clean clone; full ETL still lives in
`scripts/build_data_ready.py` (to be added).
"""

from resistancemap.data.loaders import (  # noqa: F401
    load_ccle_proteomics,
    load_ccle_epigenomics,
    load_string_ppi,
    load_drug_sensitivity,
)
from resistancemap.data.preprocessors import (  # noqa: F401
    harmonize_omics,
    build_train_val_test_splits,
)
from resistancemap.data.splits import (  # noqa: F401
    make_split,
    assert_no_overlap,
)

__all__ = [
    "load_ccle_proteomics",
    "load_ccle_epigenomics",
    "load_string_ppi",
    "load_drug_sensitivity",
    "harmonize_omics",
    "build_train_val_test_splits",
    "make_split",
    "assert_no_overlap",
]
