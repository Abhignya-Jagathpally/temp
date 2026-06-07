"""contrastiveVI isolation of drug-resistance programs (v20 Phase 3).

Uses contrastiveVI (scvi-tools external) to separate the transcriptional
variation that is *specific to drug-resistant / post-treatment* cells (the
"salient" latent) from variation shared with sensitive / pre-treatment cells
(the "background" latent). The salient latent is the resistance-program
embedding handed downstream to the WaddingtonPotential SDE.

``scvi-tools`` is an OPTIONAL dependency — importing this module never fails;
constructing :class:`ResistanceProgramIsolator` raises a clear ``ImportError``
if it is absent. The target / background split is taken verbatim from real
``obs`` labels (e.g. pre/post-treatment, responder/non-responder); nothing is
fabricated.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _require_contrastive_vi():
    try:
        from scvi.external import ContrastiveVI  # type: ignore
        return ContrastiveVI
    except ImportError as exc:  # pragma: no cover - exercised only without scvi
        raise ImportError(
            "ResistanceProgramIsolator requires scvi-tools with the ContrastiveVI "
            "external model. Install with: pip install 'scvi-tools>=1.1,<1.5'."
        ) from exc


class ResistanceProgramIsolator:
    """Isolate resistance-specific transcriptional programs via contrastiveVI."""

    def __init__(self, n_salient: int = 10, n_background: int = 10):
        self.n_salient = n_salient
        self.n_background = n_background
        self.model = None
        _require_contrastive_vi()  # fail fast if scvi-tools absent

    def fit(self, adata, *, background_key: str, target_key: str,
            batch_key: str = "dataset", layer: str = "counts", max_epochs: int = 200):
        """Fit contrastiveVI.

        ``background_key`` / ``target_key`` are boolean ``obs`` columns selecting
        the sensitive/pre-treatment (background) and resistant/post-treatment
        (target) cells respectively.
        """
        ContrastiveVI = _require_contrastive_vi()
        for key in (background_key, target_key):
            if key not in adata.obs:
                raise ValueError(f"obs column {key!r} required for contrastive split.")
        setup_kwargs = {}
        if layer in getattr(adata, "layers", {}):
            setup_kwargs["layer"] = layer
        if batch_key in adata.obs:
            setup_kwargs["batch_key"] = batch_key
        ContrastiveVI.setup_anndata(adata, **setup_kwargs)
        self.model = ContrastiveVI(
            adata, n_latent=self.n_salient, n_background_latent=self.n_background,
        )
        bg_idx = np.where(adata.obs[background_key].to_numpy().astype(bool))[0]
        tg_idx = np.where(adata.obs[target_key].to_numpy().astype(bool))[0]
        if len(bg_idx) == 0 or len(tg_idx) == 0:
            raise ValueError(
                "contrastiveVI needs non-empty background AND target sets; "
                f"got |bg|={len(bg_idx)}, |target|={len(tg_idx)}."
            )
        self.model.train(background_indices=bg_idx, target_indices=tg_idx,
                         max_epochs=max_epochs)
        return self

    def get_resistance_programs(self, adata=None):
        """Return the salient latent (resistance-specific variation)."""
        if self.model is None:
            raise RuntimeError("ResistanceProgramIsolator used before fit().")
        return self.model.get_latent_representation(adata, representation_kind="salient")
