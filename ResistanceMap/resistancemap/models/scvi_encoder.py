"""scVI latent encoder for the v20 single-cell biology layer (Phase 3).

Trains an scVI model on the merged GEO scRNA-seq cohort
(:mod:`resistancemap.data.geo_scrna_loader`) and exposes:

* the integrated latent ``z`` (initial condition for the WaddingtonPotential SDE),
* normalised expression of the 20 chromatin reader/writer genes (parameterises
  the reused :class:`~resistancemap.models.trajectory.ChromatinODE`),
* normalised expression of the M-protein / B2M / albumin / LDH proxy genes
  (training signal for :mod:`resistancemap.models.pk_observation_decoder`).

``scvi-tools`` is an OPTIONAL dependency. Importing this module never fails;
constructing :class:`SCVIEncoder` raises a clear ``ImportError`` with install
instructions if ``scvi-tools`` is absent. No latent or expression value is ever
fabricated — every method delegates to the trained scVI model.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


# The 20 chromatin reader/writer proteins ChromatinODE consumes, mirrored from
# TrajectoryConfig.reader_writer_proteins so scRNA-seq can drive the same ODE.
CHROMATIN_READER_WRITER_GENES: List[str] = [
    "EZH2", "KDM6A", "KDM6B", "KMT2A", "KMT2D", "DNMT1", "DNMT3A", "DNMT3B",
    "TET1", "TET2", "HDAC1", "HDAC2", "KAT2A", "KAT2B", "EP300", "BRD4",
    "SMARCA4", "ARID1A", "SUZ12", "EED",
]

# Gene proxies for serum biomarkers (used by the PK observation decoder).
BIOMARKER_PROXY_GENES: List[str] = [
    "IGHG1", "IGHG2", "IGHG3", "IGHA1", "IGHA2",   # M-protein (IgG/IgA heavy chains)
    "IGKC",                                          # FLC kappa
    "IGLC1", "IGLC2", "IGLC3",                       # FLC lambda
    "B2M", "ALB", "LDHA", "LDHB", "TNFRSF17",        # B2M, albumin, LDH, sBCMA proxy
]


def _require_scvi():
    try:
        import scvi  # noqa: F401
        return scvi
    except ImportError as exc:  # pragma: no cover - exercised only without scvi
        raise ImportError(
            "SCVIEncoder requires scvi-tools. Install with: "
            "pip install 'scvi-tools>=1.1,<1.5'  (extras group 'scbio')."
        ) from exc


class SCVIEncoder:
    """Thin wrapper around ``scvi.model.SCVI`` for the v20 biology layer."""

    def __init__(self, n_latent: int = 30, gene_likelihood: str = "zinb"):
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        self.model = None
        # Touch the dependency eagerly so failures surface at construction.
        _require_scvi()

    def fit(self, adata, *, batch_key: str = "dataset", layer: Optional[str] = "counts",
            max_epochs: int = 400, early_stopping: bool = True):
        """Train scVI. ``adata`` must carry raw counts (in ``layer`` or ``.X``)."""
        scvi = _require_scvi()
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("SCVIEncoder.fit got an empty AnnData — refusing to train.")
        setup_kwargs = {"batch_key": batch_key} if batch_key in adata.obs else {}
        if layer is not None and layer in getattr(adata, "layers", {}):
            setup_kwargs["layer"] = layer
        scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
        self.model = scvi.model.SCVI(
            adata, n_latent=self.n_latent, gene_likelihood=self.gene_likelihood,
        )
        self.model.train(max_epochs=max_epochs, early_stopping=early_stopping)
        return self

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("SCVIEncoder used before fit().")

    def get_latent(self, adata=None):
        """Return the integrated latent ``z`` (n_cells, n_latent)."""
        self._check_fitted()
        return self.model.get_latent_representation(adata)

    def _present_genes(self, adata, genes: Sequence[str]) -> List[str]:
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            logger.warning(
                "SCVIEncoder: %d/%d requested genes absent from var_names "
                "(masked, not fabricated): %s",
                len(missing), len(genes), missing[:8],
            )
        if not present:
            raise ValueError("None of the requested genes are in the dataset.")
        return present

    def get_chromatin_expression(self, adata, genes: Sequence[str] = CHROMATIN_READER_WRITER_GENES,
                                 *, n_samples: int = 25):
        """Normalised expression of chromatin reader/writer genes (n_cells, k)."""
        self._check_fitted()
        present = self._present_genes(adata, genes)
        return self.model.get_normalized_expression(
            adata, gene_list=present, n_samples=n_samples, return_mean=True,
        )

    def get_biomarker_proxy_expression(self, adata, genes: Sequence[str] = BIOMARKER_PROXY_GENES,
                                       *, n_samples: int = 25):
        """Normalised expression of serum-biomarker proxy genes (n_cells, k)."""
        self._check_fitted()
        present = self._present_genes(adata, genes)
        return self.model.get_normalized_expression(
            adata, gene_list=present, n_samples=n_samples, return_mean=True,
        )
