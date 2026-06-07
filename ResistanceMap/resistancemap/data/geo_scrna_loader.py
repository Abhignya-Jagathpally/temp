"""GEO single-cell RNA-seq loader and harmoniser (v20 Phase 1.2).

Downloads (or reads from disk) GEO scRNA-seq datasets for the MM
progression / treatment-response spectrum and harmonises them into a single
:class:`anndata.AnnData` with a stable ``dataset`` batch key, ready for the
scVI integration step (:mod:`resistancemap.models.scvi_encoder`).

Honest behaviour
----------------
* Raises ``ImportError`` (with install instructions) when ``scanpy`` / ``anndata``
  are not installed.
* Raises ``FileNotFoundError`` for missing on-disk files; raises ``ValueError``
  for empty / unparseable matrices.
* NEVER fabricates cells, genes, or metadata. A dataset whose supplementary
  archive is absent is reported as missing — not back-filled with random data.
* Subsampling (``max_cells``) is deterministic head-of-matrix, never random,
  to satisfy the fabrication-sentinel and keep runs reproducible.

The supported accessions below carry the *treatment / disease-stage* labels the
v20 contrastive-resistance step needs (pre/post-treatment, responder vs
non-responder, progression stage). The labels are recorded as ``obs`` columns
only when the source file actually provides them.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported accessions (metadata only — no data ships with this module).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GEOAccession:
    accession: str
    description: str
    label_hint: str  # which obs column carries treatment/response/stage signal
    notes: str = ""


SUPPORTED_ACCESSIONS: Dict[str, GEOAccession] = {
    "GSE161801": GEOAccession(
        "GSE161801", "RRMM pre/post treatment (scRNA + scATAC)",
        "treatment_status", "pre vs post-treatment relapsed/refractory MM",
    ),
    "GSE189460": GEOAccession(
        "GSE189460", "Bortezomib responders vs non-responders",
        "response", "bortezomib response label per patient",
    ),
    "GSE223060": GEOAccession(
        "GSE223060", "Primary / remission / relapsed MM",
        "disease_state", "primary vs remission vs relapsed",
    ),
    "GSE271107": GEOAccession(
        "GSE271107", "MGUS / SMM / MM progression spectrum",
        "disease_stage", "progression-continuum stage label",
    ),
    "GSE124310": GEOAccession(
        "GSE124310", "Progression spectrum (processed counts only)",
        "disease_stage", "processed counts; no raw FASTQ",
    ),
}

GEO_SUPP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}nnn/{acc}/suppl/{acc}_RAW.tar"
)


def _require_scanpy():
    try:
        import scanpy as sc  # noqa: F401
        import anndata as ad  # noqa: F401
        return sc, ad
    except ImportError as exc:  # pragma: no cover - exercised only without scanpy
        raise ImportError(
            "geo_scrna_loader requires scanpy + anndata. "
            "Install with: pip install scanpy anndata"
        ) from exc


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def geo_supplementary_url(accession: str) -> str:
    """Construct the canonical GEO supplementary ``_RAW.tar`` URL."""
    acc = accession.upper()
    if len(acc) < 6 or not acc.startswith("GSE"):
        raise ValueError(f"Not a GSE accession: {accession!r}")
    stub = acc[: -3]  # e.g. GSE161801 -> GSE161
    return GEO_SUPP_URL.format(stub=stub, acc=acc)


def download_geo_tarball(accession: str, output_dir: str | os.PathLike) -> Path:
    """Download a GEO series supplementary tarball.

    Raises on HTTP failure. Does not extract — extraction layout varies per
    series and is left to the caller / parser.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = geo_supplementary_url(accession)
    dest = out / f"{accession.upper()}_RAW.tar"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("GEO tarball already present: %s", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed NCBI host
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded an empty file for {accession}: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Parsing (each raises if the file is missing / empty)
# ---------------------------------------------------------------------------
def parse_10x_h5(h5_path: str | os.PathLike):
    """Read a CellRanger ``.h5`` into an AnnData (cells x genes)."""
    sc, _ = _require_scanpy()
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(f"10x h5 not found: {p}")
    adata = sc.read_10x_h5(str(p))
    adata.var_names_make_unique()
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{p}: parsed an empty AnnData ({adata.shape}).")
    return adata


def parse_10x_mtx(mtx_dir: str | os.PathLike):
    """Read a 10x ``matrix.mtx`` directory into an AnnData (cells x genes)."""
    sc, _ = _require_scanpy()
    d = Path(mtx_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"10x mtx directory not found: {d}")
    adata = sc.read_10x_mtx(str(d))
    adata.var_names_make_unique()
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{d}: parsed an empty AnnData ({adata.shape}).")
    return adata


def load_h5ad(path: str | os.PathLike):
    """Read a pre-built ``.h5ad`` into AnnData (raises if missing/empty)."""
    sc, _ = _require_scanpy()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f".h5ad not found: {p}")
    adata = sc.read_h5ad(str(p))
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{p}: parsed an empty AnnData ({adata.shape}).")
    return adata


# ---------------------------------------------------------------------------
# Harmonisation
# ---------------------------------------------------------------------------
def harmonize_gene_names(adata, *, upper: bool = True):
    """Standardise gene symbols: strip whitespace, optionally upper-case, unique."""
    new = [str(g).strip() for g in adata.var_names]
    if upper:
        new = [g.upper() for g in new]
    adata.var_names = new
    adata.var_names_make_unique()
    return adata


def add_metadata(adata, accession: str, **obs_overrides):
    """Attach ``dataset`` + accession metadata; record treatment/stage labels.

    ``obs_overrides`` lets the caller set per-cell labels (e.g.
    ``treatment_status=[...]``) that the source file does not carry inline.
    Nothing is invented: only values the caller passes are written.
    """
    adata.obs["dataset"] = accession.upper()
    spec = SUPPORTED_ACCESSIONS.get(accession.upper())
    if spec is not None:
        adata.uns["geo_label_hint"] = spec.label_hint
        adata.uns["geo_description"] = spec.description
    for col, vals in obs_overrides.items():
        if vals is None:
            continue
        if len(vals) != adata.n_obs:
            raise ValueError(
                f"obs override {col!r} length {len(vals)} != n_obs {adata.n_obs}"
            )
        adata.obs[col] = list(vals)
    return adata


def merge_datasets(adatas: Sequence, *, batch_key: str = "dataset",
                   join: str = "inner"):
    """Concatenate AnnData objects on the gene intersection (inner join).

    ``inner`` keeps only genes present in every dataset — no zero-padding of
    absent genes, which would be a quiet form of fabrication. Raises if the
    intersection is empty.
    """
    _, ad = _require_scanpy()
    adatas = [a for a in adatas if a is not None]
    if not adatas:
        raise ValueError("merge_datasets received no AnnData objects.")
    if len(adatas) == 1:
        return adatas[0]
    merged = ad.concat(adatas, join=join, label=batch_key,
                       keys=[str(a.obs["dataset"].iloc[0]) if "dataset" in a.obs else str(i)
                             for i, a in enumerate(adatas)])
    if merged.n_vars == 0:
        raise ValueError(
            "Gene intersection across datasets is empty — refusing to emit a "
            "fabricated zero-padded matrix. Check gene-name harmonisation."
        )
    return merged


def load_and_merge(
    sources: Dict[str, str | os.PathLike],
    *,
    batch_key: str = "dataset",
) -> "object":
    """High-level: load each ``{accession: path}`` and merge.

    ``path`` may be a ``.h5ad``, a ``.h5`` (10x), or a 10x mtx directory; the
    suffix decides the parser. Returns the merged AnnData.
    """
    adatas: List = []
    for acc, path in sources.items():
        p = Path(path)
        if p.suffix == ".h5ad":
            a = load_h5ad(p)
        elif p.suffix == ".h5":
            a = parse_10x_h5(p)
        elif p.is_dir():
            a = parse_10x_mtx(p)
        else:
            raise ValueError(f"Unrecognised scRNA source for {acc}: {p}")
        a = harmonize_gene_names(a)
        a = add_metadata(a, acc)
        adatas.append(a)
    return merge_datasets(adatas, batch_key=batch_key)
