"""
resistancemap/data/protein_sequence_cache.py
============================================
Disk cache for UniProt protein sequences + ESM-2 / ProtT5 embeddings.

Why this module exists separately from :mod:`uniprot_loader`
-----------------------------------------------------------
The :class:`UniProtLoader` parses FASTA / ID-mapping files. *Computing*
embeddings (ESM-2, ProtT5, ProtBERT) is expensive — running ESM-2 over the
~20,000 canonical human proteins takes hours on a single GPU. This cache
amortises that cost across runs and across model versions.

Honest behaviour
----------------
* :func:`load_sequences` raises :class:`FileNotFoundError` if no UniProt
  FASTA is present.
* :class:`ESMEmbeddingCache.load_or_compute` will *refuse* to compute
  embeddings if the `transformers` import fails — it raises a clear message
  pointing to ``pip install transformers torch``. It does NOT silently
  return zeros.
* The cache key includes the model name + sequence MD5 so a different ESM-2
  variant gets a fresh cache file.
* :func:`uniprot_coverage_report` writes ``logs/mortfm/uniprot_coverage_report.json``
  and the Block-A gate refuses to claim "sequence-aware" until that report
  shows ≥70% of STRING proteins map to UniProt.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------


def _iter_fasta(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        uid, gene, seq_parts = None, None, []
        for line in f:
            if line.startswith(">"):
                if uid is not None:
                    yield uid, gene, "".join(seq_parts)
                header = line[1:].strip()
                parts = header.split("|")
                uid = parts[1] if len(parts) >= 2 else parts[0]
                gene = None
                for tok in header.split():
                    if tok.startswith("GN="):
                        gene = tok[3:]
                        break
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if uid is not None:
            yield uid, gene, "".join(seq_parts)


def load_sequences(uniprot_dir: str) -> pd.DataFrame:
    """Load all UniProt sequences from ``uniprot_dir`` into a DataFrame.

    Looks for ``UP*9606*.fasta`` or ``UP*9606*.fasta.gz`` first; falls back to
    any ``*.fasta(.gz)`` in the directory.

    Returns
    -------
    DataFrame with columns ``[uniprot_id, gene_symbol, sequence, length]``.

    Raises
    ------
    FileNotFoundError
        If no FASTA is present. Message includes the UniProt download URL.
    """
    base = Path(uniprot_dir)
    candidates = list(base.glob("UP*9606*.fasta*")) + list(base.glob("*9606*.fasta*"))
    candidates = sorted(p for p in candidates if p.is_file())
    if not candidates:
        # Loosen pattern.
        candidates = sorted(base.glob("*.fasta*"))
    if not candidates:
        raise FileNotFoundError(
            f"No UniProt FASTA found under {base.resolve()}. "
            f"Download human reference proteome from "
            f"https://www.uniprot.org/proteomes/UP000005640 "
            f"(file UP000005640_9606.fasta.gz) and place under data/raw/uniprot/."
        )
    fasta_path = candidates[0]
    logger.info("Reading UniProt FASTA: %s", fasta_path)
    rows = list(_iter_fasta(fasta_path))
    df = pd.DataFrame(rows, columns=["uniprot_id", "gene_symbol", "sequence"])
    df["length"] = df["sequence"].str.len()
    df["is_canonical"] = ~df["uniprot_id"].astype(str).str.contains("-")
    df["sequence_source"] = fasta_path.name
    logger.info("Loaded %d UniProt sequences (%d canonical)", len(df), int(df["is_canonical"].sum()))
    return df


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


@dataclass
class UniProtCoverageReport:
    n_uniprot_total: int = 0
    n_string_proteins_total: int = 0
    n_string_proteins_mapped: int = 0
    string_coverage_rate: float = 0.0
    n_feature_genes_total: int = 0
    n_feature_genes_mapped: int = 0
    feature_gene_coverage_rate: float = 0.0
    has_embedding_cache: bool = False
    embedding_model: str = "none"
    embedding_cache_path: str = ""


def write_uniprot_coverage_report(
    *,
    uniprot_df: pd.DataFrame,
    string_proteins: Optional[Sequence[str]] = None,
    feature_genes: Optional[Sequence[str]] = None,
    embedding_cache_path: Optional[str] = None,
    embedding_model: str = "none",
    out_json: str = "logs/mortfm/uniprot_coverage_report.json",
) -> UniProtCoverageReport:
    rep = UniProtCoverageReport(
        n_uniprot_total=len(uniprot_df),
    )
    uniprot_set = set(uniprot_df["uniprot_id"].astype(str))
    gene_set = set(uniprot_df["gene_symbol"].astype(str).dropna())
    if string_proteins is not None:
        rep.n_string_proteins_total = len(string_proteins)
        mapped = sum(1 for p in string_proteins if str(p) in uniprot_set)
        rep.n_string_proteins_mapped = mapped
        rep.string_coverage_rate = mapped / max(len(string_proteins), 1)
    if feature_genes is not None:
        rep.n_feature_genes_total = len(feature_genes)
        mapped = sum(1 for g in feature_genes if str(g) in gene_set)
        rep.n_feature_genes_mapped = mapped
        rep.feature_gene_coverage_rate = mapped / max(len(feature_genes), 1)
    rep.has_embedding_cache = bool(embedding_cache_path) and Path(embedding_cache_path).exists()
    rep.embedding_cache_path = embedding_cache_path or ""
    rep.embedding_model = embedding_model

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep.__dict__, f, indent=2)
    logger.info("Wrote UniProt coverage report -> %s", out)
    return rep


# ---------------------------------------------------------------------------
# ESM-2 / ProtT5 embedding cache
# ---------------------------------------------------------------------------


def _sequence_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


class ESMEmbeddingCache:
    """Disk cache for protein language model embeddings.

    Storage format: a torch ``.pt`` file whose payload is
    ``{"model": str, "embeddings": dict[uniprot_id -> Tensor], "sequence_hashes": dict[uniprot_id -> md5]}``.
    Embeddings are cached only for ``len(seq) <= max_len`` to avoid the long
    tail of giant proteins blowing the cache.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", max_len: int = 1024) -> None:
        self.model_name = model_name
        self.max_len = max_len
        self._cache: Dict[str, "torch.Tensor"] = {}
        self._hashes: Dict[str, str] = {}

    def path_for(self, cache_dir: str) -> Path:
        slug = self.model_name.replace("/", "__")
        return Path(cache_dir) / f"esm_embeddings__{slug}.pt"

    def load(self, cache_dir: str) -> bool:
        try:
            import torch
        except ImportError:
            return False
        p = self.path_for(cache_dir)
        if not p.exists():
            return False
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if payload.get("model") != self.model_name:
            logger.warning(
                "Embedding cache at %s was built with model %r, but this cache expects %r — "
                "ignoring stale cache.", p, payload.get("model"), self.model_name,
            )
            return False
        self._cache = payload["embeddings"]
        self._hashes = payload.get("sequence_hashes", {})
        logger.info("Loaded %d embeddings from %s", len(self._cache), p)
        return True

    def save(self, cache_dir: str) -> Path:
        import torch
        p = self.path_for(cache_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.model_name, "embeddings": self._cache, "sequence_hashes": self._hashes},
            p,
        )
        logger.info("Saved %d embeddings -> %s", len(self._cache), p)
        return p

    def compute(self, uniprot_df: pd.DataFrame, *, batch_size: int = 4) -> None:
        """Compute embeddings for sequences not already cached.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed. The message points to
            ``pip install transformers``. The cache is NOT populated with
            zeros on failure — callers must handle the exception.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"ESM-2 embedding computation requires `transformers`. "
                f"Install with: pip install transformers torch. "
                f"(Cache will refuse to silently return zeros.)"
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model %s on %s ...", self.model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device).eval()

        to_compute = []
        for _, row in uniprot_df.iterrows():
            uid = str(row["uniprot_id"])
            seq = str(row["sequence"])
            if not seq or len(seq) > self.max_len:
                continue
            h = _sequence_hash(seq)
            if uid in self._cache and self._hashes.get(uid) == h:
                continue
            to_compute.append((uid, seq, h))
        logger.info("Need to compute %d new embeddings", len(to_compute))

        for i in range(0, len(to_compute), batch_size):
            chunk = to_compute[i : i + batch_size]
            seqs = [s for _, s, _ in chunk]
            with torch.no_grad():
                enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_len)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                # Mean-pool over residues; output: (B, L, D)
                emb = out.last_hidden_state.mean(dim=1).cpu()
            for (uid, seq, h), e in zip(chunk, emb):
                self._cache[uid] = e
                self._hashes[uid] = h

    def get(self, uniprot_id: str):
        return self._cache.get(uniprot_id)

    def __len__(self) -> int:
        return len(self._cache)


def load_or_compute_embeddings(
    *,
    uniprot_df: pd.DataFrame,
    cache_dir: str = "data/processed/proteins",
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    compute_if_missing: bool = False,
) -> ESMEmbeddingCache:
    """High-level entry point.

    If ``compute_if_missing=False`` (default) and no cache exists, returns an
    EMPTY cache. The Block-A gate will then refuse the "sequence-aware"
    claim. Pass ``compute_if_missing=True`` to actually fire ESM-2 (slow).
    """
    cache = ESMEmbeddingCache(model_name=model_name)
    ok = cache.load(cache_dir)
    if not ok and compute_if_missing:
        cache.compute(uniprot_df)
        cache.save(cache_dir)
    return cache
