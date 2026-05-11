"""UniProt FASTA loader for ESM-2 protein-sequence embeddings.

Resolves HGNC gene symbols (the column names in CCLE proteomics) to
canonical reviewed Swiss-Prot human protein sequences. Caches results to
disk so UniProt is hit at most once per protein, ever.

Strategy (in order of preference):
    1. Local cached parquet at
       ``data/external/uniprot/symbol_to_seq.parquet``  → instant.
    2. Local bulk FASTA at
       ``data/external/uniprot/uniprot_human_canonical.fasta.gz`` →
       parse once, cache to parquet.
    3. UniProt REST per-symbol query (only if
       ``allow_network=True``) → fallback for misses; rate-limited.

Symbols with no resolvable sequence return ``None`` (or ``""`` depending
on caller preference); downstream code at ``main.py:753-779`` must mask
their rows in the ESM-2 batch.

This file does NOT generate synthetic sequences — there is no fallback
to ``torch.randn``. Misses are explicit and propagate as ``None``.
"""
from __future__ import annotations

import gzip
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/search"
HUMAN_TAXON = 9606
MAX_SEQ_LEN = 1024   # ESM-2 truncation cap (matches models/protein_network.py)
REQUEST_TIMEOUT_S = 30
REST_RATE_LIMIT_S = 0.1   # UniProt sustains ~10 req/s
HUMAN_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes/Eukaryota/UP000005640/"
    "UP000005640_9606.fasta.gz"
)


def _parse_fasta_blob(text: str) -> dict[str, tuple[str, str]]:
    """Parse FASTA text → ``{gene_symbol: (uniprot_accession, sequence)}``.

    UniProt SwissProt headers:
        ``>sp|P04637|P53_HUMAN cellular tumor antigen p53 GN=TP53 PE=1 SV=4``
    """
    out: dict[str, tuple[str, str]] = {}
    current_gene: Optional[str] = None
    current_acc: Optional[str] = None
    buf: list[str] = []
    gn_re = re.compile(r"\bGN=([A-Za-z0-9._-]+)")

    for line in text.splitlines():
        if line.startswith(">"):
            if current_gene and buf:
                # First-seen-wins on duplicate gene symbols (canonical entry).
                out.setdefault(current_gene, (current_acc or "", "".join(buf)[:MAX_SEQ_LEN]))
            buf = []
            parts = line[1:].split()
            tok = parts[0].split("|")
            current_acc = tok[1] if len(tok) >= 2 else None
            m = gn_re.search(line)
            current_gene = m.group(1) if m else None
        else:
            buf.append(line.strip())

    if current_gene and buf:
        out.setdefault(current_gene, (current_acc or "", "".join(buf)[:MAX_SEQ_LEN]))
    return out


def _load_bulk_fasta(fasta_path: Path) -> dict[str, tuple[str, str]]:
    """Parse the bulk Swiss-Prot FASTA (gz-or-plain) into a dict."""
    if not fasta_path.exists() or fasta_path.stat().st_size == 0:
        return {}
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    with opener(fasta_path, "rt") as fh:
        return _parse_fasta_blob(fh.read())


def _fetch_one_symbol(symbol: str) -> Optional[tuple[str, str]]:
    """REST fallback for one symbol. Returns ``(accession, sequence)`` or ``None``."""
    try:
        import requests  # local import to keep module load light
    except ImportError:
        logger.debug("requests not installed; cannot do REST fallback")
        return None
    try:
        r = requests.get(
            UNIPROT_REST,
            params={
                "query": f"gene_exact:{symbol} AND organism_id:{HUMAN_TAXON} "
                         "AND reviewed:true",
                "format": "fasta",
                "size": 1,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        if r.status_code != 200 or not r.text.startswith(">"):
            return None
        recs = _parse_fasta_blob(r.text)
        if symbol in recs:
            return recs[symbol]
        if recs:
            # UniProt returned a single record but the GN tag did not match
            # the queried symbol exactly (alias mapping); take it anyway.
            return next(iter(recs.values()))
        return None
    except Exception as e:
        logger.debug(f"UniProt REST failed for {symbol}: {e}")
        return None


def load_protein_sequences(
    protein_names: Sequence[str],
    cache_dir: Path = Path("data/external/uniprot"),
    bulk_fasta: Optional[Path] = None,
    allow_network: bool = False,
    write_cache: bool = True,
) -> list[Optional[str]]:
    """Resolve a list of HGNC gene symbols to canonical AA sequences.

    Args:
        protein_names: Symbols to resolve (e.g., the proteomics column
            names, ``ds.protein_names``).
        cache_dir: Where to read/write the parquet symbol→sequence cache.
        bulk_fasta: Path to ``UP000005640_9606.fasta.gz``. If ``None``,
            defaults to ``cache_dir / "uniprot_human_canonical.fasta.gz"``.
        allow_network: If True, missing symbols are queried against the
            UniProt REST API. If False, misses return ``None`` and the
            caller is responsible for masking them.
        write_cache: Whether to persist the merged cache back to disk.

    Returns:
        A list of length ``len(protein_names)``. Each entry is either an
        AA sequence (truncated to ``MAX_SEQ_LEN``) or ``None``. ``None``
        always means "no real sequence available" — never a synthetic
        placeholder.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet = cache_dir / "symbol_to_seq.parquet"
    if bulk_fasta is None:
        bulk_fasta = cache_dir / "uniprot_human_canonical.fasta.gz"

    # Layer 1: parquet cache covers everything we've seen before.
    sym2seq: dict[str, str] = {}
    sym2acc: dict[str, str] = {}
    if parquet.exists():
        df = pd.read_parquet(parquet)
        for row in df.itertuples(index=False):
            if row.sequence:
                sym2seq[row.symbol] = row.sequence
                sym2acc[row.symbol] = row.acc
        logger.info(f"UniProt parquet cache: {len(sym2seq)} symbols")

    # Layer 2: bulk Swiss-Prot FASTA.
    if any(s not in sym2seq for s in protein_names) and bulk_fasta.exists():
        bulk = _load_bulk_fasta(bulk_fasta)
        logger.info(f"Bulk FASTA: {len(bulk)} GN-keyed records from {bulk_fasta}")
        for gene, (acc, seq) in bulk.items():
            sym2seq.setdefault(gene, seq)
            sym2acc.setdefault(gene, acc)
    elif not bulk_fasta.exists():
        logger.warning(
            f"Bulk Swiss-Prot FASTA missing at {bulk_fasta}. "
            "Run scripts/download_uniprot.sh or set allow_network=True."
        )

    # Layer 3: REST fallback (rate-limited; gated by allow_network).
    missing = [s for s in protein_names if s not in sym2seq]
    if missing and allow_network:
        logger.info(f"UniProt REST fallback for {len(missing)} symbols...")
        for i, sym in enumerate(missing, 1):
            res = _fetch_one_symbol(sym)
            if res is not None:
                acc, seq = res
                sym2seq[sym] = seq
                sym2acc[sym] = acc
            time.sleep(REST_RATE_LIMIT_S)
            if i % 500 == 0:
                logger.info(f"  ...{i}/{len(missing)} fetched")
                if write_cache:
                    _write_cache(parquet, sym2seq, sym2acc)
        if write_cache:
            _write_cache(parquet, sym2seq, sym2acc)

    sequences: list[Optional[str]] = []
    missed = 0
    for sym in protein_names:
        seq = sym2seq.get(sym)
        if seq:
            sequences.append(seq)
        else:
            sequences.append(None)
            missed += 1
    if missed:
        logger.warning(
            f"UniProt: {len(protein_names) - missed}/{len(protein_names)} "
            f"resolved; {missed} missing → ESM-2 will skip those rows."
        )

    if write_cache and (sym2seq or sym2acc):
        _write_cache(parquet, sym2seq, sym2acc)

    return sequences


def _write_cache(
    parquet: Path,
    sym2seq: dict[str, str],
    sym2acc: dict[str, str],
) -> None:
    """Persist resolved symbol→sequence mapping to a parquet cache."""
    rows = [
        {
            "symbol": sym,
            "acc": sym2acc.get(sym, ""),
            "sequence": seq,
            "length": len(seq),
        }
        for sym, seq in sym2seq.items()
    ]
    if rows:
        pd.DataFrame(rows).to_parquet(parquet, index=False)
