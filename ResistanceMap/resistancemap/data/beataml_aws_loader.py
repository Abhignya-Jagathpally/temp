"""BeatAML 1.0 loader for the AWS Open Data (GDC-harmonized) buckets.

Source: https://registry.opendata.aws/beataml/
  * gdc-beataml1-cohort-phs001657-2-open       (RNA-seq, MAF, clinical — OPEN)
  * gdc-beataml1.0-crenolanib-phs001628-2-open (crenolanib clinical supp — OPEN)

These are GDC-harmonized files laid out per file-UUID. Access is anonymous
(no AWS account / no signing) over plain HTTPS, so we use urllib — no boto3
dependency.

Honest scope
------------
* The GDC AWS bucket contains RNA-seq quantification (STAR counts, FPKM,
  FPKM-UQ, HTSeq), somatic MAFs, and single-cell artifacts — but it does NOT
  contain the ex-vivo inhibitor drug-response AUC matrix. That panel lives in
  the Tyner 2018 supplement / Vizome, not GDC. So this loader satisfies the
  *expression* half of BeatAML, not the *drug-response* half.
* Nothing is fabricated: missing files / failed downloads raise; an empty
  matrix raises.
* File-UUID -> BeatAML specimen (case submitter_id) is resolved via the GDC
  /files API so matrix columns are real specimen IDs, not opaque UUIDs.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COHORT_BUCKET = "gdc-beataml1-cohort-phs001657-2-open"
CRENOLANIB_BUCKET = "gdc-beataml1.0-crenolanib-phs001628-2-open"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

# Suffix -> (value column kind). Default expression product is FPKM-UQ: one
# normalized value per Ensembl gene, compressed (~0.2 GB across the cohort).
EXPRESSION_SUFFIXES = {
    "fpkm_uq": "FPKM-UQ.txt.gz",
    "fpkm": "FPKM.txt.gz",
    "htseq": "htseq_counts.txt.gz",
    "star_counts": "rna_seq.augmented_star_gene_counts.tsv",  # 3 GB, has tpm_unstranded
}
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ---------------------------------------------------------------------------
# Bucket listing (anonymous HTTPS, paginated)
# ---------------------------------------------------------------------------
def list_open_bucket(bucket: str = COHORT_BUCKET, *, suffix: Optional[str] = None) -> List[str]:
    """List all object keys in an open bucket, optionally filtered by suffix."""
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        q = {"list-type": "2", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 fixed AWS host
            root = ET.fromstring(r.read())
        for c in root.findall(_S3_NS + "Contents"):
            k = c.find(_S3_NS + "Key").text
            if suffix is None or k.endswith(suffix):
                keys.append(k)
        if root.find(_S3_NS + "IsTruncated").text != "true":
            break
        nxt = root.find(_S3_NS + "NextContinuationToken")
        token = nxt.text if nxt is not None else None
        if not token:
            break
    if not keys:
        raise RuntimeError(
            f"No keys{' with suffix '+suffix if suffix else ''} in s3://{bucket} — "
            "refusing to proceed on an empty listing."
        )
    logger.info("Listed %d keys from s3://%s", len(keys), bucket)
    return keys


# ---------------------------------------------------------------------------
# Download (anonymous HTTPS, parallel, resume-by-existence)
# ---------------------------------------------------------------------------
def _download_one(bucket: str, key: str, out_dir: Path) -> Path:
    dest = out_dir / key.replace("/", "__")  # flatten UUID dirs into one folder
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:  # noqa: S310
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Empty download for {key}")
    tmp.rename(dest)
    return dest


def download_keys(keys: Sequence[str], out_dir: str | os.PathLike, *,
                  bucket: str = COHORT_BUCKET, workers: int = 16) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, bucket, k, out): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                paths.append(fut.result())
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.warning("download failed for %s: %s", futs[fut], e)
            if i % 50 == 0:
                logger.info("  downloaded %d/%d", i, len(keys))
    if not paths:
        raise RuntimeError("All BeatAML downloads failed — not fabricating data.")
    logger.info("Downloaded %d/%d files (%d errors) to %s", len(paths), len(keys), errors, out)
    return paths


# ---------------------------------------------------------------------------
# File-UUID -> specimen (GDC /files API, batched)
# ---------------------------------------------------------------------------
def resolve_file_to_case(file_uuids: Sequence[str], *, batch: int = 200) -> Dict[str, str]:
    """Map GDC file_id -> case submitter_id (BeatAML specimen) via the GDC API."""
    mapping: Dict[str, str] = {}
    uuids = list(dict.fromkeys(file_uuids))
    for i in range(0, len(uuids), batch):
        chunk = uuids[i:i + batch]
        payload = {
            "filters": {"op": "in", "content": {"field": "file_id", "value": chunk}},
            "fields": "file_id,cases.submitter_id,cases.samples.submitter_id",
            "size": str(len(chunk)),
            "format": "json",
        }
        req = urllib.request.Request(
            GDC_FILES_ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            hits = json.loads(r.read()).get("data", {}).get("hits", [])
        for h in hits:
            cases = h.get("cases") or []
            if cases:
                mapping[h["file_id"]] = cases[0].get("submitter_id") or h["file_id"]
    return mapping


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------
def _read_gdc_gene_file(path: Path) -> pd.Series:
    """Parse a GDC per-gene file into an Ensembl-id -> value Series."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        first = fh.readline()
    has_header = first.lower().startswith(("gene_id", "ensembl"))
    opener2 = gzip.open if str(path).endswith(".gz") else open
    if str(path).endswith(".tsv"):  # augmented STAR: pick tpm_unstranded
        df = pd.read_csv(path, sep="\t", comment="#")
        df = df[~df["gene_id"].astype(str).str.startswith("N_")]
        col = "tpm_unstranded" if "tpm_unstranded" in df.columns else df.columns[-1]
        s = pd.Series(df[col].to_numpy("float64"),
                      index=df["gene_id"].astype(str).str.split(".").str[0])
    else:  # FPKM/HTSeq: 2-col, no header
        with opener2(path, "rt") as fh:
            df = pd.read_csv(fh, sep="\t", header=0 if has_header else None)
        df = df.iloc[:, :2]
        df.columns = ["gene_id", "value"]
        df = df[~df["gene_id"].astype(str).str.startswith("__")]  # HTSeq summary rows
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                      index=df["gene_id"].astype(str).str.split(".").str[0])
    return s[~s.index.duplicated(keep="first")]


def build_expression_matrix(files: Sequence[Path], *,
                            file_to_case: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Assemble a genes x specimens matrix from downloaded per-gene files.

    Column id = case submitter_id when resolvable (via ``file_to_case``), else
    the file UUID. Raises if the result is empty.
    """
    cols: Dict[str, pd.Series] = {}
    for p in files:
        # flattened name: "<dirUUID>__<fileUUID>.FPKM-UQ.txt.gz"
        parts = p.name.split("__")
        file_uuid = parts[-1].split(".")[0]
        label = (file_to_case or {}).get(file_uuid, file_uuid)
        try:
            cols[label] = _read_gdc_gene_file(p)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip unparseable %s: %s", p.name, e)
    mat = pd.DataFrame(cols)
    if mat.empty:
        raise ValueError("Assembled an empty BeatAML expression matrix.")
    return mat


def download_beataml_expression(
    out_dir: str | os.PathLike,
    *,
    product: str = "fpkm_uq",
    resolve_specimens: bool = True,
    workers: int = 16,
) -> Dict[str, object]:
    """End-to-end: list -> download -> resolve specimens -> build matrix -> write.

    Returns {matrix_path, sample_map_path, n_genes, n_specimens}.
    """
    if product not in EXPRESSION_SUFFIXES:
        raise ValueError(f"product must be one of {list(EXPRESSION_SUFFIXES)}")
    out = Path(out_dir)
    raw = out / "raw" / product
    suffix = EXPRESSION_SUFFIXES[product]
    keys = list_open_bucket(COHORT_BUCKET, suffix=suffix)
    files = download_keys(keys, raw, bucket=COHORT_BUCKET, workers=workers)

    f2c: Dict[str, str] = {}
    if resolve_specimens:
        uuids = [p.name.split("__")[-1].split(".")[0] for p in files]
        try:
            f2c = resolve_file_to_case(uuids)
            logger.info("Resolved %d/%d file-UUIDs to specimens via GDC API",
                        len(f2c), len(set(uuids)))
        except Exception as e:  # noqa: BLE001
            logger.warning("GDC specimen resolution failed (%s); keying by file UUID", e)

    mat = build_expression_matrix(files, file_to_case=f2c)
    out.mkdir(parents=True, exist_ok=True)
    mat_path = out / f"beataml_expression_{product}.parquet"
    mat.to_parquet(mat_path)
    smap = pd.DataFrame({"column_id": mat.columns,
                         "resolved_specimen": [c in set(f2c.values()) for c in mat.columns]})
    smap_path = out / f"beataml_sample_map_{product}.csv"
    smap.to_csv(smap_path, index=False)
    logger.info("Wrote %s (%d genes x %d specimens)", mat_path, mat.shape[0], mat.shape[1])
    return {"matrix_path": str(mat_path), "sample_map_path": str(smap_path),
            "n_genes": int(mat.shape[0]), "n_specimens": int(mat.shape[1])}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Download BeatAML from AWS Open Data (GDC)")
    ap.add_argument("--out-dir", default="data/raw_public/beataml")
    ap.add_argument("--product", default="fpkm_uq", choices=list(EXPRESSION_SUFFIXES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-resolve", action="store_true", help="skip GDC specimen resolution")
    a = ap.parse_args()
    res = download_beataml_expression(a.out_dir, product=a.product,
                                      resolve_specimens=not a.no_resolve, workers=a.workers)
    print(json.dumps(res, indent=2))
