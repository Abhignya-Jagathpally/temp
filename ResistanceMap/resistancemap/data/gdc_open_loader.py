"""GDC open-tier MMRF-CoMMpass loader (no dbGaP / no token required).

The GDC open access tier exposes MMRF-COMMPASS **gene expression (STAR Counts
TPM/FPKM)** and **harmonised clinical outcomes** (demographics, ISS staging,
vital status, follow-up) without controlled-access credentials. This module is
the importable library wrapper around that tier — it reuses the same REST
endpoints the standalone ingestion scripts (``scripts/pull_mmrf_clinical.py``,
``scripts/preprocess_mmrf_gdc.py``) already use, but exposes clean functions for
the v20 lab-first pipeline.

What it does NOT do
-------------------
* It does NOT provide the longitudinal ``PER_PATIENT_VISIT`` routine labs — those
  live behind the free MMRF Virtual Lab, not the GDC open tier.
* It NEVER fabricates data. Empty API responses, missing files, or empty
  matrices raise; nothing is back-filled with synthetic values.

Honest scope: this unblocks the ~995-patient open-tier RNA-seq + OS cohort that
the v20 validation track (GDC clinical outcomes) depends on.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # requests is a hard dep, but keep the import explicit for clarity.
    import requests
except ImportError as exc:  # pragma: no cover - requests is in core deps
    raise ImportError(
        "gdc_open_loader requires `requests`. Install project dependencies."
    ) from exc

logger = logging.getLogger(__name__)

GDC_BASE = "https://api.gdc.cancer.gov"
GDC_CASES_ENDPOINT = f"{GDC_BASE}/cases"
GDC_FILES_ENDPOINT = f"{GDC_BASE}/files"
GDC_DATA_ENDPOINT = f"{GDC_BASE}/data"
PROJECT_ID = "MMRF-COMMPASS"
PAGE_SIZE = 100
RETRY_LIMIT = 5
RETRY_BACKOFF_SEC = 2.0

_EXPAND_FIELDS = ",".join(
    [
        "demographic",
        "diagnoses",
        "diagnoses.treatments",
        "follow_ups",
        "samples",
        "samples.portions.analytes.aliquots",
    ]
)


# ---------------------------------------------------------------------------
# Low-level REST helpers
# ---------------------------------------------------------------------------
def _post(url: str, payload: Dict[str, Any], *, stream: bool = False) -> requests.Response:
    """POST JSON with retries; raise on persistent failure."""
    last_err: Optional[Exception] = None
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.post(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                stream=stream,
                timeout=120,
            )
            resp.raise_for_status()
            return resp
        except Exception as err:  # noqa: BLE001 - retried below
            last_err = err
            sleep_s = RETRY_BACKOFF_SEC * (attempt + 1)
            logger.warning("GDC POST %s failed (attempt %d): %s", url, attempt + 1, err)
            time.sleep(sleep_s)
    raise RuntimeError(f"GDC POST {url} failed after {RETRY_LIMIT} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Clinical outcomes (open tier)
# ---------------------------------------------------------------------------
def iter_mmrf_cases() -> Iterator[Dict[str, Any]]:
    """Paginate through all MMRF-COMMPASS cases with clinical expansion."""
    page_from = 0
    total: Optional[int] = None
    while True:
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "project.project_id", "value": [PROJECT_ID]},
            },
            "expand": _EXPAND_FIELDS,
            "size": PAGE_SIZE,
            "from": page_from,
            "format": "json",
        }
        data = _post(GDC_CASES_ENDPOINT, payload).json().get("data", {})
        hits = data.get("hits", [])
        if total is None:
            total = data.get("pagination", {}).get("total")
            logger.info("Discovered %s MMRF-COMMPASS cases", total)
        if not hits:
            break
        for case in hits:
            yield case
        page_from += len(hits)
        if total is not None and page_from >= total:
            break


def flatten_clinical_hit(case: Dict[str, Any]) -> Dict[str, Any]:
    """One row per case: demographic + first-diagnosis + survival.

    Mirrors ``scripts/pull_mmrf_clinical.flatten_clinical_row`` so downstream
    schemas stay identical between the script and library paths.
    """
    demo = case.get("demographic") or {}
    diags = case.get("diagnoses") or []
    primary = diags[0] if diags else {}
    return {
        "submitter_id": case.get("submitter_id"),
        "case_id": case.get("case_id") or case.get("id"),
        "iss_stage": primary.get("iss_stage"),
        "age_at_diagnosis_days": primary.get("age_at_diagnosis"),
        "gender": demo.get("gender"),
        "race": demo.get("race"),
        "vital_status": demo.get("vital_status"),
        "days_to_death": demo.get("days_to_death"),
        "days_to_last_follow_up": primary.get("days_to_last_follow_up"),
    }


def parse_gdc_clinical_hits(hits: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Pure: flatten a list of GDC case hits into a clinical DataFrame.

    Separated from the network call so it is unit-testable without the API.
    """
    if not hits:
        raise ValueError("No GDC case hits provided — refusing to emit an empty cohort.")
    return pd.DataFrame([flatten_clinical_hit(h) for h in hits])


def download_mmrf_open_clinical(output_dir: str | os.PathLike) -> pd.DataFrame:
    """Fetch open-tier MMRF clinical outcomes, write ``clinical_open.tsv``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hits = list(iter_mmrf_cases())
    df = parse_gdc_clinical_hits(hits)
    path = out / "clinical_open.tsv"
    df.to_csv(path, sep="\t", index=False)
    logger.info("Wrote %d clinical rows to %s", len(df), path)
    return df


def extract_survival_labels(
    clinical_df: pd.DataFrame, *, dead_value: str = "Dead"
) -> Dict[str, np.ndarray]:
    """Build an OS (event, time) structured array from open clinical fields.

    event = (vital_status == dead_value); time = days_to_death if dead else
    days_to_last_follow_up. Rows with no usable time are dropped (never
    imputed). Returns ``{submitter_id, event, time, structured}``.
    """
    required = {"vital_status", "days_to_death", "days_to_last_follow_up"}
    missing = required - set(clinical_df.columns)
    if missing:
        raise ValueError(f"clinical_df missing survival columns: {sorted(missing)}")

    dead = clinical_df["vital_status"].astype(str).str.lower() == dead_value.lower()
    time = np.where(
        dead, clinical_df["days_to_death"], clinical_df["days_to_last_follow_up"]
    ).astype("float64")
    keep = np.isfinite(time) & (time > 0)
    if not keep.any():
        raise ValueError("No patients with a usable OS time — refusing empty labels.")

    sub = clinical_df.loc[keep, "submitter_id"].to_numpy() if "submitter_id" in clinical_df else np.arange(keep.sum())
    event = dead.to_numpy()[keep]
    t = time[keep]
    structured = np.array(
        [(bool(e), float(x)) for e, x in zip(event, t)],
        dtype=[("event", bool), ("time", float)],
    )
    return {"submitter_id": sub, "event": event, "time": t, "structured": structured}


# ---------------------------------------------------------------------------
# Gene expression (STAR Counts, open tier)
# ---------------------------------------------------------------------------
def query_rnaseq_manifest(workflow: str = "STAR - Counts") -> List[Dict[str, Any]]:
    """List open-access MMRF RNA-seq quantification files (file_id + name)."""
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJECT_ID]}},
                {"op": "in", "content": {"field": "data_category", "value": ["Transcriptome Profiling"]}},
                {"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
            ],
        },
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.submitter_id",
        "size": 100000,
        "format": "json",
    }
    hits = _post(GDC_FILES_ENDPOINT, payload).json().get("data", {}).get("hits", [])
    if not hits:
        raise RuntimeError(
            "GDC returned no open-access STAR-Counts files for MMRF-COMMPASS — "
            "check connectivity / workflow name; not fabricating a manifest."
        )
    logger.info("Found %d open RNA-seq files", len(hits))
    return hits


def download_files(file_ids: Sequence[str], output_dir: str | os.PathLike) -> List[Path]:
    """Download GDC files by UUID to ``output_dir`` (one file per UUID)."""
    if not file_ids:
        raise ValueError("No file_ids provided to download.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fid in file_ids:
        resp = _post(GDC_DATA_ENDPOINT, {"ids": [fid]}, stream=True)
        # Single-file download returns the file body directly.
        dest = out / f"{fid}.tsv"
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        written.append(dest)
    logger.info("Downloaded %d files to %s", len(written), out)
    return written


def _read_star_counts(path: str | os.PathLike, *, value_col: str = "tpm_unstranded") -> pd.Series:
    """Parse one STAR-Counts TSV into a gene_id -> value Series (no summary rows)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", comment="#")
    # STAR summary rows have gene_id like N_unmapped/N_multimapping/etc.
    df = df[~df["gene_id"].astype(str).str.startswith("N_")]
    if value_col not in df.columns:
        raise ValueError(f"{path}: expected column {value_col!r}, have {list(df.columns)}")
    gid = df["gene_id"].astype(str).str.split(".").str[0]  # strip Ensembl version
    s = pd.Series(df[value_col].to_numpy(dtype="float64"), index=gid)
    return s[~s.index.duplicated(keep="first")]


def build_expression_matrix(
    star_dir: str | os.PathLike, *, value_col: str = "tpm_unstranded"
) -> pd.DataFrame:
    """Assemble a genes x samples expression matrix from STAR-Counts TSVs.

    Sample id = the file stem. Raises if the directory has no parseable files.
    """
    d = Path(star_dir)
    files = sorted([p for p in d.glob("*.tsv")] + [p for p in d.glob("*.tsv.gz")])
    if not files:
        raise FileNotFoundError(f"No STAR-Counts TSVs found under {d}")
    cols: Dict[str, pd.Series] = {}
    for p in files:
        cols[p.name.split(".")[0]] = _read_star_counts(p, value_col=value_col)
    mat = pd.DataFrame(cols)
    if mat.empty:
        raise ValueError(f"Parsed an empty expression matrix from {d}.")
    return mat


def merge_expression_clinical(
    expr_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    *,
    sample_to_case: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Align a genes x samples matrix to per-case clinical rows.

    ``sample_to_case`` maps expression column (sample/aliquot id) -> submitter_id.
    Returns a samples x (genes + clinical) table for matched samples only.
    """
    samples = list(expr_df.columns)
    if sample_to_case is not None:
        case_ids = [sample_to_case.get(s) for s in samples]
    else:
        case_ids = samples  # assume columns are already submitter_ids
    expr_t = expr_df.T.copy()
    expr_t.insert(0, "submitter_id", case_ids)
    merged = expr_t.merge(clinical_df, on="submitter_id", how="inner")
    if merged.empty:
        raise ValueError(
            "No expression sample matched a clinical case — check sample_to_case map."
        )
    return merged
