"""Pull MMRF-COMMPASS clinical + treatment + sample timeline from GDC /cases API.

The GDC stores clinical and treatment data for MMRF-COMMPASS as case-level
metadata, not as downloadable files. This script paginates through all 995
cases and flattens the nested JSON into three TSVs that the rest of the
pipeline can consume:

    data/raw/mmrf_commpass/clinical.tsv      (one row per case)
    data/raw/mmrf_commpass/treatments.tsv    (one row per (case, treatment))
    data/raw/mmrf_commpass/samples.tsv       (one row per (case, sample, aliquot))

Usage:
    python scripts/pull_mmrf_clinical.py
    python scripts/pull_mmrf_clinical.py --output-dir data/raw/mmrf_commpass

Open-access; no GDC token required.

License: code is MIT (this repo). MMRF-COMMPASS data is governed by the GDC
data use policy (https://gdc.cancer.gov/about-data/data-sources).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("pull_mmrf_clinical")


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
PROJECT_ID = "MMRF-COMMPASS"
PAGE_SIZE = 100
RETRY_LIMIT = 5
RETRY_BACKOFF_SEC = 2.0

EXPAND_FIELDS = ",".join(
    [
        "demographic",
        "diagnoses",
        "diagnoses.treatments",
        "follow_ups",
        "follow_ups.molecular_tests",
        "samples",
        "samples.portions",
        "samples.portions.analytes",
        "samples.portions.analytes.aliquots",
        "exposures",
    ]
)


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to url with retries; return parsed response or raise."""
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(RETRY_LIMIT):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            sleep_for = RETRY_BACKOFF_SEC * (2 ** attempt)
            logger.warning(
                "GDC POST failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                RETRY_LIMIT,
                e,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"GDC POST failed after {RETRY_LIMIT} attempts: {last_err}")


def iter_cases() -> Iterator[dict[str, Any]]:
    """Paginate through all MMRF-COMMPASS cases with full expansion."""
    page_from = 0
    total: int | None = None
    while True:
        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": [PROJECT_ID],
                },
            },
            "expand": EXPAND_FIELDS,
            "size": PAGE_SIZE,
            "from": page_from,
            "format": "json",
        }
        resp = _post(GDC_CASES_ENDPOINT, payload)
        data = resp.get("data", {})
        hits = data.get("hits", [])
        if total is None:
            total = data.get("pagination", {}).get("total")
            logger.info("Discovered %d MMRF-COMMPASS cases", total)
        if not hits:
            break
        for case in hits:
            yield case
        page_from += len(hits)
        if total is not None and page_from >= total:
            break


def _join(values: list[str] | None, sep: str = ";") -> str:
    if not values:
        return ""
    return sep.join(str(v) for v in values if v not in (None, ""))


def flatten_clinical_row(case: dict[str, Any]) -> dict[str, Any]:
    """One row per case: demographic + first-diagnosis + survival + counts."""
    sub_id = case.get("submitter_id")
    case_id = case.get("case_id") or case.get("id")
    demo = case.get("demographic") or {}
    diags = case.get("diagnoses") or []
    primary = diags[0] if diags else {}

    # Aggregate treatment lines across all diagnoses (some patients have multiple
    # diagnosis records — first-line / relapse).
    n_treatments = sum(len(d.get("treatments") or []) for d in diags)
    therapeutic_agents_all = sorted(
        {
            t.get("therapeutic_agents")
            for d in diags
            for t in (d.get("treatments") or [])
            if t.get("therapeutic_agents")
        }
    )

    n_followups = len(case.get("follow_ups") or [])
    n_samples = len(case.get("samples") or [])

    return {
        "submitter_id": sub_id,
        "case_id": case_id,
        "iss_stage": primary.get("iss_stage"),
        "primary_diagnosis": primary.get("primary_diagnosis"),
        "age_at_diagnosis_days": primary.get("age_at_diagnosis"),
        "year_of_diagnosis": primary.get("year_of_diagnosis"),
        "gender": demo.get("gender"),
        "race": demo.get("race"),
        "ethnicity": demo.get("ethnicity"),
        "vital_status": demo.get("vital_status"),
        "days_to_death": demo.get("days_to_death"),
        "days_to_last_follow_up": primary.get("days_to_last_follow_up"),
        "n_treatments": n_treatments,
        "therapeutic_agents_all": _join(therapeutic_agents_all, sep="|"),
        "n_followups": n_followups,
        "n_samples": n_samples,
    }


def flatten_treatment_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (case, treatment). The longitudinal time-to-resistance signal
    lives here: regimen_or_line_of_therapy + days_to_treatment_start/end."""
    sub_id = case.get("submitter_id")
    rows: list[dict[str, Any]] = []
    for diag in case.get("diagnoses") or []:
        for trt in diag.get("treatments") or []:
            rows.append(
                {
                    "submitter_id": sub_id,
                    "diagnosis_id": diag.get("diagnosis_id"),
                    "treatment_submitter_id": trt.get("submitter_id"),
                    "regimen_or_line_of_therapy": trt.get("regimen_or_line_of_therapy"),
                    "therapeutic_agents": trt.get("therapeutic_agents"),
                    "days_to_treatment_start": trt.get("days_to_treatment_start"),
                    "days_to_treatment_end": trt.get("days_to_treatment_end"),
                    "treatment_or_therapy": trt.get("treatment_or_therapy"),
                    "treatment_outcome": trt.get("treatment_outcome"),
                    "treatment_intent_type": trt.get("treatment_intent_type"),
                    "initial_disease_status": trt.get("initial_disease_status"),
                }
            )
    return rows


def flatten_sample_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (case, sample, aliquot). Aliquot submitter IDs are the join
    key against gdc-client-downloaded RNA/SNV/CNV file names."""
    sub_id = case.get("submitter_id")
    rows: list[dict[str, Any]] = []
    for sample in case.get("samples") or []:
        sample_sub = sample.get("submitter_id")
        sample_type = sample.get("sample_type")
        tissue_type = sample.get("tissue_type")
        for portion in sample.get("portions") or []:
            for analyte in portion.get("analytes") or []:
                analyte_type = analyte.get("analyte_type")
                for aliquot in analyte.get("aliquots") or []:
                    rows.append(
                        {
                            "submitter_id": sub_id,
                            "sample_submitter_id": sample_sub,
                            "sample_type": sample_type,
                            "tissue_type": tissue_type,
                            "analyte_type": analyte_type,
                            "aliquot_id": aliquot.get("aliquot_id"),
                            "aliquot_submitter_id": aliquot.get("submitter_id"),
                        }
                    )
                if not analyte.get("aliquots"):
                    rows.append(
                        {
                            "submitter_id": sub_id,
                            "sample_submitter_id": sample_sub,
                            "sample_type": sample_type,
                            "tissue_type": tissue_type,
                            "analyte_type": analyte_type,
                            "aliquot_id": None,
                            "aliquot_submitter_id": None,
                        }
                    )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        logger.warning("No rows for %s — writing empty file", path)
        path.write_text("")
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    lines = ["\t".join(keys)]
    for r in rows:
        lines.append(
            "\t".join(
                "" if r.get(k) is None else str(r.get(k)).replace("\t", " ").replace("\n", " ")
                for k in keys
            )
        )
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %d rows × %d cols → %s", len(rows), len(keys), path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/raw/mmrf_commpass",
        help="Where to write clinical.tsv / treatments.tsv / samples.tsv",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical_rows: list[dict[str, Any]] = []
    treatment_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    n_cases = 0
    for case in iter_cases():
        n_cases += 1
        clinical_rows.append(flatten_clinical_row(case))
        treatment_rows.extend(flatten_treatment_rows(case))
        sample_rows.extend(flatten_sample_rows(case))
        if n_cases % 100 == 0:
            logger.info(
                "Processed %d cases (%d treatments, %d aliquots so far)",
                n_cases,
                len(treatment_rows),
                len(sample_rows),
            )

    logger.info(
        "Final: %d cases, %d treatments, %d aliquots",
        n_cases,
        len(treatment_rows),
        len(sample_rows),
    )

    write_tsv(out_dir / "clinical.tsv", clinical_rows)
    write_tsv(out_dir / "treatments.tsv", treatment_rows)
    write_tsv(out_dir / "samples.tsv", sample_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
