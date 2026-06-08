#!/usr/bin/env python3
"""Stage MMRF-COMMPASS open-tier RNA-seq (STAR counts) for the Spark lakehouse.

Downloads the GDC open STAR-Counts files into data/raw/mmrf_commpass/rna/,
writes the ~/.gdc/mmrf_rna.tsv manifest, then runs preprocess_mmrf_gdc.py to
build data/raw/mmrf_commpass/gene_expression.tsv (genes x aliquots).

No token / no dbGaP — MMRF gene expression quantification is GDC open access.
Nothing fabricated: failed downloads are logged and the per-file matrix is built
only from what actually landed.
"""
from __future__ import annotations

import concurrent.futures
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resistancemap.data.gdc_open_loader import query_rnaseq_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage_mmrf_rna")

RNA_DIR = Path("data/raw/mmrf_commpass/rna")
GDC_DIR = Path.home() / ".gdc"
DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def _download(h: dict) -> tuple[str, bool]:
    fid, fn = h["file_id"], h["file_name"]
    dest = RNA_DIR / fn
    if dest.exists() and dest.stat().st_size > 0:
        return fn, True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(f"{DATA_ENDPOINT}/{fid}", tmp)  # noqa: S310
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return fn, False
        tmp.rename(dest)
        return fn, True
    except Exception as e:  # noqa: BLE001
        logger.warning("download failed %s: %s", fn, e)
        tmp.unlink(missing_ok=True)
        return fn, False


def main() -> int:
    RNA_DIR.mkdir(parents=True, exist_ok=True)
    GDC_DIR.mkdir(exist_ok=True)
    hits = query_rnaseq_manifest("STAR - Counts")
    logger.info("MMRF STAR-Counts files to stage: %d", len(hits))

    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for i, (fn, good) in enumerate(ex.map(_download, hits), 1):
            ok += int(good)
            if i % 100 == 0:
                logger.info("  downloaded %d/%d (%d ok)", i, len(hits), ok)
    logger.info("Downloaded %d/%d STAR files", ok, len(hits))
    if ok == 0:
        logger.error("No STAR files downloaded — aborting (not fabricating).")
        return 1

    # GDC manifest for build_file_to_case_lookup (needs id + filename).
    man = GDC_DIR / "mmrf_rna.tsv"
    with open(man, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for h in hits:
            f.write(f"{h['file_id']}\t{h['file_name']}\t\t\treleased\n")
    logger.info("Wrote manifest %s", man)

    # Build gene_expression.tsv (RNA only; skip snv/cnv).
    cmd = [
        sys.executable, "-u", "scripts/preprocess_mmrf_gdc.py",
        "--mmrf-dir", "data/raw/mmrf_commpass",
        "--manifest-dir", str(GDC_DIR),
        "--skip", "snv", "--skip", "cnv",
    ]
    logger.info("Running: %s", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    logger.info("preprocess_mmrf_gdc exit=%d", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
