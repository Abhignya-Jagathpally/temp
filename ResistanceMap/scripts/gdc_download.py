"""Download GDC open-access files from a manifest TSV.

Replaces the broken gdc-client binary path. Uses the GDC `/data/<uuid>`
endpoint with parallel HTTP, MD5 verification, and resume-by-existence.

The manifest format (matching gdc-client's manifest export and our API
manifests in ~/.gdc/mmrf_*.tsv) is a TSV with header columns:
    id  filename  md5  size  state

Usage:
    python scripts/gdc_download.py \
        --manifest ~/.gdc/mmrf_rna.tsv \
        --out-dir data/raw/mmrf_commpass/rna \
        --workers 8

Open-access only; no GDC token. For controlled data add `--token PATH`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("gdc_download")

GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
RETRY_LIMIT = 3
RETRY_BACKOFF_SEC = 2.0
CHUNK_SIZE = 1 << 20  # 1 MiB


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(
    uuid: str,
    filename: str,
    expected_md5: str,
    expected_size: int,
    out_dir: Path,
    token: str | None,
) -> tuple[str, str, int]:
    """Download one file. Returns (uuid, status, size_bytes).
    status ∈ {"ok", "skip", "fail"}."""
    target = out_dir / filename
    if target.exists() and target.stat().st_size == expected_size:
        if md5_of(target) == expected_md5:
            return uuid, "skip", expected_size
        target.unlink()  # corrupt; re-download

    url = f"{GDC_DATA_ENDPOINT}/{uuid}"
    headers = {}
    if token:
        headers["X-Auth-Token"] = token

    last_err: Exception | None = None
    for attempt in range(RETRY_LIMIT):
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
            actual_size = tmp.stat().st_size
            if actual_size != expected_size:
                tmp.unlink()
                raise IOError(
                    f"size mismatch: got {actual_size}, expected {expected_size}"
                )
            actual_md5 = md5_of(tmp)
            if actual_md5 != expected_md5:
                tmp.unlink()
                raise IOError(
                    f"md5 mismatch: got {actual_md5}, expected {expected_md5}"
                )
            tmp.rename(target)
            return uuid, "ok", actual_size
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, TimeoutError) as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            sleep_for = RETRY_BACKOFF_SEC * (2 ** attempt)
            logger.warning(
                "[%s] attempt %d/%d failed: %s — retrying in %.1fs",
                filename,
                attempt + 1,
                RETRY_LIMIT,
                e,
                sleep_for,
            )
            time.sleep(sleep_for)

    logger.error("[%s] gave up after %d attempts: %s", filename, RETRY_LIMIT, last_err)
    return uuid, "fail", 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token", type=Path, default=None,
                        help="GDC user token file (only needed for controlled data)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    token = args.token.read_text().strip() if args.token else None

    with args.manifest.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in rows if r.get("id")]
    total_bytes = sum(int(r["size"]) for r in rows)
    logger.info(
        "Manifest: %d files, %.2f GB total → %s",
        len(rows),
        total_bytes / 1e9,
        args.out_dir,
    )

    n_ok = n_skip = n_fail = 0
    bytes_done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                fetch_one,
                r["id"],
                r["filename"],
                r["md5"],
                int(r["size"]),
                args.out_dir,
                token,
            )
            for r in rows
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            uuid, status, sz = fut.result()
            bytes_done += sz
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
            if i % 25 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = bytes_done / 1e6 / max(elapsed, 1e-3)
                logger.info(
                    "[%d/%d] ok=%d skip=%d fail=%d  %.2f GB / %.2f GB  %.1f MB/s",
                    i,
                    len(futures),
                    n_ok,
                    n_skip,
                    n_fail,
                    bytes_done / 1e9,
                    total_bytes / 1e9,
                    rate,
                )

    elapsed = time.time() - t0
    logger.info(
        "DONE in %.1fs: %d ok, %d skip (already-present), %d fail. %.2f GB written.",
        elapsed,
        n_ok,
        n_skip,
        n_fail,
        bytes_done / 1e9,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
