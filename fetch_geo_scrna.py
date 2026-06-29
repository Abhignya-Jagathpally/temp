"""Download single-cell supplementary files for a GEO series (RQ10 staging).

Uses GEOparse (already installed in your venv). GEO scRNA counts are usually
distributed as per-sample supplementary 10x triplets (barcodes/features/matrix)
inside .tar.gz. This pulls them into data/raw/scrna/<ACC>/ for geo_to_h5ad.R.

    python scripts/fetch_geo_scrna.py --acc GSE223060 --out data/raw/scrna/GSE223060

Verified open accessions (see ../DATA_SCRNA.md): GSE223060 (scRNA) / GSE223061 (bulk).
This is a data download (NCBI GEO), not web scraping; no fabrication.
"""
from __future__ import annotations
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", required=True, help="GEO series accession, e.g. GSE223060")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("data", "raw", "scrna", a.acc)
    os.makedirs(out, exist_ok=True)
    try:
        import GEOparse
    except ImportError:
        sys.exit("pip install GEOparse (already in your venv per the Phase-A report).")
    print(f"[GEO] fetching {a.acc} metadata -> {out}")
    gse = GEOparse.get_GEO(geo=a.acc, destdir=out, silent=True)
    # series-level supplementary (often the count matrices live here)
    try:
        gse.download_supplementary_files(directory=out, download_sra=False)
    except Exception as e:
        print(f"[warn] series supplementary fetch: {e}")
    # per-sample supplementary as fallback
    for gsm_name, gsm in getattr(gse, "gsms", {}).items():
        try:
            gsm.download_supplementary_files(directory=os.path.join(out, gsm_name), download_sra=False)
        except Exception as e:
            print(f"[warn] {gsm_name}: {e}")
    print(f"[GEO] done. Unpack any .tar.gz, then run scripts/geo_to_h5ad.R on the 10x dir.")
    # write the sample->patient mapping skeleton (for CoMMpass linkage)
    import csv
    with open(os.path.join(out, "sample_metadata.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gsm", "title", "patient_id_TODO"])
        for name, gsm in getattr(gse, "gsms", {}).items():
            w.writerow([name, ";".join(gsm.metadata.get("title", [])), ""])
    print("[GEO] wrote sample_metadata.csv — fill patient_id_TODO for bulk<->cell linkage.")


if __name__ == "__main__":
    main()