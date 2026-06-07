#!/usr/bin/env python3
"""
Download GEO scRNA-seq datasets for ResistanceMap v20.
All datasets are freely downloadable without registration.

Downloads:
  - GSE161801: RRMM pre/post-treatment (20 patients, scRNA+scATAC)
  - GSE189460: Bortezomib responders vs non-responders (18 patients)
  - Zenodo MM Atlas: panImmune.h5ad (integrated multi-cohort, CC-BY-4.0)
"""

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data/open_access/geo_scrna")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path, description: str):
    """Download a file with wget or requests fallback."""
    if output_path.exists():
        print(f"  [SKIP] {description} already exists: {output_path}")
        return

    print(f"  [DOWNLOADING] {description}...")
    print(f"    URL: {url}")
    print(f"    Output: {output_path}")

    # Try wget first (fastest, shows progress)
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_path), url],
            check=True, timeout=600
        )
        print(f"  [OK] {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to requests
    try:
        import requests
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.0f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="")
        print(f"\n  [OK] {output_path.name}")
    except Exception as e:
        print(f"  [FAIL] {description}: {e}")
        if output_path.exists():
            output_path.unlink()


def main():
    print("=" * 60)
    print("ResistanceMap v20 — GEO scRNA-seq Download")
    print("=" * 60)

    # --- 1. Zenodo MM Atlas (panImmune.h5ad) ---
    # This is the fastest to use: already preprocessed, h5ad format, CC-BY-4.0
    print("\n[1/3] Zenodo MM Immune Atlas (panImmune.h5ad, ~2.7 GB)")
    download_file(
        url="https://zenodo.org/records/13646014/files/panImmune.h5ad?download=1",
        output_path=DATA_DIR / "panImmune.h5ad",
        description="Zenodo MM panImmune Atlas"
    )

    # Also grab the metadata
    download_file(
        url="https://zenodo.org/records/13646014/files/metadata-donor.csv?download=1",
        output_path=DATA_DIR / "metadata-donor.csv",
        description="Zenodo donor metadata"
    )

    # --- 2. GSE161801 (RRMM multi-omics) ---
    print("\n[2/3] GSE161801 — RRMM pre/post-treatment scRNA-seq")
    download_file(
        url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE161801&format=file",
        output_path=DATA_DIR / "GSE161801_RAW.tar",
        description="GSE161801 supplementary files"
    )

    # Extract if tar exists
    tar_path = DATA_DIR / "GSE161801_RAW.tar"
    extract_dir = DATA_DIR / "GSE161801"
    if tar_path.exists() and not extract_dir.exists():
        print("  Extracting...")
        extract_dir.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(extract_dir)], check=True)
        print(f"  Extracted to {extract_dir}")

    # --- 3. GSE189460 (Bortezomib response) ---
    print("\n[3/3] GSE189460 — Bortezomib response scRNA-seq")
    download_file(
        url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE189460&format=file",
        output_path=DATA_DIR / "GSE189460_RAW.tar",
        description="GSE189460 supplementary files"
    )

    tar_path = DATA_DIR / "GSE189460_RAW.tar"
    extract_dir = DATA_DIR / "GSE189460"
    if tar_path.exists() and not extract_dir.exists():
        print("  Extracting...")
        extract_dir.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(extract_dir)], check=True)
        print(f"  Extracted to {extract_dir}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Download Summary:")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(DATA_DIR)} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()