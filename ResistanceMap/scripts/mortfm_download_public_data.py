#!/usr/bin/env python3
"""
scripts/mortfm_download_public_data.py
======================================
Download (or link) raw inputs expected by ``scripts/mortfm_ingest_*.py`` into
``data/raw_public/``.

Usage:
    python3 scripts/mortfm_download_public_data.py
    python3 scripts/mortfm_download_public_data.py --only depmap,uniprot,reactome,chembl
    python3 scripts/mortfm_download_public_data.py --link-legacy   # reuse data/raw/
    python3 scripts/mortfm_download_public_data.py --dry-run

Sources without stable public URLs (Beat AML dbGaP molecular exports) are
skipped with instructions; use ``mortfm_ingest_beataml.py --mode processed`` when
``data/processed/beataml_*`` already exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw_public"
LEGACY_DATA_ROOT = REPO_ROOT / "data" / "raw"
DEPMAP_API = "https://depmap.org/portal/api/download/files"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_download")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_url(url: str, dest: Path, *, dry_run: bool = False, timeout: int = 3600) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("exists: %s", dest)
        return True
    if dry_run:
        logger.info("DRY-RUN would download %s -> %s", url[:120], dest)
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ResistanceMap-mortfm/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        tmp.replace(dest)
        logger.info("downloaded: %s (%d bytes)", dest, dest.stat().st_size)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.error("failed %s: %s", dest.name, exc)
        tmp.unlink(missing_ok=True)
        return False


def _link_or_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return True
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)
    logger.info("linked %s -> %s", dst, src)
    return True


def _merge_dir(src_dir: Path, dst_dir: Path) -> int:
    """Symlink/copy files from src_dir into dst_dir when missing."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if target.exists() or target.is_symlink():
            continue
        if item.is_dir():
            n += _merge_dir(item, target)
        else:
            if _link_or_copy(item, target):
                n += 1
    return n


def _depmap_catalog(release: str | None = None) -> list[dict[str, str]]:
    req = urllib.request.Request(DEPMAP_API, headers={"User-Agent": "ResistanceMap-mortfm/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    rows: list[dict[str, str]] = []
    for line in text.strip().splitlines():
        if not line or line.startswith("release,"):
            continue
        parts = line.split(",", 4)
        if len(parts) < 4:
            continue
        rel, _date, filename, url = parts[0], parts[1], parts[2], parts[3]
        if release and rel != release:
            continue
        rows.append({"release": rel, "filename": filename, "url": url})
    return rows


def _depmap_latest_release(catalog: list[dict[str, str]]) -> str:
    public = [r["release"] for r in catalog if r["release"].startswith("DepMap Public")]
    if not public:
        raise RuntimeError("No 'DepMap Public' releases in DepMap API catalog")
    # Releases sort lexicographically as 26Q1 > 25Q3 ...
    return sorted(set(public), reverse=True)[0]


def _depmap_url(catalog: list[dict[str, str]], release: str, filename: str) -> str | None:
    for row in catalog:
        if row["release"] == release and row["filename"] == filename:
            return row["url"]
    return None


# ---------------------------------------------------------------------------
# Per-source downloaders
# ---------------------------------------------------------------------------


def download_depmap(data_root: Path, *, dry_run: bool, release: str | None) -> bool:
    out = data_root / "depmap"
    catalog = _depmap_catalog()
    rel = release or _depmap_latest_release(catalog)
    logger.info("DepMap release: %s", rel)

    files = {
        "Model.csv": "Model.csv",
        "CRISPRGeneEffect.csv": "CRISPRGeneEffect.csv",
    }
    # Prefer new 26Q1 name; fall back to legacy name in older releases.
    expr_candidates = (
        "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    )
    expr_name = next(
        (n for n in expr_candidates if _depmap_url(catalog, rel, n)),
        expr_candidates[0],
    )
    files[expr_name] = expr_name

    ok = True
    for api_name, local_name in files.items():
        url = _depmap_url(catalog, rel, api_name)
        if not url:
            logger.warning("DepMap %s not in release %s", api_name, rel)
            ok = False
            continue
        dest = out / local_name
        if not _fetch_url(url, dest, dry_run=dry_run):
            ok = False

    # Legacy ingest alias.
    legacy_expr = out / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    new_expr = out / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    if not dry_run and new_expr.exists() and not legacy_expr.exists():
        try:
            os.symlink(new_expr.name, legacy_expr)
            logger.info("symlink %s -> %s", legacy_expr.name, new_expr.name)
        except OSError:
            pass
    return ok


def download_prism(data_root: Path, *, dry_run: bool, release: str | None) -> bool:
    out = data_root / "prism"
    catalog = _depmap_catalog()
    target = "secondary-screen-dose-response-curve-parameters.csv"
    prism_rows = [r for r in catalog if target in r["filename"] and "PRISM" in r["release"]]
    if not prism_rows:
        logger.error("PRISM file not found in DepMap API catalog")
        return False
    rels = sorted({r["release"] for r in prism_rows}, reverse=True)
    rel = release if release and any(r["release"] == release for r in prism_rows) else rels[0]
    url = _depmap_url(catalog, rel, target)
    if not url:
        logger.error("PRISM URL missing for release %s", rel)
        return False
    logger.info("PRISM release: %s", rel)
    return _fetch_url(url, out / target, dry_run=dry_run)


def download_gdsc(data_root: Path, *, dry_run: bool) -> bool:
    out = data_root / "gdsc"
    base = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5"
    files = {
        "GDSC2_fitted_dose_response_27Oct23.xlsx": f"{base}/GDSC2_fitted_dose_response_27Oct23.xlsx",
        "GDSC1_fitted_dose_response_27Oct23.xlsx": f"{base}/GDSC1_fitted_dose_response_27Oct23.xlsx",
        "Cell_Lines_Details.xlsx": f"{base}/Cell_Lines_Details.xlsx",
        "screened_compounds_rel_8.5.csv": f"{base}/screened_compounds_rel_8.5.csv",
    }
    return all(_fetch_url(url, out / name, dry_run=dry_run) for name, url in files.items())


def download_string(data_root: Path, *, dry_run: bool) -> bool:
    out = data_root / "string"
    files = {
        "9606.protein.links.full.v12.0.txt.gz":
            "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz",
        "9606.protein.aliases.v12.0.txt.gz":
            "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
    }
    return all(_fetch_url(url, out / name, dry_run=dry_run) for name, url in files.items())


def download_uniprot(data_root: Path, *, dry_run: bool) -> bool:
    out = data_root / "uniprot"
    url = (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"
    )
    return _fetch_url(url, out / "UP000005640_9606.fasta.gz", dry_run=dry_run)


def download_reactome(data_root: Path, *, dry_run: bool) -> bool:
    out = data_root / "reactome"
    base = "https://reactome.org/download/current"
    # HGNC2Reactome was removed from the public bundle; NCBI + UniProt still ship.
    candidates = (
        "NCBI2Reactome_All_Levels.txt",
        "UniProt2Reactome_All_Levels.txt",
        "NCBI2Reactome.txt",
    )
    ok = False
    for name in candidates:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            ok = True
            continue
        if _fetch_url(f"{base}/{name}", dest, dry_run=dry_run):
            ok = True
    return ok


def _chembl_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ResistanceMap-mortfm/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chembl_target_maps(target_ids: list[str], cache: dict[str, tuple[str, str]]) -> None:
    """Batch-resolve target_chembl_id -> (uniprot, gene_symbol)."""
    missing = [t for t in target_ids if t and t not in cache]
    if not missing:
        return
    chunk = 40
    for i in range(0, len(missing), chunk):
        batch = missing[i : i + chunk]
        in_clause = ",".join(batch)
        url = f"https://www.ebi.ac.uk/chembl/api/data/target.json?target_chembl_id__in={in_clause}&limit={chunk}"
        try:
            payload = _chembl_get_json(url)
        except urllib.error.HTTPError:
            for tid in batch:
                try:
                    t = _chembl_get_json(f"https://www.ebi.ac.uk/chembl/api/data/target/{tid}.json")
                except urllib.error.HTTPError:
                    continue
                _chembl_parse_target(t, cache)
            time.sleep(0.15)
            continue
        for t in payload.get("targets") or []:
            _chembl_parse_target(t, cache)
        time.sleep(0.15)


def _chembl_parse_target(t: dict, cache: dict[str, tuple[str, str]]) -> None:
    tid = t.get("target_chembl_id") or ""
    if not tid:
        return
    acc, gene = "", ""
    for comp in t.get("target_components") or []:
        if comp.get("accession"):
            acc = comp["accession"]
        for s in comp.get("target_component_synonyms") or []:
            if s.get("syn_type") == "GENE_SYMBOL" and s.get("component_synonym"):
                gene = s["component_synonym"]
                break
        if acc:
            break
    if acc:
        cache[tid] = (acc, gene)


def _chembl_molecule_names(mol_ids: list[str], cache: dict[str, str]) -> None:
    missing = [m for m in mol_ids if m and m not in cache]
    if not missing:
        return
    chunk = 50
    for i in range(0, len(missing), chunk):
        batch = missing[i : i + chunk]
        in_clause = ",".join(batch)
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_chembl_id__in={in_clause}&limit={chunk}"
        try:
            payload = _chembl_get_json(url)
        except urllib.error.HTTPError:
            for mid in batch:
                cache[mid] = mid
            continue
        for mol in payload.get("molecules") or []:
            mid = mol.get("molecule_chembl_id") or ""
            if mid:
                cache[mid] = mol.get("pref_name") or mid
        for mid in batch:
            cache.setdefault(mid, mid)
        time.sleep(0.15)


def download_chembl(data_root: Path, *, dry_run: bool, max_mechanisms: int | None = None) -> bool:
    out = data_root / "chembl" / "chembl_drug_targets.csv"
    min_bytes = 50_000 if max_mechanisms is None else 0
    if out.exists() and out.stat().st_size >= min_bytes:
        logger.info("exists: %s (%d bytes)", out, out.stat().st_size)
        return True
    if dry_run:
        logger.info("DRY-RUN would build ChEMBL drug-target export via REST API -> %s", out)
        return True

    out.parent.mkdir(parents=True, exist_ok=True)
    target_cache: dict[str, tuple[str, str]] = {}
    mol_cache: dict[str, str] = {}
    rows_out: list[dict[str, str]] = []

    offset = 0
    limit = 1000
    total = 0
    while True:
        url = (
            "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
            f"?molecular_mechanism=1&limit={limit}&offset={offset}"
        )
        payload = _chembl_get_json(url)
        mechs = payload.get("mechanisms") or []
        if not mechs:
            break

        tgt_ids = list({m.get("target_chembl_id") or "" for m in mechs} - {""})
        mol_ids = list({m.get("molecule_chembl_id") or "" for m in mechs} - {""})
        _chembl_target_maps(tgt_ids, target_cache)
        _chembl_molecule_names(mol_ids, mol_cache)

        for m in mechs:
            mol_id = m.get("molecule_chembl_id") or ""
            tgt_id = m.get("target_chembl_id") or ""
            if not mol_id or not tgt_id or tgt_id not in target_cache:
                continue
            uniprot, gene = target_cache[tgt_id]
            rows_out.append({
                "drug_chembl_id": mol_id,
                "drug_name": mol_cache.get(mol_id, mol_id),
                "target_uniprot": uniprot,
                "target_gene": gene,
                "mechanism_of_action": m.get("mechanism_of_action") or "",
            })
            total += 1
            if max_mechanisms and total >= max_mechanisms:
                break

        if max_mechanisms and total >= max_mechanisms:
            break
        meta = payload.get("page_meta") or {}
        if offset + limit >= int(meta.get("total_count", 0)):
            break
        offset += limit
        logger.info("ChEMBL mechanisms: %d rows (%d targets cached, offset %d)",
                    total, len(target_cache), offset)
        time.sleep(0.1)

    if not rows_out:
        logger.error("ChEMBL export produced zero rows")
        return False

    fieldnames = ["drug_chembl_id", "drug_name", "target_uniprot", "target_gene", "mechanism_of_action"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    logger.info("wrote %d ChEMBL drug-target rows -> %s", len(rows_out), out)
    return True


def download_geo(data_root: Path, *, dry_run: bool) -> bool:
    """Stage GEO supplementary archives under geo_single_cell/<GSE>/."""
    gses = ("GSE161195", "GSE223060", "GSE199373", "GSE153380", "GSE167968",
            "GSE106218", "GSE110499")
    out_base = data_root / "geo_single_cell"
    ok = True
    for gse in gses:
        gse_dir = out_base / gse
        if any(gse_dir.rglob("*.h5ad")):
            logger.info("GEO %s: h5ad already present", gse)
            continue
        stub = f"{gse[:len(gse) - 3]}nnn"
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/suppl/"
        if dry_run:
            logger.info("DRY-RUN would list and fetch %s", base)
            continue
        gse_dir.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(base, headers={"User-Agent": "ResistanceMap-mortfm/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            logger.warning("GEO %s: cannot list suppl/ (%s)", gse, exc)
            ok = False
            continue
        hrefs = re.findall(r'href="([^"]+)"', html)
        exts = (".tar", ".gz", ".tgz", ".h5", ".h5ad", ".mtx", ".zip")
        files = [
            h for h in hrefs
            if not h.startswith(("http", "/", "?", "#", ".."))
            and any(h.endswith(e) or h.endswith(e + ".gz") for e in exts)
        ]
        if not files:
            logger.warning("GEO %s: no supplementary archives found at %s", gse, base)
            ok = False
            continue
        for fname in files:
            dest = gse_dir / fname
            if not _fetch_url(base + fname, dest, dry_run=False, timeout=7200):
                ok = False
    return ok


def check_beataml(data_root: Path, processed_dir: Path) -> bool:
    """Beat AML dbGaP molecular files are not curl-able; verify processed or raw."""
    raw = data_root / "beataml"
    processed_markers = [
        processed_dir / "beataml_rpkm.parquet",
        processed_dir / "beataml" / "beataml_expression.parquet",
        processed_dir / "beataml_drug_response.tsv",
    ]
    if any(p.exists() for p in processed_markers):
        logger.info(
            "BeatAML: processed artifacts present under %s — use "
            "mortfm_ingest_beataml.py --mode processed (no public bulk download).",
            processed_dir,
        )
        return True
    if raw.is_dir() and any(raw.iterdir()):
        logger.info("BeatAML: raw files present under %s", raw)
        return True
    logger.warning(
        "BeatAML: no public automated download. Register at https://biodev.github.io/BeatAML2/ "
        "or place Tyner 2018 supplement files under %s, then run mortfm_ingest_beataml.py --mode auto.",
        raw,
    )
    return False


def link_legacy(data_root: Path, legacy_root: Path) -> None:
    """Reuse existing ``data/raw/`` tree where MORT-FM ingest inputs already exist."""
    pairs = (
        ("gdsc", "gdsc"),
        ("string", "string"),
        ("prism", "prism"),
        ("depmap", "depmap"),
        ("beataml", "beataml"),
    )
    for sub, name in pairs:
        src = legacy_root / sub
        dst = data_root / name
        if src.is_dir():
            n = _merge_dir(src, dst)
            if n:
                logger.info("legacy: merged %d file(s) %s -> %s", n, src, dst)

    # Top-level legacy h5ad -> geo_single_cell if accession matches.
    geo = data_root / "geo_single_cell"
    for h5ad in legacy_root.glob("*.h5ad"):
        gse = h5ad.stem.upper()
        if not gse.startswith("GSE"):
            continue
        gse_dir = geo / gse
        gse_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(h5ad, gse_dir / h5ad.name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DOWNLOADERS: dict[str, Callable[..., bool]] = {
    "depmap": download_depmap,
    "prism": download_prism,
    "gdsc": download_gdsc,
    "string": download_string,
    "uniprot": download_uniprot,
    "reactome": download_reactome,
    "chembl": download_chembl,
    "geo": download_geo,
}


def _selected(name: str, only: set[str] | None, skip: set[str]) -> bool:
    if only and name not in only:
        return False
    return name not in skip


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download MORT-FM public raw inputs.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--legacy-root", type=Path, default=LEGACY_DATA_ROOT)
    ap.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--only", default="", help="Comma-separated subset of sources")
    ap.add_argument("--skip", default="", help="Comma-separated sources to skip")
    ap.add_argument("--depmap-release", default=None, help="e.g. 'DepMap Public 26Q1'")
    ap.add_argument("--chembl-max", type=int, default=None, help="Limit mechanisms (debug)")
    ap.add_argument("--link-legacy", action="store_true",
                    help="Symlink/copy existing data/raw/ files before downloading")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_root: Path = args.data_root
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    data_root.mkdir(parents=True, exist_ok=True)
    if args.link_legacy and args.legacy_root.is_dir():
        link_legacy(data_root, args.legacy_root)

    results: dict[str, bool] = {}
    for name, fn in DOWNLOADERS.items():
        if not _selected(name, only, skip):
            continue
        logger.info("=== %s ===", name)
        if name in ("depmap", "prism"):
            results[name] = fn(data_root, dry_run=args.dry_run, release=args.depmap_release)
        elif name == "chembl":
            results[name] = fn(data_root, dry_run=args.dry_run, max_mechanisms=args.chembl_max)
        else:
            results[name] = fn(data_root, dry_run=args.dry_run)

    if _selected("beataml", only, skip):
        logger.info("=== beataml ===")
        results["beataml"] = check_beataml(data_root / "beataml", args.processed_dir)

    logger.info("=== summary ===")
    for name, ok in results.items():
        logger.info("  %-10s %s", name, "OK" if ok else "FAILED/SKIPPED")
    failed = [k for k, v in results.items() if not v]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
