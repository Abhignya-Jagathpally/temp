#!/usr/bin/env bash
################################################################################
# ResistanceMap — Automated data download
#
# Fetches all *publicly* curl-able datasets the pipeline needs and stages them
# under data/raw/ in the layout expected by configs/default.yaml.
#
# Sources (all stable bulk endpoints — no browser sessions):
#   STRING PPI v12 .................. stringdb-downloads.org
#   GEO scRNA-seq (GSE124310/271107)  ftp.ncbi.nlm.nih.gov
#   ENCODE H3K4me3 / H3K27me3 ....... encodeproject.org REST API
#   DepMap CCLE proteomics .......... depmap.org / Broad figshare mirror
#   GDSC1 + GDSC2 drug response ..... cog.sanger.ac.uk Cell Model Passports
#   CTRPv2 drug sensitivity ......... ctd2-data.nci.nih.gov
#
# NOT downloaded (controlled access OR release-versioned IDs):
#   MMRF CoMMpass ................... GDC dbGaP, requires IRB + GDC token
#   DepMap CRISPR (CRISPRGeneEffect)  figshare ID changes per quarterly release
#   HMCL Keats Lab MM panel ......... MMRF researcher gateway (request access)
#   PRISM Broad Repurposing Hub ..... figshare ID changes per release
#
# Usage:
#   bash scripts/download_data.sh                      # everything public
#   bash scripts/download_data.sh --skip ENCODE,GEO    # comma-list to skip
#   bash scripts/download_data.sh --only STRING,GDSC   # comma-list to keep
#   bash scripts/download_data.sh --dry-run            # print URLs only
#   DATA_ROOT=/scratch/foo bash scripts/download_data.sh
#
# Requirements:
#   curl, wget, gzip, gunzip, jq, python3 (for JSON parsing fallback)
#   For MMRF only: gdc-client + valid GDC token (not invoked here)
################################################################################

set -uo pipefail  # not -e: per-source failures must not abort the whole run

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-data/raw}"
DRY_RUN=0
SKIP=""
ONLY=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()   { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}  !${NC} $*"; }
err()   { echo -e "${RED}  ✗${NC} $*" >&2; }
section(){ echo; echo -e "${GREEN}=== $* ===${NC}"; }

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=1; shift ;;
    --skip)    SKIP="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,30p' "$0"
      exit 0 ;;
    *) err "Unknown option: $1"; exit 2 ;;
  esac
done

want() {
  local name="$1"
  if [[ -n "$ONLY" ]]; then
    [[ ",$ONLY," == *",$name,"* ]]
  else
    [[ ",$SKIP," != *",$name,"* ]]
  fi
}

# fetch URL DEST  — resumable, idempotent, dry-run aware
fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    ok "exists: $dest"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    DRY: curl -L --fail --retry 5 --retry-delay 5 -C - -o $dest $url"
    return 0
  fi
  if curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"
    ok "downloaded: $dest"
  else
    err "failed: $url"
    rm -f "$dest.part"
    return 1
  fi
}

require() {
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || { err "missing required command: $c"; exit 3; }
  done
}

require curl gzip
command -v jq      >/dev/null 2>&1 || warn "jq not found — ENCODE step will fall back to python3"
command -v python3 >/dev/null 2>&1 || warn "python3 not found — ENCODE fallback unavailable"

mkdir -p "$DATA_ROOT"
log "DATA_ROOT=$DATA_ROOT  DRY_RUN=$DRY_RUN  SKIP='$SKIP'  ONLY='$ONLY'"

# ─── STRING PPI v12 (human, 9606) ────────────────────────────────────────────
if want STRING; then
  section "STRING PPI v12 (Homo sapiens)"
  STRING_DIR="$DATA_ROOT/string"
  fetch "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz" \
        "$STRING_DIR/9606.protein.links.v12.0.txt.gz"
  fetch "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz" \
        "$STRING_DIR/9606.protein.info.v12.0.txt.gz"

  if [[ $DRY_RUN -eq 0 ]]; then
    # Decompress + project to (protein1 protein2 combined_score) at the path
    # configs/default.yaml expects: data/raw/string_ppi.txt
    if [[ ! -s "$DATA_ROOT/string_ppi.txt" ]]; then
      log "Building $DATA_ROOT/string_ppi.txt from STRING bulk files"
      gunzip -kf "$STRING_DIR/9606.protein.links.v12.0.txt.gz"
      # Bulk file already has header: protein1 protein2 combined_score
      cp "$STRING_DIR/9606.protein.links.v12.0.txt" "$DATA_ROOT/string_ppi.txt"
      ok "wrote $DATA_ROOT/string_ppi.txt"
      warn "Ensembl→gene-symbol conversion still required; see scripts/convert_string_ids.py"
    fi
  fi
fi

# ─── GEO scRNA-seq series via NCBI FTP ───────────────────────────────────────
fetch_geo_series() {
  local gse="$1"
  # GSE124310 -> GSE124nnn
  local stub="${gse:0:$((${#gse}-3))}nnn"
  local base="https://ftp.ncbi.nlm.nih.gov/geo/series/${stub}/${gse}"
  local outdir="$DATA_ROOT/geo/${gse}"
  mkdir -p "$outdir"
  log "GEO ${gse}: listing ${base}/suppl/"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    DRY: curl -s ${base}/suppl/ | parse + fetch each file"
    return 0
  fi
  # Pull the directory index, extract hrefs, fetch each supplementary file.
  local listing
  listing=$(curl -sL --fail "${base}/suppl/" || true)
  if [[ -z "$listing" ]]; then
    warn "${gse}: no suppl/ listing returned"
    return 0
  fi
  # Only relative filenames; reject anything with a scheme, slash, or query
  echo "$listing" \
    | grep -oE 'href="[^"]+"' \
    | sed 's/href="//; s/"$//' \
    | grep -vE '^(https?:|/|\?|#|\.\.)' \
    | grep -E '\.(tar|gz|tgz|h5|h5ad|txt|csv|tsv|mtx|xlsx|bw|bed|zip)(\.gz)?$' \
    | while read -r f; do
        fetch "${base}/suppl/${f}" "${outdir}/${f}" || warn "skip ${f}"
      done
  ok "GEO ${gse} staged in ${outdir}"
}

if want GEO; then
  section "GEO scRNA-seq (GSE124310, GSE271107)"
  fetch_geo_series GSE124310 || warn "GSE124310 partial"
  fetch_geo_series GSE271107 || warn "GSE271107 partial"
  warn "If you need .h5ad specifically, convert from the supplementary matrices via scanpy"
fi

# ─── ENCODE H3K4me3 / H3K27me3 ChIP-seq ──────────────────────────────────────
ENCODE_BIOSAMPLES="${ENCODE_BIOSAMPLES:-K562,MCF-7,A549}"
encode_search() {
  local target="$1" outdir="$2"
  local biosample_filter=""
  IFS=',' read -ra _bs <<< "$ENCODE_BIOSAMPLES"
  for b in "${_bs[@]}"; do
    biosample_filter+="&biosample_ontology.term_name=$(printf '%s' "$b" | sed 's/ /+/g')"
  done
  local query="https://www.encodeproject.org/search/?type=Experiment&assay_title=Histone+ChIP-seq&target.label=${target}&status=released&assembly=GRCh38${biosample_filter}&format=json&limit=all"
  log "ENCODE: querying experiments for ${target}"
  mkdir -p "$outdir"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    DRY: curl '$query' | jq '.@graph[].files[].cloud_metadata.url' | xargs curl -O"
    return 0
  fi
  local meta="$outdir/_search.json"
  curl -sL --fail -H "Accept: application/json" "$query" -o "$meta"
  local urls=""
  if command -v jq >/dev/null 2>&1; then
    urls=$(jq -r '.["@graph"][]?.files[]? | select(.file_format=="bed" or .file_format=="bigBed") | select(.cloud_metadata?.url) | .cloud_metadata.url' "$meta" | sort -u)
  else
    urls=$(python3 -c "
import json,sys
d=json.load(open('$meta'))
out=set()
for exp in d.get('@graph',[]):
    for f in exp.get('files',[]) or []:
        if f.get('file_format') in ('bed','bigBed') and f.get('cloud_metadata',{}).get('url'):
            out.add(f['cloud_metadata']['url'])
for u in sorted(out): print(u)
")
  fi
  local n; n=$(printf '%s\n' "$urls" | grep -c . || true)
  log "ENCODE ${target}: ${n} peak files"
  printf '%s\n' "$urls" | while read -r u; do
    [[ -z "$u" ]] && continue
    fetch "$u" "$outdir/$(basename "$u")"
  done
}

if want ENCODE; then
  section "ENCODE Histone ChIP-seq (H3K4me3, H3K27me3)"
  encode_search H3K4me3  "$DATA_ROOT/ccle_epigenomics/h3k4me3"
  encode_search H3K27me3 "$DATA_ROOT/ccle_epigenomics/h3k27me3"
  warn "Pipeline expects CSVs at h3k4me3.csv / h3k27me3.csv — aggregate peak files per cell line as a downstream step"
fi

# ─── DepMap CCLE proteomics (Nusinow 2020, normalized) ───────────────────────
if want DEPMAP; then
  section "DepMap CCLE proteomics"
  # Stable Broad/figshare mirror for protein_quant_current_normalized.csv.gz
  # Pinned release: 21Q4 (Nusinow 2020 normalized matrix)
  DEPMAP_URL="https://ndownloader.figshare.com/files/27902376"
  fetch "$DEPMAP_URL" "$DATA_ROOT/depmap/protein_quant_current_normalized.csv.gz"
  if [[ $DRY_RUN -eq 0 && -s "$DATA_ROOT/depmap/protein_quant_current_normalized.csv.gz" ]]; then
    gunzip -kf "$DATA_ROOT/depmap/protein_quant_current_normalized.csv.gz"
    cp "$DATA_ROOT/depmap/protein_quant_current_normalized.csv" "$DATA_ROOT/ccle_proteomics.csv"
    ok "wrote $DATA_ROOT/ccle_proteomics.csv"
  fi
fi

# ─── GDSC1 / GDSC2 drug response (Sanger Cell Model Passports) ───────────────
if want GDSC; then
  section "GDSC1 + GDSC2 drug response"
  GDSC_BASE="https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5"
  fetch "$GDSC_BASE/GDSC1_fitted_dose_response_27Oct23.xlsx" \
        "$DATA_ROOT/gdsc/GDSC1_fitted_dose_response_27Oct23.xlsx"
  fetch "$GDSC_BASE/GDSC2_fitted_dose_response_27Oct23.xlsx" \
        "$DATA_ROOT/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"
  # Cell line + compound annotations
  fetch "$GDSC_BASE/Cell_Lines_Details.xlsx" \
        "$DATA_ROOT/gdsc/Cell_Lines_Details.xlsx"
  fetch "$GDSC_BASE/screened_compounds_rel_8.5.csv" \
        "$DATA_ROOT/gdsc/screened_compounds_rel_8.5.csv"
  warn "Pipeline expects gdsc_drug_sensitivity.csv — convert XLSX→CSV (e.g. python -m pandas), then symlink or copy"
fi

# ─── CTRPv2 (NCI CTD² Public Drop) ───────────────────────────────────────────
if want CTRP; then
  section "CTRPv2 drug sensitivity (CTD² public drop)"
  CTRP_BASE="https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset"
  CTRP_TGZ="CTRPv2.0_2015_ctd2_ExpandedDataset.zip"
  fetch "$CTRP_BASE/$CTRP_TGZ" "$DATA_ROOT/ctrp/$CTRP_TGZ"
  if [[ $DRY_RUN -eq 0 && -s "$DATA_ROOT/ctrp/$CTRP_TGZ" ]]; then
    if command -v unzip >/dev/null 2>&1; then
      unzip -o -q "$DATA_ROOT/ctrp/$CTRP_TGZ" -d "$DATA_ROOT/ctrp/"
      ok "extracted CTRPv2 expanded dataset"
      # Canonical AUC table from the expanded set
      if [[ -f "$DATA_ROOT/ctrp/v20.data.curves_post_qc.txt" ]]; then
        cp "$DATA_ROOT/ctrp/v20.data.curves_post_qc.txt" "$DATA_ROOT/ctrpv2_drug_sensitivity.csv"
        ok "wrote $DATA_ROOT/ctrpv2_drug_sensitivity.csv (TSV with .csv name — fix delimiter if needed)"
      fi
    else
      warn "unzip not installed; leaving archive at $DATA_ROOT/ctrp/$CTRP_TGZ"
    fi
  fi
fi

# ─── MMRF CoMMpass — controlled access, NOT auto-fetched ─────────────────────
if want MMRF; then
  section "MMRF CoMMpass (controlled access — manual)"
  warn "MMRF requires GDC dbGaP authorization (IRB-approved protocol + GDC token)."
  warn "If you have a manifest + token, run:"
  echo  "    gdc-client download -m <manifest.txt> -t <gdc-user-token.txt> \\"
  echo  "        -d $DATA_ROOT/mmrf_commpass/"
fi

# ─── DepMap CRISPR essentiality — release-versioned, manual fetch ────────────
if want DEPMAP_CRISPR; then
  section "DepMap CRISPR gene effect (release-versioned, manual)"
  warn "CRISPRGeneEffect.csv lives on figshare with a per-release file ID"
  warn "(24Q4, 24Q2, 23Q4, ...). Pinning a stale ID here would silently fetch"
  warn "outdated essentiality scores. Visit:"
  echo  "    https://depmap.org/portal/data_page/?tab=allData"
  echo  "  -> choose the latest 'DepMap Public' release"
  echo  "  -> download CRISPRGeneEffect.csv"
  echo  "  -> place at: $DATA_ROOT/depmap/CRISPRGeneEffect.csv"
  warn "Used by similar models (DrugCell, MOLI) as gene-knockout phenotypes."
fi

# ─── HMCL Keats Lab — controlled access, MM-specific ─────────────────────────
if want HMCL; then
  section "HMCL Keats Lab MM cell-line characterization (manual)"
  warn "Keats Lab Human Myeloma Cell Line panel data is hosted on the MMRF"
  warn "researcher gateway, not curl-able. Process:"
  echo  "    1. Register at https://research.themmrf.org"
  echo  "    2. Request access to the Keats Lab HMCL Characterization dataset"
  echo  "    3. Download SNP / RNA-seq / WGS / methylation files"
  echo  "    4. Place under: $DATA_ROOT/hmcl_keats/"
  warn "MM-specific cohort -> better proteomics->IC50 mapping for MM lineages"
  warn "where CCLE has thin coverage."
fi

# ─── PRISM Repurposing Hub — release-versioned, manual fetch ─────────────────
if want PRISM; then
  section "PRISM Broad Repurposing Hub (manual)"
  warn "PRISM secondary-screen results are released via figshare with a"
  warn "per-release file ID. Visit:"
  echo  "    https://depmap.org/portal/data_page/?tab=allData"
  echo  "  -> search 'PRISM Repurposing'"
  echo  "  -> download secondary-screen-dose-response-curve-parameters.csv"
  echo  "  -> place at: $DATA_ROOT/prism/secondary-screen-dose-response-curve-parameters.csv"
  warn "PRISM provides ~4,500 compounds x ~500 cell lines (vs GDSC's 11 drugs"
  warn "x ~250 lines used here) — largest small-molecule x cell-line panel public."
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
section "Summary"
log "Staged tree:"
if command -v tree >/dev/null 2>&1; then
  tree -L 3 "$DATA_ROOT" || true
else
  find "$DATA_ROOT" -maxdepth 3 -type f -printf '  %p  (%s bytes)\n' 2>/dev/null || ls -R "$DATA_ROOT"
fi
log "Done. Verify with:  python -m resistancemap.data.validate"
