# RQ10 staging — unblock the scRNA cross-scale arm (today, open data)

RQ10 was BLOCKED only because no scRNA counts were on disk. It is **not** access-gated
(unlike RQ7/Gateway). These four steps stage it with the `GEOparse` already installed.

## Pick a source (verified open)
- **GSE223060** (scRNA) / **GSE223061** (bulk) — concrete, immediately downloadable.
- **Nature Cancer atlas s43018-025-01072-4** — CoMMpass-linked (337 newly-dx patients) →
  best for a true patient-level cell↔bulk join; pull its deposited matrices.

## Steps
```bash
# 1. download supplementary 10x files
python scripts/fetch_geo_scrna.py --acc GSE223060 --out data/raw/scrna/GSE223060
#    (unpack any .tar.gz to per-sample 10x dirs)

# 2. Seurat -> .h5ad  (run on your box; has Seurat R)
Rscript scripts/geo_to_h5ad.R --in data/raw/scrna/GSE223060/<sample_10x_dir> \
                              --out data/raw/scrna/GSE223060.h5ad

# 3. Python: build cell-state manifold + per-patient features
python - <<'PY'
from resistancemap.data import scrna_atlas as S
ad = S.normalize(S.qc_filter(S.load_h5ad("data/raw/scrna/GSE223060.h5ad")))
ad = S.cellstate_manifold(ad, use_scvi=True)          # batch-corrected latent across patients
pids, feats, names = S.patient_progression_features(ad, patient_key="patient_id")
# -> attach_to_bulk(bulk_phi_df, pids, feats, names)  joins onto bulk on patient_id
PY

# 4. ablate the lift (RQ10 gate)
make cross_scale
```

## Linkage note (the high-value bit)
The Nature atlas shares `patient_id` with CoMMpass → join the per-patient cell-state
features directly onto the bulk φ (no transfer needed). For GSE223060, fill
`patient_id_TODO` in the `sample_metadata.csv` the fetcher writes (from the GEO series
matrix sample titles) before the join. Missing patients → NaN → train-fold imputer +
missingness mask (never zero-impute). If a download is unavailable, keep
`BLOCKED_SCRNA_RQ10.md`; do not synthesize single cells.