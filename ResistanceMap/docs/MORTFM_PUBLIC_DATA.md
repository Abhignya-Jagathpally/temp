# MORT-FM — Public-Data Assembly Plan

> The strongest defensible public-only assembly of MORT-FM. Read alongside
> `docs/MORTFM_DATA_CONTRACT.md` (the typed primitives), `docs/MORTFM_LIMITATIONS.md`
> (what claims are blocked), and `docs/MORTFM_ARCHITECTURE.md`.

## 0. The four blocks

| Block | Purpose | Sources |
|------:|---------|---------|
| **A** — cell-line foundation pretraining | Train modality encoders + drug-response + PPI propagation + drug-target conditioning + CRISPR weak supervision | DepMap/CCLE, GDSC, PRISM, STRING, UniProt, ChEMBL, Reactome, HPA |
| **B** — hematologic patient fine-tuning | Move from cell lines to primary specimens | Beat AML 1.0 (open clinical supplement + RNA + ex vivo drug response) |
| **C** — single-cell state pretraining | Malignant-cell state, resistance-state annotation, epigenomic state, pseudotime | GEO: GSE161195, GSE223060, GSE199373, GSE153380, GSE167968 |
| **D** — pathway / counterfactual weak supervision | Pathway route + drug-target recovery + CRISPR consistency | STRING + Reactome + ChEMBL + DepMap CRISPR (+ optional Perturb-seq) |

## 1. Ingestion scripts (one per source)

All ingestion scripts:
* Live under `scripts/mortfm_ingest_*.py`.
* Take a `--raw-dir` pointing at the downloaded archive and an `--out-dir`
  (defaulting to `data/processed/`).
* Raise `FileNotFoundError` with the source URL printed if expected files
  are absent.
* Emit one row to `data/processed/metadata/dataset_manifest.csv` (created if
  not present).

| Script                                       | Source / license                                        |
|----------------------------------------------|---------------------------------------------------------|
| `mortfm_ingest_depmap.py`                    | DepMap 24Q2+, CC-BY-4.0                                 |
| `mortfm_ingest_gdsc.py`                      | GDSC bulk download, free academic                        |
| `mortfm_ingest_prism.py`                     | PRISM Repurposing (DepMap), CC-BY-4.0                    |
| `mortfm_ingest_string.py`                    | STRING v12.0 (taxon 9606), CC-BY 4.0                    |
| `mortfm_ingest_uniprot.py`                   | UniProt UP000005640, CC-BY 4.0                          |
| `mortfm_ingest_reactome.py`                  | Reactome (HGNC/NCBI/UniProt2Reactome), CC-BY 4.0        |
| `mortfm_ingest_chembl.py`                    | ChEMBL drug-target export, CC-BY-SA 3.0                 |
| `mortfm_ingest_beataml.py`                   | Beat AML 1.0, open clinical + dbGaP for some molecular  |
| `mortfm_ingest_geo_singlecell.py`            | GEO accessions, varies by submitter                     |
| `mortfm_ingest_perturbation.py`              | DepMap CRISPR + optional Perturb-seq                    |
| `mortfm_build_biological_graph.py`           | Composes STRING + ChEMBL + Reactome into one graph      |

## 2. Recommended download checklist

The repo does NOT auto-download. Before running any ingestion:

```text
data/raw_public/
├── depmap/
│   ├── Model.csv
│   ├── OmicsExpressionProteinCodingGenesTPMLogp1.csv
│   ├── CRISPRGeneEffect.csv
│   └── (optional) OmicsCNGene.csv, OmicsSomaticMutations.csv
├── gdsc/
│   ├── GDSC2_fitted_dose_response_*.{csv,xlsx}
│   ├── screened_compounds_rel_*.csv
│   └── Cell_Lines_Details.xlsx
├── prism/
│   └── secondary-screen-dose-response-curve-parameters.csv
├── string/
│   ├── 9606.protein.links.full.v12.0.txt.gz
│   └── 9606.protein.aliases.v12.0.txt.gz
├── uniprot/
│   └── UP000005640_9606.fasta.gz
├── reactome/
│   └── {HGNC,NCBI,UniProt}2Reactome_All_Levels.txt
├── chembl/
│   └── chembl_drug_targets.csv     # generate via web/portal or live ChEMBL MCP
├── beataml/
│   ├── beataml_waves1to4_norm_exp_dbgap.txt
│   ├── beataml_wv1to4_clinical.{tsv,xlsx}
│   └── beataml_probit_curve_fits_v4_dbgap.txt
├── geo_single_cell/
│   ├── GSE161195/<files>
│   ├── GSE223060/<files>
│   ├── GSE199373/<files>
│   ├── GSE153380/<files>
│   └── GSE167968/<files>
└── identifier_maps/
    ├── hgnc_ensembl.tsv
    ├── uniprot_xref.tsv
    ├── string_aliases.tsv     # (extracted from string ingestion)
    ├── reactome_membership.tsv
    └── chembl_targets.tsv
```

Run order:
```bash
# Block A
python scripts/mortfm_ingest_depmap.py
python scripts/mortfm_ingest_gdsc.py
python scripts/mortfm_ingest_prism.py
python scripts/mortfm_ingest_string.py
python scripts/mortfm_ingest_uniprot.py
python scripts/mortfm_ingest_reactome.py
python scripts/mortfm_ingest_chembl.py

# Block D (depends on A)
python scripts/mortfm_build_biological_graph.py

# Block B
python scripts/mortfm_ingest_beataml.py

# Block C
python scripts/mortfm_ingest_geo_singlecell.py

# Block D (perturbation -- depends on DepMap CRISPR from A)
python scripts/mortfm_ingest_perturbation.py --crispr data/processed/perturbation/crispr_gene_effect.parquet
```

After ingestion you should have a populated `data/processed/metadata/` directory.
The trainer scripts (`mortfm_pretrain_foundation.py`, `mortfm_train_survival.py`,
`mortfm_train_trajectory.py`) consume these processed files.

## 3. Acceptance thresholds (enforced by code)

Located at `resistancemap.mortfm.acceptance_gate.evaluate_cohort`, called by
`MORTFMTrainer.fit_stage(..., acceptance_level=...)`.

| Level         | n_patients | n_events | n_modalities | Per-patient timepoints | Operational |
|---------------|-----------:|---------:|-------------:|------------------------|-------------|
| `debug`       | ≥ 10       | —        | ≥ 1          | —                      | none        |
| `technical`   | ≥ 100 (samples) | —   | ≥ 1          | —                      | patient-disjoint split |
| `research`    | ≥ 200      | ≥ 50     | ≥ 2          | ≥ 2 (if trajectory)    | baseline comparison required |
| `strong`      | ≥ 200      | ≥ 50     | ≥ 2          | ≥ 2 (if trajectory)    | external cohort + ablations + pathway validation + calibration + claim-critic report |

The `strong` level **never auto-passes** — it requires explicit operational
evidence that the gate cannot infer. This is the mechanism that prevents an
underpowered cohort from accidentally producing a "patient-level
time-to-resistance" claim.

## 4. What you can claim from a public-only run

| Claim                                                | Public-only feasibility |
|------------------------------------------------------|-------------------------|
| Drug-specific risk on cell lines                     | Strong (DepMap + GDSC/PRISM) |
| Static resistance/sensitivity                        | Strong (GDSC + PRISM + Beat AML) |
| PPI / pathway route hypothesis                       | Moderate (STRING + Reactome + ChEMBL) |
| Cell-line counterfactual ranking vs CRISPR oracle    | Moderate (DepMap CRISPR) |
| AML patient drug response                            | Moderate (Beat AML) |
| Multiple-myeloma patient single-cell state           | Moderate (GEO MM datasets) |
| MM patient time-to-resistance                        | Weak with open-only data; MMRF requires Synapse/dbGaP for some molecular |
| Trajectory over real calendar time                   | Weak — most public MM scRNA is single-timepoint |
| Strong-level "patient-level resistance prediction"   | Blocked by acceptance gate; needs ≥200 patients, ≥50 events, ≥2 timepoints, external cohort |

## 5. Identifier harmonisation

`resistancemap/data/identifier_mapping.py` is the single source of truth for
mapping across HGNC ⇄ Ensembl ⇄ UniProt ⇄ STRING ⇄ ChEMBL ⇄ Reactome.

All ingestion scripts write IDs in their native namespace (DepMap uses HGNC,
STRING uses 9606.ENSP, UniProt uses accessions, ChEMBL uses CHEMBL IDs).
The graph builder + foundation model use `IdentifierMaps.harmonise_gene_symbol`
to project everything onto canonical HGNC symbols.

`unify_identifiers([...])` returns `None` for any identifier it cannot map.
Callers MUST handle the `None` — never fall back silently to the input string.

## 6. Hard requirements before any biology claim

(From the user's data-spec section 7, now enforced in code.)

| Requirement | Enforcement |
|-------------|-------------|
| Stable biological identifiers | `identifier_mapping.IdentifierMaps` + `feature_manifest` validation |
| Patient/model-disjoint splits | `samples.csv` validator + `data_module.split_patients` test |
| Explicit modality mask | `MORTBatch.modality_mask` + `samples.csv` `modality_available` column |
| Clear time semantics | `PatientCellSnapshot.timepoint` + `treatment_exposure.py` |
| Outcome censoring | `ResistanceOutcome.event_observed` + survival loss validation |
| Drug normalization | `drug_response_long.parquet` preserves `response_metric` per row |
| Batch / domain labels | `ModalityTensor.batch_id` + `samples.csv` `batch_id` |
