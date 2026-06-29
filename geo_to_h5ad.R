#!/usr/bin/env Rscript
# geo_to_h5ad.R -- Seurat -> SeuratDisk -> .h5ad for the RQ10 cross-scale loader.
# Run on your box (has Seurat R). scanpy/anndata read the output .h5ad.
#
#   Rscript scripts/geo_to_h5ad.R --in data/raw/scrna/GSE223060/SAMPLE_10x_dir \
#                                 --out data/raw/scrna/GSE223060.h5ad
#
# Input = a 10x directory (barcodes.tsv.gz, features/genes.tsv.gz, matrix.mtx.gz)
# OR a 10x .h5. Counts are preserved so the Python side can renormalize.

suppressPackageStartupMessages({
  library(Seurat); library(SeuratDisk); library(Matrix); library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--in",  dest = "input",  type = "character"),
  make_option("--out", dest = "output", type = "character"),
  make_option("--min_cells",    default = 3),
  make_option("--min_features", default = 200),
  make_option("--max_mt",       default = 20.0),
  make_option("--patient_col",  default = NA, help = "metadata col to rename -> patient_id")
)))

message("[R] reading ", opt$input)
counts <- if (grepl("\\.h5$", opt$input)) Read10X_h5(opt$input) else Read10X(data.dir = opt$input)
obj <- CreateSeuratObject(counts = counts, min.cells = opt$min_cells,
                          min.features = opt$min_features)
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
obj <- subset(obj, subset = nFeature_RNA > opt$min_features & percent.mt < opt$max_mt)
obj <- NormalizeData(obj, verbose = FALSE)               # log1p; counts kept in @counts
obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
if (!is.na(opt$patient_col) && opt$patient_col %in% colnames(obj@meta.data)) {
  obj$patient_id <- obj@meta.data[[opt$patient_col]]     # for bulk<->cell linkage
}
message("[R] cells=", ncol(obj), " genes=", nrow(obj))

tmp <- sub("\\.h5ad$", ".h5Seurat", opt$output)
SaveH5Seurat(obj, filename = tmp, overwrite = TRUE)
Convert(tmp, dest = "h5ad", overwrite = TRUE)
message("[R] wrote ", opt$output, "  -> load with resistancemap.data.scrna_atlas.load_h5ad()")