#!/usr/bin/env Rscript
# Apply mmSYGNAL risk models to MMRF 787-patient cohort using published IA12 program activity.
# Inputs:
#   /tmp/our_program_activity.csv   : 787 x 141 (patient_id + programs 0..140; values in {-1,0,1})
#   /tmp/mmsygnal_*.Rds             : 6 risk models from baliga-lab/mmSYGNAL-risk-prediction-models
# Output:
#   /tmp/our_mmsygnal_per_model.csv : per-patient risk per model (NA where not applied)

suppressPackageStartupMessages({
  library(caret)
  library(readr)
})

pa <- read.csv("/tmp/our_program_activity.csv", check.names = FALSE)
rownames(pa) <- pa$patient_id
pa$patient_id <- NULL
# columns are "0".."140" character; convert to numeric matrix
prog_cols <- as.character(0:140)
stopifnot(all(prog_cols %in% colnames(pa)))
X <- pa[, prog_cols]
# caret models were trained with backticked names like `0` `1` `2`...
# the data.frame already has names "0".."140" so predict() should match coefnames
cat("X shape:", dim(X), "\n")
cat("X dtype:", class(X[1,1]), "\n")

models <- list(
  agnostic = "/tmp/mmsygnal_agnostic_risk_model.Rds",
  amp1q    = "/tmp/mmsygnal_amp1q_risk_model.Rds",
  del13    = "/tmp/mmsygnal_del13_risk_model.Rds",
  del1p    = "/tmp/mmsygnal_del1p_risk_model.Rds",
  t4_14    = "/tmp/mmsygnal_t4_14_risk_model.Rds",
  FGFR3    = "/tmp/mmsygnal_FGFR3_risk_model.Rds"
)

results <- data.frame(patient_id = rownames(X))

for (nm in names(models)) {
  cat("Running model:", nm, "\n")
  mdl <- readRDS(models[[nm]])
  prob <- predict(mdl, X, type = "prob")
  results[[nm]] <- prob$high
}

write.csv(results, "/tmp/our_mmsygnal_per_model.csv", row.names = FALSE)
cat("Wrote /tmp/our_mmsygnal_per_model.csv\n")
cat("head:\n"); print(head(results, 3))
