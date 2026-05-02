**Figure 2.** Per-drug predicted vs ground-truth drug-sensitivity on the test
split, colored by ResistanceMap's stability pseudotime. The Ridge regression
is from VAE latents (a clean head, isolating the latent's predictive content
from the rest of the architecture). Points off the diagonal are model errors;
color shows whether errors concentrate in any pseudotime regime. Panobinostat
and Doxorubicin (high MSE in `per_drug_metrics.csv`) are expected to show the
widest spread.
