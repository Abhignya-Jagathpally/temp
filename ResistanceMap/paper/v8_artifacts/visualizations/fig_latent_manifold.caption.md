**Figure 1.** PHATE unavailable (ModuleNotFoundError); UMAP embedding of all 886 cell-line VAE latents
(`checkpoints/fusion_trained.pt:epi_states`), colored by ResistanceMap's
calibrated stability score. The 132 held-out test cell lines
are circled in red. Interpretation: do the test points sit in stable
manifold regions (well-represented neighbors), or are they in sparse / edge
regions where the model extrapolates? Edge-located test points are where
prediction risk is highest.
