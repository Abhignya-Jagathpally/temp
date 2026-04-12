"""STRING PPI debiasing, CRISPR validation, and phosphoproteomics augmentation.

This module addresses text-mining circularity in STRING PPIs and provides
validation against experimental data (DepMap CRISPR) and enrichment of protein
node features with phosphoproteomics.

Agent 5 concerns:
- STRING PPI edges may suffer from text-mining circularity (publication bias)
- Model attributions should be validated against CRISPR dependency data
- Phosphoproteomics can augment protein-level features for improved GNN signal
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import sparse
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

__all__ = [
    "STRINGDebiaser",
    "CRISPRValidator",
    "PhosphoproteomicsAugmenter",
]

logger = logging.getLogger(__name__)


class STRINGDebiaser:
    """Filters STRING PPI edges to remove text-mined circularity.

    Computes a debiased combined score by excluding textmining evidence,
    then filters edges below a confidence threshold and identifies genes
    with high publication bias risk.

    Attributes
    ----------
    confidence_threshold : float
        Minimum combined score (0-1) for retained edges after debiasing.
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """Initialize STRINGDebiaser.

        Parameters
        ----------
        confidence_threshold : float, default=0.7
            Minimum combined score for edge retention.
        """
        if not HAS_PANDAS:
            logger.warning("pandas not available; STRINGDebiaser will not function properly")

        self.confidence_threshold = confidence_threshold
        logger.info(f"STRINGDebiaser initialized with threshold={confidence_threshold}")

    def filter_edges(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Filter STRING edges to remove text-mining circularity.

        Takes a DataFrame of STRING edges and computes a debiased combined score
        by excluding textmining evidence. Retains only edges above the confidence
        threshold.

        Parameters
        ----------
        edges_df : pd.DataFrame
            DataFrame with columns: [protein1, protein2, combined_score, experimental,
            database, coexpression, textmining, ...]. Values should be in [0, 1].

        Returns
        -------
        pd.DataFrame
            Filtered edges with added 'debiased_combined_score' column.

        Raises
        ------
        ValueError
            If required columns are missing or scores are outside [0, 1].
        """
        required_cols = {'protein1', 'protein2', 'combined_score', 'textmining'}
        if not required_cols.issubset(edges_df.columns):
            raise ValueError(
                f"edges_df missing required columns. Need: {required_cols}, "
                f"got: {set(edges_df.columns)}"
            )

        # Validate score ranges
        score_cols = ['combined_score', 'textmining']
        for col in score_cols:
            if (edges_df[col] < 0).any() or (edges_df[col] > 1).any():
                raise ValueError(f"Column {col} contains values outside [0, 1]")

        # Create a working copy
        df = edges_df.copy()

        # Compute debiased score: remove text-mining contribution
        # Approximate debiasing: combined_score_debiased ≈ combined_score - textmining weight
        # A more rigorous approach would back out textmining from the combined score calculation,
        # but we use the simpler heuristic: zero out textmining contribution
        df['debiased_combined_score'] = df['combined_score'] - (df['textmining'] * 0.5)
        df['debiased_combined_score'] = df['debiased_combined_score'].clip(lower=0, upper=1)

        # Filter by threshold
        filtered = df[df['debiased_combined_score'] >= self.confidence_threshold].copy()

        logger.info(
            f"Filtered {len(edges_df)} edges to {len(filtered)} "
            f"(removed {len(edges_df) - len(filtered)} edges below threshold)"
        )

        return filtered

    def gene_level_bias_report(self, edges_df: pd.DataFrame) -> dict[str, Any]:
        """Per-gene analysis of text-mining dependency.

        Identifies genes where >50% of edge weight comes from text-mining evidence,
        flagging them as having "high publication bias risk".

        Parameters
        ----------
        edges_df : pd.DataFrame
            DataFrame with columns: [protein1, protein2, combined_score, textmining, ...].

        Returns
        -------
        dict
            Report with keys:
            - 'high_bias_risk_genes': list of gene symbols with >50% textmining weight
            - 'bias_scores': dict mapping gene -> fraction of weight from textmining
            - 'summary': str summarizing findings
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for gene_level_bias_report")

        if 'textmining' not in edges_df.columns:
            raise ValueError("edges_df missing 'textmining' column")

        # Aggregate textmining weight per gene
        bias_by_gene = {}

        for _, row in edges_df.iterrows():
            p1, p2 = row['protein1'], row['protein2']
            txt_score = row.get('textmining', 0.0)
            combined = row.get('combined_score', 1.0)

            # Proportion of combined score from textmining
            if combined > 0:
                txt_fraction = txt_score / combined
            else:
                txt_fraction = 0.0

            for gene in [p1, p2]:
                if gene not in bias_by_gene:
                    bias_by_gene[gene] = []
                bias_by_gene[gene].append(txt_fraction)

        # Compute mean textmining fraction per gene
        bias_scores = {gene: np.mean(fractions) for gene, fractions in bias_by_gene.items()}

        # Flag high-risk genes
        high_risk = [gene for gene, score in bias_scores.items() if score > 0.5]
        high_risk.sort()

        summary = (
            f"Found {len(high_risk)} genes with >50% textmining dependency. "
            f"Mean textmining fraction across all genes: {np.mean(list(bias_scores.values())):.3f}"
        )

        logger.info(summary)

        return {
            'high_bias_risk_genes': high_risk,
            'bias_scores': bias_scores,
            'summary': summary,
        }

    def debiased_adjacency(
        self,
        edges_df: pd.DataFrame,
        protein_list: list[str],
    ) -> Any:
        """Return debiased adjacency matrix as sparse tensor.

        Parameters
        ----------
        edges_df : pd.DataFrame
            Filtered edges with 'protein1', 'protein2', 'debiased_combined_score' columns.
        protein_list : list[str]
            Ordered list of protein identifiers (for indexing).

        Returns
        -------
        sparse matrix (scipy.sparse.csr_matrix or similar)
            Sparse adjacency matrix with debiased combined scores as edge weights.

        Raises
        ------
        RuntimeError
            If scipy not available.
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy required for debiased_adjacency")

        if 'debiased_combined_score' not in edges_df.columns:
            raise ValueError("edges_df missing 'debiased_combined_score' column")

        # Create protein -> index mapping
        protein_to_idx = {p: i for i, p in enumerate(protein_list)}

        # Extract edges that map to known proteins
        valid_edges = []
        for _, row in edges_df.iterrows():
            p1, p2 = row['protein1'], row['protein2']
            if p1 in protein_to_idx and p2 in protein_to_idx:
                valid_edges.append(row)

        if not valid_edges:
            logger.warning("No valid edges found in protein_list")
            return sparse.csr_matrix((len(protein_list), len(protein_list)))

        valid_edges_df = pd.DataFrame(valid_edges)

        # Build sparse matrix
        row_idx = [protein_to_idx[p] for p in valid_edges_df['protein1']]
        col_idx = [protein_to_idx[p] for p in valid_edges_df['protein2']]
        data = valid_edges_df['debiased_combined_score'].values

        adj = sparse.csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(protein_list), len(protein_list))
        )

        # Make symmetric
        adj = adj + adj.T

        logger.info(f"Built debiased adjacency matrix: {adj.shape}, {adj.nnz} edges")

        return adj


class CRISPRValidator:
    """Validates model-attributed genes against DepMap CRISPR dependency.

    Checks whether genes attributed by the model to resistance/phenotype
    show CRISPR dependency (essentiality) in DepMap data, supporting
    the attribution's biological plausibility.

    Attributes
    ----------
    depmap_path : Path or None
        Path to DepMap gene effect scores file.
    depmap_data : dict or None
        Loaded DepMap data, keys: cell_line_id, values: dicts of {gene: effect_score}.
    """

    def __init__(self, depmap_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize CRISPRValidator.

        Parameters
        ----------
        depmap_path : str, Path, or None
            Path to DepMap gene effect scores (e.g., CSV with columns
            [cell_line_id, gene, gene_effect]). If None, validation
            will require pre-loaded data.
        """
        self.depmap_path = Path(depmap_path) if depmap_path else None
        self.depmap_data = None

        if self.depmap_path and self.depmap_path.exists():
            self._load_depmap()
            logger.info(f"CRISPRValidator initialized with DepMap data from {self.depmap_path}")
        else:
            logger.info("CRISPRValidator initialized without DepMap data")

    def _load_depmap(self) -> None:
        """Load DepMap gene effect scores from file."""
        if not HAS_PANDAS:
            raise RuntimeError("pandas required to load DepMap data")

        try:
            df = pd.read_csv(self.depmap_path)

            # Organize by cell line
            self.depmap_data = {}
            for cell_line in df['cell_line_id'].unique():
                cell_df = df[df['cell_line_id'] == cell_line]
                self.depmap_data[cell_line] = dict(zip(cell_df['gene'], cell_df['gene_effect']))

            logger.info(f"Loaded DepMap data for {len(self.depmap_data)} cell lines")
        except Exception as e:
            logger.error(f"Failed to load DepMap data: {e}")
            raise

    def validate_attributions(
        self,
        attributed_genes: list[str],
        cell_line_ids: list[str],
        threshold: float = -0.5,
    ) -> dict[str, Any]:
        """Validate attributed genes against CRISPR dependency.

        Checks if attributed genes show CRISPR dependency (negative gene effect)
        in the specified cell lines.

        Parameters
        ----------
        attributed_genes : list[str]
            Gene symbols attributed by the model.
        cell_line_ids : list[str]
            DepMap cell line identifiers to validate against.
        threshold : float, default=-0.5
            Gene effect threshold below which gene is considered essential.
            (negative = essential, positive = not essential)

        Returns
        -------
        dict
            Report with keys:
            - 'validated_genes': genes with CRISPR dependency
            - 'unvalidated_genes': genes without CRISPR dependency
            - 'missing_genes': genes not in DepMap
            - 'validation_rates': dict {cell_line_id -> validation rate}
            - 'summary': str summary
        """
        if not self.depmap_data:
            raise RuntimeError("No DepMap data loaded")

        validated = set()
        unvalidated = set()
        missing = set()
        validation_rates = {}

        for cell_line in cell_line_ids:
            if cell_line not in self.depmap_data:
                logger.warning(f"Cell line {cell_line} not in DepMap data")
                continue

            cell_effects = self.depmap_data[cell_line]
            cell_validated = 0

            for gene in attributed_genes:
                if gene not in cell_effects:
                    missing.add(gene)
                    continue

                effect = cell_effects[gene]
                if effect < threshold:
                    validated.add(gene)
                    cell_validated += 1
                else:
                    unvalidated.add(gene)

            rate = cell_validated / len(attributed_genes) if attributed_genes else 0.0
            validation_rates[cell_line] = rate
            logger.info(f"Cell line {cell_line}: {rate:.2%} genes validated")

        summary = (
            f"Validated {len(validated)} / {len(attributed_genes)} attributed genes. "
            f"Mean validation rate: {np.mean(list(validation_rates.values())):.2%}"
        )

        return {
            'validated_genes': list(validated),
            'unvalidated_genes': list(unvalidated),
            'missing_genes': list(missing),
            'validation_rates': validation_rates,
            'summary': summary,
        }

    def compute_precision_at_k(
        self,
        attributed_genes: list[str],
        crispr_hits: set[str],
        k_values: list[int] = None,
    ) -> dict[int, float]:
        """Compute precision@k for attributed genes vs CRISPR hits.

        Parameters
        ----------
        attributed_genes : list[str]
            Ranked list of attributed genes.
        crispr_hits : set[str]
            Set of genes with CRISPR dependency.
        k_values : list[int], optional
            k values to compute precision for (default: [10, 25, 50]).

        Returns
        -------
        dict[int, float]
            Precision@k for each k in k_values.
        """
        if k_values is None:
            k_values = [10, 25, 50]

        precision = {}
        for k in k_values:
            top_k = set(attributed_genes[:k])
            hits = len(top_k & crispr_hits)
            precision[k] = hits / k if k > 0 else 0.0

        logger.info(f"Precision@k: {precision}")
        return precision

    def enrichment_test(
        self,
        attributed_genes: set[str],
        crispr_hits: set[str],
        background_size: int,
    ) -> dict[str, Any]:
        """Perform Fisher's exact test for enrichment.

        Tests whether attributed genes are significantly enriched for CRISPR
        dependency vs. background genes.

        Parameters
        ----------
        attributed_genes : set[str]
            Set of attributed genes.
        crispr_hits : set[str]
            Set of genes with CRISPR dependency.
        background_size : int
            Total size of background gene set.

        Returns
        -------
        dict
            Report with keys:
            - 'contingency_table': 2x2 table for Fisher's exact test
            - 'oddsratio': odds ratio
            - 'pvalue': p-value from Fisher's exact test
            - 'significant': bool (p < 0.05)
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy required for enrichment_test")

        # 2x2 contingency table: [in_attributed & in_crispr, in_attributed & not_in_crispr,
        #                          not_in_attributed & in_crispr, not_in_attributed & not_in_crispr]
        both = len(attributed_genes & crispr_hits)
        attr_only = len(attributed_genes - crispr_hits)
        crispr_only = len(crispr_hits - attributed_genes)
        neither = background_size - len(attributed_genes | crispr_hits)

        contingency = [[both, attr_only], [crispr_only, neither]]

        oddsratio, pvalue = fisher_exact(contingency, alternative='greater')

        result = {
            'contingency_table': contingency,
            'oddsratio': oddsratio,
            'pvalue': pvalue,
            'significant': pvalue < 0.05,
        }

        logger.info(f"Fisher's exact test p-value: {pvalue:.4e}, significant: {result['significant']}")
        return result


class PhosphoproteomicsAugmenter:
    """Augments protein node features with phosphoproteomics data.

    Integrates phosphorylation site-level information into protein-level features
    for improved GNN signal in resistance prediction tasks.

    Attributes
    ----------
    phospho_sites_path : Path or None
        Path to phosphoproteomics data file.
    phospho_data : dict or None
        Loaded phosphoproteomics data.
    """

    def __init__(self, phospho_sites_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize PhosphoproteomicsAugmenter.

        Parameters
        ----------
        phospho_sites_path : str, Path, or None
            Path to phosphoproteomics data (e.g., CSV with columns
            [protein, phosphosite, abundance, cell_line, ...]).
            If None, augmentation requires pre-loaded data.
        """
        self.phospho_sites_path = Path(phospho_sites_path) if phospho_sites_path else None
        self.phospho_data = None

        if self.phospho_sites_path and self.phospho_sites_path.exists():
            self._load_phosphoproteomics()
            logger.info(f"PhosphoproteomicsAugmenter initialized from {self.phospho_sites_path}")
        else:
            logger.info("PhosphoproteomicsAugmenter initialized without phosphoproteomics data")

    def _load_phosphoproteomics(self) -> None:
        """Load phosphoproteomics data from file."""
        if not HAS_PANDAS:
            raise RuntimeError("pandas required to load phosphoproteomics data")

        try:
            df = pd.read_csv(self.phospho_sites_path)
            self.phospho_data = df
            logger.info(f"Loaded phosphoproteomics data: {len(df)} phosphosites")
        except Exception as e:
            logger.error(f"Failed to load phosphoproteomics data: {e}")
            raise

    def site_to_protein_mapping(
        self,
        phospho_df: pd.DataFrame = None,
    ) -> dict[str, list[str]]:
        """Create a mapping from phosphosites to parent proteins.

        Parameters
        ----------
        phospho_df : pd.DataFrame, optional
            Phosphoproteomics DataFrame. If None, uses self.phospho_data.

        Returns
        -------
        dict[str, list[str]]
            Maps each phosphosite to its parent protein(s).
        """
        if phospho_df is None:
            if self.phospho_data is None:
                raise RuntimeError("No phosphoproteomics data loaded")
            phospho_df = self.phospho_data

        mapping = {}
        for _, row in phospho_df.iterrows():
            site = row.get('phosphosite', None)
            protein = row.get('protein', None)

            if site and protein:
                if site not in mapping:
                    mapping[site] = []
                if protein not in mapping[site]:
                    mapping[site].append(protein)

        logger.info(f"Built mapping for {len(mapping)} phosphosites")
        return mapping

    def augment_features(
        self,
        protein_features: np.ndarray,
        protein_names: list[str],
        phospho_df: pd.DataFrame = None,
        aggregation: str = 'mean',
    ) -> np.ndarray:
        """Augment protein features with phosphoproteomics data.

        Concatenates phospho-site level features to protein node features,
        aggregating across sites per protein.

        Parameters
        ----------
        protein_features : np.ndarray
            Shape (n_proteins, n_features) original protein features.
        protein_names : list[str]
            Protein identifiers corresponding to rows of protein_features.
        phospho_df : pd.DataFrame, optional
            Phosphoproteomics DataFrame. If None, uses self.phospho_data.
        aggregation : {'mean', 'max', 'sum'}, default='mean'
            How to aggregate phospho features across sites per protein.

        Returns
        -------
        np.ndarray
            Augmented features of shape (n_proteins, n_features + n_phospho_features).
        """
        if phospho_df is None:
            if self.phospho_data is None:
                raise RuntimeError("No phosphoproteomics data loaded")
            phospho_df = self.phospho_data

        if not HAS_PANDAS:
            raise RuntimeError("pandas required for augment_features")

        # Build protein -> phosphosite mapping
        protein_to_sites = {}
        for _, row in phospho_df.iterrows():
            protein = row.get('protein', None)
            if protein:
                if protein not in protein_to_sites:
                    protein_to_sites[protein] = []
                protein_to_sites[protein].append(row)

        # Extract feature columns (non-metadata)
        feature_cols = [c for c in phospho_df.columns
                       if c not in {'protein', 'phosphosite', 'cell_line'}]

        if not feature_cols:
            logger.warning("No feature columns found in phosphoproteomics data")
            return protein_features

        # Build augmented features
        augmented_list = []
        for protein in protein_names:
            if protein in protein_to_sites:
                sites = protein_to_sites[protein]
                site_features = np.array([
                    [s.get(col, 0.0) for col in feature_cols]
                    for s in sites
                ])

                # Aggregate
                if aggregation == 'mean':
                    agg_features = site_features.mean(axis=0)
                elif aggregation == 'max':
                    agg_features = site_features.max(axis=0)
                elif aggregation == 'sum':
                    agg_features = site_features.sum(axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")
            else:
                # No phospho data for this protein; use zeros
                agg_features = np.zeros(len(feature_cols))

            augmented_list.append(agg_features)

        phospho_features = np.array(augmented_list)

        # Concatenate
        augmented = np.hstack([protein_features, phospho_features])

        logger.info(
            f"Augmented features from {protein_features.shape[1]} to {augmented.shape[1]} dims"
        )

        return augmented
