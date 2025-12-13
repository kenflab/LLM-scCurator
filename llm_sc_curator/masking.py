
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import re
import logging
from .noise_lists import NOISE_PATTERNS, NOISE_LISTS

# Configure logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Rescue policy
# -------------------------------------------------------------------------
# We only rescue:
#   - LINC_Noise: keep a single top-expressed LINC as a tumor/lineage sentinel
#   - Hemo_Contam: keep a single Hb gene so true erythrocyte clusters remain
#                  recognizable, while redundant Hb copies are removed.
RESCUE_MODULES_DEFAULT = ("LINC_Noise", "Hemo_Contam")

# Pre-compile regex patterns once for efficiency
_COMPILED_PATTERNS = {
    name: re.compile(pat) for name, pat in NOISE_PATTERNS.items()
}

class FeatureDistiller:
    def __init__(self, global_adata):
        """
        Initialize with the GLOBAL dataset to establish background context.
        Args:
            global_adata: The full AnnData object containing all cells/lineages.
                          Expected to be log-normalized (log1p).
        """
        self.adata = global_adata
        self.gene_stats = None

    def calculate_gene_stats(self):
        """
        Calculates global Gini coefficient and mean expression for each gene.

        Notes
        -----
        - For sparse matrices (typically CSR in Scanpy), column-wise slicing
          (X[:, i]) inside a Python loop is inefficient.
          We therefore convert once to CSC (column-oriented) and extract
          columns from that representation.
        - This function can still be computationally heavy on very large
          atlases (e.g. > 2e5 cells × > 1e4 genes). In such cases, we
          recommend setting a balanced global context via `set_global_context`
          to limit the number of cells used for global statistics.
        """
        X = self.adata.X

        # --- 1. Mean expression (Dense/Sparse agnostic) ---
        if issparse(X):
            mean_expr = np.asarray(X.mean(axis=0)).ravel()
        else:
            mean_expr = np.mean(X, axis=0)

        # --- 2. Gini helper function ---
        def gini(array_1d: np.ndarray) -> float:
            """
            Calculate Gini coefficient of a 1D numpy array.
            """
            array = np.asarray(array_1d).ravel().astype(float)

            # Shift to non-negative if necessary (e.g. if scaled data is input)
            if np.any(array < 0):
                array = array - np.min(array)

            # Add small epsilon to avoid division by zero for all-zero genes
            array = array + 1e-7

            array = np.sort(array)
            n = array.shape[0]
            index = np.arange(1, n + 1, dtype=float)

            return float(np.sum((2 * index - n - 1) * array) / (n * np.sum(array)))

        # --- 3. Gini per gene (Sparse Optimized) ---
        n_cells, n_genes = X.shape
        ginis = np.zeros(n_genes, dtype=float)

        if issparse(X):
            # Convert to CSC once for efficient column access
            X_csc = X.tocsc()

            # Soft warning for very large contexts
            if n_cells > 200_000 and n_genes > 10_000:
                logger.warning(
                    "[Gini] Very large global context detected "
                    "(cells=%d, genes=%d). "
                    "Consider using `set_global_context(..., balance_by=...)` "
                    "to downsample before calling `detect_biological_noise`.",
                    n_cells,
                    n_genes,
                )

            for i in range(n_genes):
                col_dense = X_csc[:, i].toarray().ravel()
                ginis[i] = gini(col_dense)
        else:
            for i in range(n_genes):
                col_dense = np.asarray(X[:, i]).ravel()
                ginis[i] = gini(col_dense)

        # --- 4. Store Results ---
        self.gene_stats = (
            pd.DataFrame(
                {"gene": self.adata.var_names, "mean": mean_expr, "gini": ginis}
            )
            .set_index("gene")
        )

        return self.gene_stats

    def detect_biological_noise(
        self,
        gini_threshold: float | None = None,
        gini_q: float = 0.01,
        mean_floor: float = 0.05,
        whitelist=None,
        rescue_mean_floor: float = 0.1,
        low_gini_cap: float | None = 0.15,
        rescue_modules=None,
    ):
        """
        Stage 2: Detects noise genes using global statistics and regex patterns.

        Global low-specificity genes are defined using either:
        - an absolute Gini threshold (gini_threshold), if provided, or
        - the lower gini_q quantile among genes with mean >= mean_floor,
          optionally capped by `low_gini_cap` (e.g., 0.15).

        Args:
            gini_threshold: If not None, use this absolute Gini cutoff.
            gini_q: Quantile for low-Gini band (e.g., 0.01). If None, quantile
                    based cutoff is disabled.
            mean_floor: Exclude genes with very low mean expression when
                        estimating the Gini quantile.
            low_gini_cap: Optional absolute upper bound on the cutoff; the
                          effective cutoff is min(q-cutoff, low_gini_cap).
        """
        if self.gene_stats is None:
            self.calculate_gene_stats()

        if whitelist is None:
            whitelist = []

        # If not provided, use the conservative default set
        if rescue_modules is None:
            rescue_modules = RESCUE_MODULES_DEFAULT

        gs = self.gene_stats.copy()
        gs = gs.replace([np.inf, -np.inf], np.nan).dropna(subset=["gini", "mean"])

        # --- 1. Determine Gini cutoff (absolute or percentile) ---
        mask_for_cut = gs["mean"] >= mean_floor
        if gini_threshold is not None:
            low_gini_cut = float(gini_threshold)
        else:
            if mask_for_cut.sum() == 0:
                low_gini_cut = -1.0
                logger.warning(
                    "No genes passed mean_floor filtering when computing Gini quantile; "
                    "housekeeping filtering via Gini is effectively disabled."
                )
            else:
                low_gini_cut = float(
                    np.quantile(gs.loc[mask_for_cut, "gini"].values, gini_q)
                )

        if low_gini_cut > 0 and low_gini_cap is not None:
            low_gini_cut = min(low_gini_cut, float(low_gini_cap))

        mask_reasons: dict[str, str] = {}

        # --- 2. Global Statistics (Low Gini) ---
        if low_gini_cut > 0:
            low_gini_genes = gs[
                mask_for_cut & (gs["gini"] < low_gini_cut)
            ].index
        else:
            low_gini_genes = []

        for g in low_gini_genes:
            mask_reasons[g] = (
                f"Low_Gini_Housekeeping(q={gini_q:.2f}, cutoff={low_gini_cut:.3f})"
            )

        # --- 3. Biological Modules (Regex with Rescue) ---
        for module_name, pattern in NOISE_PATTERNS.items():
            regex = _COMPILED_PATTERNS.get(module_name)
            if regex is None:
                continue

            matched = [g for g in self.adata.var_names if regex.match(g)]
            if not matched:
                continue

            if module_name in rescue_modules:
                # Top-1 Rescue Logic with minimum mean-expression safeguard
                matched_stats = self.gene_stats.loc[matched].sort_values(
                    by='mean', ascending=False
                )
                top_gene = matched_stats.index[0]
                top_mean = float(matched_stats["mean"].iloc[0])

                if top_mean < rescue_mean_floor:
                    # If even the strongest clone is barely expressed, mask all
                    for g in matched_stats.index:
                        mask_reasons[g] = f"Module_{module_name} (AllLowExpr)"
                else:
                    # Keep top 1 as sentinel, mask the rest
                    for g in matched_stats.index[1:]:
                        mask_reasons[g] = f"Module_{module_name} (Redundant)"

                    # Ensure the rescued top gene is not masked elsewhere
                    if top_gene in mask_reasons:
                        del mask_reasons[top_gene]
            else:
                # Mask all for non-rescue modules (Mito, Stress, etc.)
                for g in matched:
                    mask_reasons[g] = f"Module_{module_name}"

        # --- 4. Explicit Gene Lists (Cell Cycle etc.) ---
        for module_name, gene_set in NOISE_LISTS.items():
            matched = [g for g in self.adata.var_names if g in gene_set]
            for g in matched:
                mask_reasons[g] = f"Module_{module_name}"

        # --- 5. User Whitelist Rescue ---
        for gene in whitelist:
            if gene in mask_reasons:
                del mask_reasons[gene]

        return mask_reasons

    def calculate_lineage_specificity(
        self,
        target_genes,
        target_adata,
        target_group,
        group_col,
        coarse_col,
        expr_percentile=90,
        tail_percentile=95,
        abs_threshold=0.1,
    ):
        """
        Stage 3: Cross-Lineage Specificity Check (Data-Driven).

        Compares expression in the local target subset against global major lineages.
        Uses percentile-based dynamic thresholding to identify leakage.
        """
        # Safety checks
        if coarse_col not in self.adata.obs:
            logger.warning(
                "Stage 3 Skipped: '%s' not found in global metadata.", coarse_col
            )
            return {}

        # 1. Identify Target Expression (Local Context)
        if group_col not in target_adata.obs:
            logger.warning(
                "Stage 3 Skipped: '%s' not found in target metadata.", group_col
            )
            return {}

        target_cells = target_adata[target_adata.obs[group_col] == target_group]
        if target_cells.n_obs == 0:
            return {}

        if coarse_col not in target_cells.obs:
            logger.warning(
                "Stage 3 Skipped: '%s' missing in target subset.", coarse_col
            )
            return {}

        my_coarse_label = target_cells.obs[coarse_col].mode()[0]

        # Helper for Percentiles (Sparse optimized; multi-gene)
        def get_percentiles(adata_obj, genes, pct):
            # Restrict to genes that exist in this object
            var_names = adata_obj.var_names
            existing = [g for g in genes if g in var_names]

            # Initialize all to 0.0 by default
            out = {g: 0.0 for g in genes}
            if not existing:
                return out

            idx = [var_names.get_loc(g) for g in existing]
            X_sub = adata_obj.X[:, idx]

            if issparse(X_sub):
                X_sub = X_sub.toarray()

            # Percentile per gene (axis=0)
            p_vals = np.percentile(X_sub, pct, axis=0)

            for g, v in zip(existing, p_vals):
                out[g] = float(v)

            return out

        # 2. Target Percentiles (Local)
        target_pcts = get_percentiles(target_cells, target_genes, expr_percentile)

        # 3. Iterate over OTHER Lineages (Global Context)
        unique_coarse = self.adata.obs[coarse_col].unique()
        leakage_mask = {}

        for cl in unique_coarse:
            if cl == my_coarse_label:
                continue  # Skip self

            other_cells = self.adata[self.adata.obs[coarse_col] == cl]
            if other_cells.n_obs < 10:
                continue

            other_pcts = get_percentiles(other_cells, target_genes, expr_percentile)

            # --- Dynamic Thresholding ---
            ratios = []
            eps = 1e-6

            for g in target_genes:
                p_my = target_pcts.get(g, 0.0)
                p_other = other_pcts.get(g, 0.0)

                if p_my > abs_threshold:
                    ratios.append(p_other / (p_my + eps))

            if not ratios:
                ratio_threshold = 1.2
            else:
                ratio_threshold = max(
                    float(np.percentile(ratios, tail_percentile)), 1.2
                )

            # --- Detection ---
            for g in target_genes:
                p_my = target_pcts.get(g, 0.0)
                p_other = other_pcts.get(g, 0.0)

                if (p_other > abs_threshold) and (p_other > p_my * ratio_threshold):
                    leakage_mask[g] = (
                        f"CrossLineage_Leak (High in {cl}, "
                        f"Ratio > {ratio_threshold:.1f})"
                    )

        return leakage_mask
