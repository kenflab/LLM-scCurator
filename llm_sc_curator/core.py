
import scanpy as sc
import pandas as pd
import numpy as np
import time
import json
import re
from scipy.sparse import issparse
import warnings
import logging
from .noise_lists import PROLIFERATION_SENTINELS, NOISE_PATTERNS
from .masking import FeatureDistiller
from .backends import BaseLLMBackend, GeminiBackend, OpenAIBackend, LocalLLMBackend

# Setup Logging
logger = logging.getLogger(__name__)

# Canonical lineage markers for automatic context inference
# Canonical lineage markers - Optimized for Context Inference & Rescue
# Reviewer 2 Note:
# - Includes Human (UPPER) and Mouse (Title/Lower) orthologs.
# - Designed to capture broad lineages for context inference (Stage 0/Context),
#   while confounders (TCR/Ig/Hb) are handled by dedicated noise modules.

LINEAGE_MARKERS = {
    "T cell": [
        # CD3 complex
        "CD3E", "CD3D", "CD3G",
        "Cd3e", "Cd3d", "Cd3g",
        # Co-receptors
        "CD4", "CD8A", "CD8B",
        "Cd4", "Cd8a", "Cd8b1",
        # TCR constant regions (NOT VDJ clones)
        "TRAC", "TRBC1", "TRBC2", "TRGC1", "TRGC2", "TRDC",
        "Trac", "Trbc1", "Trbc2", "Trgc1", "Trgc2", "Trdc",
    ],

    "NK cell": [
        # NK receptors / adapters (avoid cytotoxic-only markers to keep T vs NK separable)
        "NCR1", "KLRD1", "KLRC1", "KLRF1", "TYROBP", "FCGR3A", "NCAM1",
        "Ncr1", "Klrd1", "Klrc1", "Klrf1", "Tyrobp", "Fcgr3", "Ncam1",
    ],

    "B cell": [
        # Pan-B identity
        "MS4A1", "CD79A", "CD79B", "CD19", "BLK", "PAX5",
        "Ms4a1", "Cd79a", "Cd79b", "Cd19", "Blk", "Pax5",
    ],

    "Plasma cell": [
        # Maturation markers & Ig production
        "JCHAIN", "MZB1",
        # Ig constant regions (identity; noise modules will handle redundancy)
        "IGKC", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2",
        # Surface markers
        "SDC1", "TNFRSF17",  # CD138, BCMA
        # Mouse
        "Jchain", "Mzb1",
        "Igkc", "Ighg1", "Ighg2b", "Ighg2c", "Ighg3", "Igha",
        "Sdc1", "Tnfrsf17",
    ],

    "Myeloid": [
        # Monocyte / Macrophage core
        "CD14", "CD68", "LYZ", "CST3", "AIF1", "FCGR3A", "CD163", "MRC1",
        "Cd14", "Cd68", "Lyz2", "Cst3", "Aif1", "Fcgr3", "Cd163", "Mrc1",
        # DC-enriched markers (cDC context)
        "FCER1A", "CLEC9A", "CD1C", "FLT3",
        "Fcer1a", "Clec9a", "Cd1c", "Flt3",
        # NOTE: S100A8/S100A9 are intentionally excluded to avoid　misclassifying stressed epithelium as myeloid.
    ],

    "pDC": [
        # Plasmacytoid DC specific
        "LILRA4", "CLEC4C", "TCF4", "IRF8", "BST2", "TCL1A",
        "Lilra4", "Clec4c", "Tcf4", "Irf8", "Bst2", "Tcl1",
    ],

    "Mast cell": [
        "CPA3", "TPSAB1", "TPSB2", "MS4A2", "KIT", "GATA2",
        "Cpa3", "Tpsab1", "Tpsb2", "Ms4a2", "Kit", "Gata2", "Mcpt4", "Mcpt8",
    ],

    "Fibroblast/Stromal": [
        # Pan-fibroblast / stromal
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "C1S",
        "Col1a1", "Col1a2", "Col3a1", "Dcn", "Lum", "Pdgfra", "C1s",
        # Mural cells (Pericytes / Smooth muscle) included as stromal
        "ACTA2", "RGS5", "MCAM", "TAGLN",
        "Acta2", "Rgs5", "Mcam", "Tagln",
    ],

    "Endothelial": [
        "PECAM1", "VWF", "CDH5", "CLDN5", "FLT1", "KDR",
        "Pecam1", "Vwf", "Cdh5", "Cldn5", "Flt1", "Kdr",
    ],

    "Epithelial": [
        # Pan-epithelial / carcinoma
        "EPCAM", "CDH1",
        "KRT8", "KRT18", "KRT19",
        # Basal / squamous
        "KRT5", "KRT14", "TP63",
        "Epcam", "Cdh1",
        "Krt8", "Krt18", "Krt19",
        "Krt5", "Krt14", "Trp63",
    ],

    "HSPC": [
        "CD34", "SPINK2", "CRHBP", "PROM1", "MN1",
        "Cd34", "Spink2", "Crhbp", "Prom1", "Mn1",
    ],

    "Erythrocyte": [
        # RBC identity: Hb + RBC-specific markers.
        "HBB", "HBA1", "HBA2", "GYPA", "ALAS2",
        "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2", "Hbb-y", "Gypa", "Alas2",
        # Note:
        # - Hemo_Contam noise module will down-weight Hb in non-RBC lineages.
        # - Erythrocyte clusters themselves should remain recognizable via Hb.
    ],
}


class LLMscCurator:
    def __init__(self, api_key=None, model_name=None, backend=None, allow_internal_normalization: bool = False, normalization_target_sum: float = 1e4):
        """
        Initialize LLM-scCurator.

        Args:
            backend (BaseLLMBackend): An instance of a backend (GeminiBackend, OpenAIBackend, etc.).
            api_key (str): (Legacy/Convenience) API key for default Gemini backend.
            model_name (str): (Legacy/Convenience) Model name for default Gemini backend.
        """
        # Dependency Injection Logic
        if backend is not None:
            if not isinstance(backend, BaseLLMBackend):
                raise ValueError("Backend must inherit from BaseLLMBackend.")
            self.llm = backend
            logger.info(f"LLM-scCurator initialized with custom backend: {type(backend).__name__}")
        elif api_key is not None:
            model_name = model_name or "models/gemini-2.0-flash"
            self.llm = GeminiBackend(api_key=api_key, model_name=model_name)
            logger.info(f"LLM-scCurator initialized with default GeminiBackend ({model_name})")
        else:
            raise ValueError("Either a `backend` instance or an `api_key` must be provided.")

        self.masker = None
        self.last_inferred_context = None
        self.allow_internal_normalization = allow_internal_normalization
        self.normalization_target_sum = normalization_target_sum

    # -------------------------------------------------------------------------
    # Normalization helper
    # -------------------------------------------------------------------------
    def _check_normalization(self, adata):
        """
        Ensure that the returned AnnData has log1p-normalized values in `.X`.

        Heuristic:
          - If the dynamic range is small OR values are not integer-like,
            we assume `.X` already contains log1p-normalized expression.
          - If the matrix contains large (> raw_threshold) integer-like values,
            we treat it as raw counts (UMI-like) and optionally normalize internally.

        This function never modifies `adata` in-place. If internal normalization
        is performed, it happens on a COPY that is returned.
        """

        X = adata.X

        # -----------------------------
        # 1) Estimate max and sample a small subset for integer-likeness check
        # -----------------------------
        if issparse(X):
            # Sparse: max is cheap; sample from non-zero data
            max_val = float(X.max())
            if X.nnz > 0:
                sample = X.data[: min(100, X.data.size)]
            else:
                sample = np.array([], dtype=float)
        else:
            # Dense: avoid flattening entire matrix; sample a small top-left block
            n_cells, n_genes = X.shape
            if n_cells == 0 or n_genes == 0:
                max_val = 0.0
                sample = np.array([], dtype=float)
            else:
                n_cells_sample = min(10, n_cells)
                n_genes_sample = min(10, n_genes)
                block = X[:n_cells_sample, :n_genes_sample]
                block_np = np.asarray(block)
                max_val = float(block_np.max()) if block_np.size > 0 else 0.0
                sample = block_np.ravel()[: min(100, block_np.size)]

        # Integer-likeness check (robust to floating point representation)
        if sample.size == 0:
            is_integer_like = False
        else:
            is_integer_like = np.allclose(sample, np.round(sample), atol=1e-6)

        raw_threshold = 50.0
        likely_raw = (max_val > raw_threshold) and is_integer_like

        # -----------------------------
        # 2) Decision logic
        # -----------------------------
        if not likely_raw:
            # Optional: warn if dynamic range is high but non-integer-like
            if max_val > raw_threshold and not is_integer_like:
                logger.warning(
                    "LLM-scCurator detected a large dynamic range in `adata.X` "
                    f"(max={max_val:.1f}) but values are not integer-like. "
                    "Assuming the matrix is already normalized/log-transformed. "
                    "If this is actually raw counts with unusual scaling, please "
                    "normalize explicitly upstream."
                )
            return adata  # treat as already log1p-normalized

        # From here: likely raw counts
        msg = (
            "LLM-scCurator detected likely raw counts in `adata.X` "
            f"(max={max_val:.1f}, integer-like values). "
            "LLM-scCurator expects log1p-normalized expression in `.X`."
        )

        if not getattr(self, "allow_internal_normalization", False):
            logger.error(msg)
            raise ValueError(
                msg
                + " Either normalize upstream "
                  "(e.g. `sc.pp.normalize_total`; `sc.pp.log1p`), or set "
                  "`allow_internal_normalization=True` when initializing "
                  "LLMscCurator."
            )

        logger.warning(
            msg
            + " Proceeding with internal normalization on a COPY of the input "
              "AnnData (`layers['counts']` = raw, `layers['logcounts']` = log1p)."
        )

        # -----------------------------
        # 3) Internal normalization path (opt-in)
        # -----------------------------
        adata_tmp = adata.copy()

        # Preserve raw counts layer if not present
        if "counts" not in adata_tmp.layers:
            adata_tmp.layers["counts"] = adata_tmp.X.copy()

        # Normalize + log1p
        sc.pp.normalize_total(adata_tmp, target_sum=self.normalization_target_sum)
        sc.pp.log1p(adata_tmp)

        # Store log-transformed values
        adata_tmp.layers["logcounts"] = adata_tmp.X.copy()

        # Record normalization metadata for reproducibility
        if "llm_sc_curator" not in adata_tmp.uns:
            adata_tmp.uns["llm_sc_curator"] = {}
        adata_tmp.uns["llm_sc_curator"]["internal_normalization"] = {
            "method": "normalize_total+log1p",
            "target_sum": float(self.normalization_target_sum),
            "source": "LLMscCurator._check_normalization",
            "raw_threshold": float(raw_threshold),
            "integer_tolerance": 1e-6,
        }

        return adata_tmp

    def set_global_context(
        self,
        adata,
        balance_by: str | None = None,
        max_cells_per_group: int | str = "auto",
        min_cells_per_group: int = 50,
        random_state: int = 42,
    ):
        """
        Set the global dataset context for Gini / housekeeping detection and
        cross-lineage specificity checks, with optional dynamic balanced subsampling.

        Parameters
        ----------
        adata
            AnnData object containing all cells. Can be raw counts or log1p-normalized;
            normalization is checked via `_check_normalization`.
        balance_by
            Optional column name in `adata.obs` used to construct a balanced subsample.
            Typical choices are a broad lineage label (e.g. "major_type") or a
            sample/patient ID (e.g. "sample_id"). If None (default), all cells are
            used without subsampling.
        max_cells_per_group
            If an integer, strictly caps the number of cells retained per group
            defined by `balance_by`.

            If the string "auto" (default), the target size per group is dynamically
            determined from the median group size in `balance_by`, with a hard floor
            at `min_cells_per_group`. This adapts to dataset topology (rare vs.
            abundant groups) while preventing pathologically small targets.
        min_cells_per_group
            Minimum number of cells required for a group to be included in the global
            context. Groups with fewer cells than this threshold are skipped when
            building the balanced subsample, to avoid unstable global statistics.
        random_state
            Seed for the NumPy random number generator used during subsampling.

        Notes
        -----
        - This function never modifies `adata` in-place. All normalization and
          subsampling happen on internal copies.
        - When `balance_by` is provided, global statistics are computed on a
          balanced subsample, which prevents very large groups from dominating the
          Gini distribution and cross-lineage percentiles. This is intended to
          improve robustness across heterogeneous atlases (e.g. pan-cancer T cell
          datasets) and is suitable for rigorous benchmarking.
        """
        # 1) Ensure we are working on log1p-normalized values (or raise)
        adata_log = self._check_normalization(adata)
        n_cells_input = int(adata_log.n_obs)

        # 2) No balancing requested → use all cells
        if balance_by is None:
            context = adata_log
            logger.info(
                "Global context set: using all %d cells (no balanced subsampling).",
                context.n_obs,
            )
        else:
            # 3) Dynamic balancing logic
            if balance_by not in adata_log.obs.columns:
                logger.warning(
                    "Column '%s' not found in adata.obs; falling back to using all cells.",
                    balance_by,
                )
                context = adata_log
            else:
                groups = adata_log.obs[balance_by].astype("category")
                counts = groups.value_counts()

                # --- Determine target N per group ---
                if max_cells_per_group == "auto":
                    median_size = int(counts.median())
                    target_n = max(median_size, min_cells_per_group)
                    logger.info(
                        "[GlobalContext] Dynamic balancing enabled. "
                        "Median group size: %d → target per group: %d.",
                        median_size,
                        target_n,
                    )
                else:
                    target_n = int(max_cells_per_group)
                    logger.info(
                        "[GlobalContext] Fixed balancing enabled. "
                        "Target per group: %d cells.",
                        target_n,
                    )

                rng = np.random.default_rng(random_state)
                keep_indices = []

                for level in groups.cat.categories:
                    idx = np.where((groups == level).values)[0]
                    n = idx.size

                    if n == 0:
                        continue

                    if n < min_cells_per_group:
                        logger.info(
                            "[GlobalContext] Skipping group '%s' with only %d cells "
                            "(< min_cells_per_group=%d).",
                            level, n, min_cells_per_group,
                        )
                        continue

                    if n > target_n:
                        chosen = rng.choice(idx, size=target_n, replace=False)
                        logger.info(
                            "[GlobalContext] Subsampled group '%s': %d → %d cells.",
                            level, n, target_n,
                        )
                    else:
                        chosen = idx
                        logger.info(
                            "[GlobalContext] Using all %d cells from group '%s'.",
                            n, level,
                        )

                    keep_indices.append(chosen)

                if not keep_indices:
                    logger.warning(
                        "[GlobalContext] No groups met min_cells_per_group=%d. "
                        "Falling back to using all cells as global context.",
                        min_cells_per_group,
                    )
                    context = adata_log
                else:
                    keep_indices = np.concatenate(keep_indices)
                    keep_indices.sort()
                    context = adata_log[keep_indices].copy()
                    logger.info(
                        "[GlobalContext] Balanced subsample created using '%s'. "
                        "Input cells: %d → context cells: %d (target per group: %d).",
                        balance_by,
                        n_cells_input,
                        context.n_obs,
                        target_n,
                    )

        # 4) Initialize FeatureDistiller with the chosen context
        self.masker = FeatureDistiller(context)
        logger.info("Global context set. Cross-lineage specificity checks enabled.")

        # 5) Record configuration for reproducibility / Methods
        self._global_context_config = {
            "balance_by": balance_by,
            "mode": str(max_cells_per_group),
            "target_n": int(target_n) if "target_n" in locals() else None,
            "min_cells_per_group": int(min_cells_per_group),
            "random_state": int(random_state),
            "n_cells_input": n_cells_input,
            "n_cells_context": int(context.n_obs),
        }


    # -------------------------------------------------------------------------
    # Lineage inference
    # -------------------------------------------------------------------------
    def _infer_lineage_context(self, adata, target_group, group_col):
        """Infers broad lineage context."""
        try:
            cells = adata[adata.obs[group_col] == target_group]
            scores = {}
            for lineage, markers in LINEAGE_MARKERS.items():
                valid_markers = [m for m in markers if m in adata.var_names]
                if not valid_markers:
                    scores[lineage] = 0.0
                    continue

                X_sub = cells[:, valid_markers].X
                expr = float(X_sub.mean()) if hasattr(X_sub, "mean") else float(np.mean(X_sub))
                scores[lineage] = expr

            best_lineage = max(scores, key=scores.get)
            if scores[best_lineage] < 0.1:
                return "Unknown/Mixed"
            return best_lineage
        except Exception:
            return "Unknown"

    # -------------------------------------------------------------------------
    # High-Gini rescue
    # -------------------------------------------------------------------------
    def _get_high_gini_genes(
        self,
        gini_q: float = 0.9,
        mean_upper_q: float = 0.5,
        mean_lower_abs: float = 0.1,
    ):
        """
        HVG fails to capture low-to-moderate, lineage-restricted markers of rare populations;
        we therefore add a global high-Gini rescue mechanism.

        Args:
            gini_q (float): quantile for Gini (e.g., 0.9 = top 10%).
            mean_upper_q (float): upper quantile for mean (e.g., 0.5 = below median).
            mean_lower_abs (float): absolute lower bound on mean to avoid pure drop-out noise.
        """
        # 1) No masker → Disable high-Gini rescue itself
        if self.masker is None:
            logger.warning(
                "Masker not initialized; high-Gini rescue disabled. "
                "Call `set_global_context()` for global Gini-based rescue."
            )
            return set()

        # 2) No gene_stats → Calculate global Gini/mean
        if self.masker.gene_stats is None:
            logger.info("Computing global gene_stats for high-Gini rescue...")
            try:
                self.masker.calculate_gene_stats()
            except Exception as e:
                logger.warning(f"Failed to compute gene_stats: {e}. Disabling high-Gini rescue.")
                return set()

        gs = self.masker.gene_stats.copy()
        gs = gs.replace([np.inf, -np.inf], np.nan).dropna(subset=["gini", "mean"])

        g_thr = np.quantile(gs["gini"].values, gini_q)
        m_hi = np.quantile(gs["mean"].values, mean_upper_q)

        sel = gs[
            (gs["gini"] >= g_thr) &
            (gs["mean"] <= m_hi) &
            (gs["mean"] >= mean_lower_abs)
        ].index

        logger.info(
            f"High-Gini rescue: gini >= {g_thr:.3f} (q={gini_q}), "
            f"mean between {mean_lower_abs:.3f} and <= {m_hi:.3f}, "
            f"selected {len(sel)} genes."
        )
        return set(sel)

    # -------------------------------------------------------------------------
    # Core Feature Curation
    # -------------------------------------------------------------------------

    def curate_features(
        self,
        target_adata,
        group_col,
        target_group,
        reference="rest",
        n_top=50,
        use_statistics=True,
        use_hvg=True,
        coarse_col=None,
        whitelist=None,
        batch_key=None,
        n_candidates: int = 500,
        min_target_mean: float = 0.02,
        min_delta_mean: float = 0.02,
        min_logfc: float = 0.2,
        min_target_pct: float = 0.02,
        min_delta_pct: float = 0.02,
    ):
        """
        4-Stage Feature Distillation.

        Args:
            target_adata: AnnData object (can be a subset or full).
            group_col: Column with cluster labels in target_adata.
            target_group: The cluster ID to analyze.
            reference: "rest" or a specific group label.
            coarse_col: (Optional) Column in GLOBAL adata for cross-lineage check.

            min_target_mean: Minimum mean expression (log1p) in the target cluster.
            min_delta_mean: Minimum difference in mean (target - rest).
            min_logfc: Minimum log fold-change (if available in DE table).
            min_target_pct: Minimum detection fraction in target cluster.
            min_delta_pct: Minimum difference in detection fraction (target - rest).
        """
        # 0. Ensure log1p-normalized data
        target_adata = self._check_normalization(target_adata)

        if self.masker is None:
            logger.warning(
                "Global context not set before feature curation; "
                "initializing FeatureDistiller on local `target_adata`. "
                "For cross-dataset benchmarking and lineage leakage checks, "
                "consider calling `set_global_context()` on a larger atlas."
            )
            self.masker = FeatureDistiller(target_adata)

        # --- Stage 0: HVG Selection + High-Gini Rescue ---
        if use_hvg:
            # Case 1: HVGs are already present (e.g., pre-computed globally)
            if "highly_variable" in target_adata.var.columns:
                hvgs = set(target_adata.var_names[target_adata.var["highly_variable"]])
            else:
                # Case 2: Compute HVGs on-the-fly
                logger.warning("HVG not found. Calculating on-the-fly.")
                adata_hvg = target_adata.copy()
                try:
                    n_genes = adata_hvg.n_vars
                    n_hvg = min(2000, n_genes)

                    # ---- HVG flavor/layer policy ----
                    hvg_kwargs = {
                        "n_top_genes": n_hvg,
                        "subset": False,
                    }

                    if batch_key is not None:
                        hvg_kwargs["batch_key"] = batch_key

                        # Prefer Seurat v3 on raw counts if available
                        if "counts" in adata_hvg.layers:
                            hvg_kwargs["flavor"] = "seurat_v3"
                            hvg_kwargs["layer"] = "counts"
                        else:
                            logger.warning(
                                "[HVG] batch_key provided but no `layers['counts']` found; "
                                "using flavor='seurat' on current `.X` "
                                "(assumed log1p-normalized)."
                            )
                            hvg_kwargs["flavor"] = "seurat"
                    else:
                        # No batch correction → classical Seurat on log1p `.X`
                        hvg_kwargs["flavor"] = "seurat"

                    sc.pp.highly_variable_genes(adata_hvg, **hvg_kwargs)
                    hvgs = set(
                        adata_hvg.var_names[adata_hvg.var["highly_variable"]]
                    )
                except Exception:
                    logger.warning(
                        "[HVG] highly_variable_genes failed; falling back to using all genes."
                    )
                    hvgs = set(target_adata.var_names)

            # High-Gini rescue (requires global masker)
            high_gini = self._get_high_gini_genes(
                gini_q=0.9, mean_upper_q=0.5, mean_lower_abs=0.1
            )

            # Lineage Marker rescue
            lineage_genes = set()
            for markers in LINEAGE_MARKERS.values():
                for g in markers:
                    if g in target_adata.var_names:
                        lineage_genes.add(g)

            keep_genes = sorted(
                hvgs.union(high_gini).union(lineage_genes).intersection(target_adata.var_names)
            )

            if len(keep_genes) == 0:
                adata_work = target_adata.copy()
            else:
                adata_work = target_adata[:, keep_genes].copy()
        else:
            adata_work = target_adata.copy()

        # --- Stage 1: Differential Expression ---
        if reference == "rest":
            adata_work.obs["binary_group"] = "Rest"
            if target_group not in adata_work.obs[group_col].unique():
                raise ValueError(f"Target group '{target_group}' not found.")
            adata_work.obs.loc[
                adata_work.obs[group_col] == target_group, "binary_group"
            ] = "Target"

            sc.tl.rank_genes_groups(
                adata_work,
                groupby="binary_group",
                groups=["Target"],
                reference="Rest",
                method="wilcoxon",
                use_raw=False,
            )
            de_df = sc.get.rank_genes_groups_df(adata_work, group="Target")
            de_df_raw = de_df.copy()
            target_mask = adata_work.obs["binary_group"] == "Target"
            rest_mask   = adata_work.obs["binary_group"] == "Rest"
        else:
            sc.tl.rank_genes_groups(
                adata_work,
                groupby=group_col,
                groups=[target_group],
                reference=reference,
                method="wilcoxon",
                use_raw=False,
            )
            de_df = sc.get.rank_genes_groups_df(adata_work, group=target_group)
            de_df_raw = de_df.copy()
            target_mask = adata_work.obs[group_col] == target_group
            rest_mask   = adata_work.obs[group_col] == reference

        # --- Attach expression statistics (cluster-level means and detection) ---
        X = adata_work.X
        if issparse(X):
            X_target = X[target_mask.values, :]
            X_rest   = X[rest_mask.values, :]
            target_mean = np.asarray(X_target.mean(axis=0)).ravel()
            rest_mean   = np.asarray(X_rest.mean(axis=0)).ravel()
            target_pct  = np.asarray((X_target > 0).mean(axis=0)).ravel()
            rest_pct    = np.asarray((X_rest > 0).mean(axis=0)).ravel()
        else:
            X_target = X[target_mask.values, :]
            X_rest   = X[rest_mask.values, :]
            target_mean = X_target.mean(axis=0)
            rest_mean   = X_rest.mean(axis=0)
            target_pct  = (X_target > 0).mean(axis=0)
            rest_pct    = (X_rest > 0).mean(axis=0)

        expr_stats = pd.DataFrame({
            "names": adata_work.var_names,
            "target_mean": target_mean,
            "rest_mean": rest_mean,
            "target_pct": target_pct,
            "rest_pct": rest_pct,
        })
        de_df = de_df.merge(expr_stats, on="names", how="left")

        de_df["delta_mean"] = de_df["target_mean"] - de_df["rest_mean"]
        de_df["delta_pct"]  = de_df["target_pct"] - de_df["rest_pct"]

        # --- Effect-size / abundance filters to suppress lowly expressed DEGs ---
        eff_mask = (
            (de_df["target_mean"] >= min_target_mean) &
            (de_df["delta_mean"]  >= min_delta_mean) &
            (de_df["target_pct"]  >= min_target_pct) &
            (de_df["delta_pct"]   >= min_delta_pct)
        )

        if "logfoldchanges" in de_df.columns:
            eff_mask &= (de_df["logfoldchanges"].fillna(0) >= min_logfc)

        de_df_filtered = de_df.loc[eff_mask].copy()
        if de_df_filtered.empty:
            logger.warning(
                "All DE genes filtered out by expression/effect-size thresholds; "
                "falling back to unfiltered DE ranking."
            )
            de_df = de_df_raw
        else:
            de_df = de_df_filtered

        # --- Update lineage context ---
        self.last_inferred_context = self._infer_lineage_context(
            target_adata, target_group, group_col
        )

        # --- Stage 2 & 3: Biological Masking ---
        if self.masker is None:
            logger.warning(
                "Global context not set. Initializing masker with local data "
                "(Stage 3 limited). Call set_global_context() first for best results."
            )
            local_adata = self._check_normalization(target_adata)
            self.masker = FeatureDistiller(local_adata)

        mask_reasons = {}
        if use_statistics:
            combined_whitelist = set(whitelist or [])
            combined_whitelist |= PROLIFERATION_SENTINELS

            lineage_genes_all = set()
            for markers in LINEAGE_MARKERS.values():
                lineage_genes_all.update(markers)

            # Here, items matching NOISE_PATTERNS (TCR_Clone, Ig_Clone, Ig_Constant*, Hemo_Contam...) are excluded from the whitelist and handled by dedicated modules.
            def _is_confounded_gene(g):
                for key in ["TCR_Clone", "TCR_Clone_Mouse",
                            "Ig_Clone", "Ig_Clone_Mouse",
                            "Ig_Constant_Heavy", "Ig_Constant_Light_Kappa", "Ig_Constant_Light_Lambda",
                            "Ig_Constant_Heavy_Mouse", "Ig_Constant_Light_Kappa_Mouse", "Ig_Constant_Light_Lambda_Mouse",
                            "Hemo_Contam"]:
                    pattern = NOISE_PATTERNS.get(key, None)
                    if pattern is None:
                        continue
                    if re.match(pattern, g):
                        return True
                return False
            
            lineage_whitelist = {g for g in lineage_genes_all if not _is_confounded_gene(g)}
            combined_whitelist |= lineage_whitelist

            mask_reasons = self.masker.detect_biological_noise(
                gini_threshold=None,
                gini_q=0.01,
                mean_floor=0.01,
                whitelist=list(combined_whitelist),
                rescue_mean_floor=0.05,
                low_gini_cap=0.15
            )
            
        # Stage 3: Cross-lineage leak check
        if coarse_col:
            candidates = de_df["names"].head(n_candidates).tolist()
            lineage_mask = self.masker.calculate_lineage_specificity(
                target_genes=candidates,
                target_adata=target_adata,
                target_group=target_group,
                group_col=group_col,
                coarse_col=coarse_col,
            )
            mask_reasons.update(lineage_mask)

        # --- Final selection: top n_top clean genes ---
        all_candidates = de_df["names"].tolist()[:n_candidates]
        clean_genes: list[str] = []
        for g in all_candidates:
            if g in mask_reasons:
                continue
            clean_genes.append(g)
            if len(clean_genes) >= n_top:
                break

        # --- Fallback: if everything got masked, relax filters ---
        if len(clean_genes) == 0:
            logger.warning(
                "No clean genes remained after noise masking; "
                "falling back to DE ranking without noise mask for this cluster."
            )
            # Fallback 1: use expression/effect-size filtered DE genes (no mask)
            fallback = de_df["names"].tolist()[:n_top]

            if len(fallback) == 0:
                logger.warning(
                    "Expression-filtered DE list is also empty; "
                    "falling back to raw DE ranking."
                )
                # Fallback 2: raw DE
                fallback = de_df_raw["names"].tolist()[:n_top]

            clean_genes = fallback

        return clean_genes

    # -------------------------------------------------------------------------
    # Annotation
    # -------------------------------------------------------------------------
    def annotate(
        self,
        gene_list,
        cell_type: str = "",
        context: dict | None = None,
        use_auto_context: bool = True,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
    ):
        """
        Query the injected LLM backend with a distilled marker gene list and optional context,
        and return a dict with {cell_type, confidence, reasoning}.

        Robust to transient backend failures:
        - Retries up to `max_retries` times on any backend / JSON error.
        - If all attempts fail, returns {"cell_type": "Error" or "ParseError", ...}
          without raising, so that callers can decide how to handle failures.
        """
        genes_str = ", ".join(gene_list)

        # -----------------------------
        # 1) Build biological context string
        # -----------------------------
        context_str = "\n[Biological Context]\n"

        if context:
            for k, v in context.items():
                context_str += f"- {k}: {v}\n"

        if cell_type:
            context_str += f"- Known Parent Lineage: {cell_type}\n"
        elif use_auto_context and self.last_inferred_context:
            context_str += f"- Inferred Parent Lineage: {self.last_inferred_context}\n"
        else:
            context_str += "- Tissue/Condition: Unspecified\n"

        # Proliferation sentinel note
        prolif_hits = [g for g in gene_list if g in PROLIFERATION_SENTINELS]
        if prolif_hits:
            show = ", ".join(prolif_hits[:3])
            context_str += (
                f"- Note: Proliferation markers detected ({show}). "
                "Prioritize identifying the lineage/subtype, but append "
                "'(proliferating)' after the main cell type when appropriate.\n"
            )

        # -----------------------------
        # 2) Construct prompt
        # -----------------------------
        prompt = f"""
        Role: Expert in single-cell transcriptomics.
        Context:
        {context_str}
        Input Genes: [{genes_str}]

        Task:
          Identify the SINGLE BEST Subtype & Lineage.
          The main part of `cell_type` MUST describe the lineage/subtype
          (e.g., "CD8+ exhausted T cell", "CD4 Temra.EffMem T cell", "MAIT cell", "Naive B cell").
          State information (ISG-high, proliferating, etc.) should be placed
          in parentheses after the lineage, e.g. "CD8+ T cell (ISG-high)".

        Output STRICTLY in JSON. Do not include any additional text:
        {{
          "cell_type": "The precise subtype name",
          "confidence": "High/Medium/Low",
          "reasoning": "Brief justification based on key markers"
        }}
        """

        last_exception = None
        last_raw = None

        for attempt in range(1, max_retries + 1):
            try:
                response_text = self.llm.generate(prompt, json_mode=True)
                last_raw = response_text

                # --- JSON parsing (primary path) ---
                clean_text = (
                    response_text.replace("```json", "")
                                 .replace("```", "")
                                 .strip()
                )
                try:
                    parsed = json.loads(clean_text)
                except json.JSONDecodeError:
                    # Fallback: try to extract the first JSON object from the raw text
                    match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if not match:
                        raise  # escalate to outer except

                    fallback_json = match.group(0)
                    parsed = json.loads(fallback_json)

                result_dict = parsed

                # If the backend returns a list of objects, pick the first dict
                if isinstance(result_dict, list):
                    if not result_dict:
                        raise ValueError("Empty JSON list returned from LLM.")
                    first = result_dict[0]
                    if isinstance(first, dict):
                        result_dict = first
                    else:
                        found = None
                        for el in result_dict:
                            if isinstance(el, dict):
                                found = el
                                break
                        if found is None:
                            raise ValueError(
                                f"JSON list does not contain a dictionary object: {result_dict}"
                            )
                        result_dict = found

                if not isinstance(result_dict, dict):
                    raise ValueError(
                        f"Parsed JSON is not a dict (got {type(result_dict)}). "
                        f"Raw response: {response_text}"
                    )

                cell_type_val = result_dict.get("cell_type", "Unknown")
                conf_val = result_dict.get("confidence", "Low")
                reason_val = result_dict.get("reasoning", "")

                # Normalize confidence
                if conf_val not in {"High", "Medium", "Low"}:
                    conf_val = "Low"

                return {
                    "cell_type": cell_type_val,
                    "confidence": conf_val,
                    "reasoning": reason_val,
                }

            except json.JSONDecodeError as e:
                last_exception = e
                logger.error(
                    f"JSON parsing failed in annotate() "
                    f"(attempt {attempt}/{max_retries}). Raw response: {last_raw}"
                )
            except Exception as e:
                last_exception = e
                logger.error(
                    f"Unexpected error in annotate() "
                    f"(attempt {attempt}/{max_retries}): {e} | Raw: {last_raw}"
                )

            # If not returned yet and attempts remain, wait and retry
            if attempt < max_retries:
                time.sleep(retry_sleep)

        # -----------------------------
        # 3) If all attempts failed, return a soft failure
        # -----------------------------
        if isinstance(last_exception, json.JSONDecodeError):
            label = "ParseError"
            reason = (
                f"Failed to parse LLM output as JSON after {max_retries} attempts: "
                f"{last_exception}"
            )
        else:
            label = "Error"
            reason = (
                f"LLM backend error after {max_retries} attempts: {last_exception}"
            )

        return {
            "cell_type": label,
            "confidence": "Low",
            "reasoning": reason,
        }

    # -------------------------------------------------------------------------
    # Hierarchical discovery
    # -------------------------------------------------------------------------
    def run_hierarchical_discovery(
        self,
        adata,
        coarse_res=0.2,
        fine_res=0.5,
        n_top=50,
        batch_key=None,
        global_context: dict | None = None,
        random_state: int = 42,
    ):
        """
        Executes hierarchical annotation.
        Efficiently handles HVGs by pre-calculating them once.
        """
        logger.info(
            f"Starting Hierarchical Discovery "
            f"(Coarse: {coarse_res}, Fine: {fine_res}, n_top={n_top})..."
        )

        base_context: dict = global_context or {}

        # Ensure we operate on a log1p-normalized view
        adata = self._check_normalization(adata)

        # ------------------------------------------------------------------
        # Global HVG setup
        # ------------------------------------------------------------------
        if "highly_variable" not in adata.var.columns:
            logger.info("  [Setup] Pre-calculating HVGs globally...")
            try:
                n_genes = adata.n_vars
                n_hvg = min(2000, n_genes)

                # HVG policy:
                # - If batch_key is provided and raw counts are available in
                #   `layers['counts']`, use Seurat v3 on counts.
                # - Otherwise, fall back to classical Seurat on log1p-normalized `.X`.
                hvg_kwargs = {
                    "n_top_genes": n_hvg,
                    "subset": False,
                }

                if batch_key is not None:
                    hvg_kwargs["batch_key"] = batch_key

                    if "counts" in adata.layers:
                        # Preferred: v3 on counts
                        hvg_kwargs["flavor"] = "seurat_v3"
                        hvg_kwargs["layer"] = "counts"
                        logger.info(
                            "[HVG] Using flavor='seurat_v3' on `layers['counts']` "
                            "with batch_key='%s'.",
                            batch_key,
                        )
                    else:
                        # No raw layer → stay in log1p space with classical Seurat
                        logger.warning(
                            "[HVG] batch_key provided but no `layers['counts']` found; "
                            "using flavor='seurat' on current `.X` "
                            "(assumed log1p-normalized)."
                        )
                        hvg_kwargs["flavor"] = "seurat"
                else:
                    # No batch correction: classical Seurat on log1p-normalized `.X`
                    hvg_kwargs["flavor"] = "seurat"
                    logger.info(
                        "[HVG] Using flavor='seurat' on log1p-normalized `.X` "
                        "(no batch_key)."
                    )

                sc.pp.highly_variable_genes(adata, **hvg_kwargs)

            except Exception as e:
                logger.warning(
                    "  HVG pre-calculation failed; proceeding without HVG flags. "
                    f"Reason: {e}"
                )

        # Initialize Global Masker Context
        # Step 1: Coarse annotation using a global (unbalanced) context.
        # Here we quantify cell-wise Gini / housekeeping statistics across all cells.
        self.set_global_context(adata, balance_by=None)

        # --- Step 1: Major Lineage ---
        logger.info("[Step 1] Identifying Major Lineages...")

        sc.pp.pca(adata, random_state=random_state)
        sc.pp.neighbors(adata, random_state=random_state)
        sc.tl.leiden(adata, resolution=coarse_res, key_added="leiden_coarse", random_state=random_state)

        major_map = {}
        adata.obs["major_type"] = "Unknown"

        # Initialize container for reasoning logs
        if "llm_reasoning" not in adata.uns: adata.uns["llm_reasoning"] = {}

        for cl in adata.obs["leiden_coarse"].unique():
            genes = self.curate_features(
                adata,
                group_col="leiden_coarse",
                target_group=cl,
                n_top=n_top,
                use_hvg=True,
                batch_key=batch_key,
            )
            res = self.annotate(genes, context=base_context)
            identity = res.get("cell_type", "Unknown")

            if identity in {"Error", "ParseError"}:
                logger.error(
                    f"Coarse cluster {cl}: annotation failed (cell_type={identity}); "
                    "falling back to 'Unknown'."
                )
                identity = "Unknown"

            major_map[cl] = identity
            logger.info(f"  Coarse cluster {cl} -> {identity}")

            adata.uns["llm_reasoning"][f"Coarse_{cl}"] = res
            time.sleep(2)

        adata.obs["major_type"] = adata.obs["leiden_coarse"].map(major_map)

        # In Step 2's fine subtyping, we examine cross-lineage leaks in a balanced global context for each major_type.
        # After coarse annotation, we reinitialize the global context using a balanced design across major lineages. This prevents abundant
        # lineages from dominating cross-lineage percentile estimates and is used in Stage 3 (lineage leakage checks) during fine subtyping.
        self.set_global_context(adata, balance_by="major_type", max_cells_per_group="auto")

        # --- Step 2: Fine Subtyping ---
        logger.info("[Step 2] Drilling down into Subtypes...")
        fine_results = pd.Series(index=adata.obs_names, dtype="object")
        adata.obs["fine_type"] = "Unknown"

        for major_cl in adata.obs["leiden_coarse"].unique():
            major_name = major_map[major_cl]
            logger.info(f"  Analyzing '{major_name}' (Coarse {major_cl})...")

            subset = adata[adata.obs["leiden_coarse"] == major_cl].copy()
            if subset.n_obs < 50:
                fine_results.loc[subset.obs_names] = major_name
                continue

            sc.pp.pca(subset, random_state=random_state)
            sc.pp.neighbors(subset, random_state=random_state)
            sc.tl.leiden(subset, resolution=fine_res, key_added="leiden_fine", random_state=random_state)
            fine_clusters = subset.obs["leiden_fine"].unique()

            if len(fine_clusters) == 1:
                fine_results.loc[subset.obs_names] = major_name
                logger.info(
                    f"  Coarse {major_cl} ('{major_name}') did not split at "
                    f"fine_res={fine_res}; propagating major label to fine_type."
                )
                continue

            for sub_cl in fine_clusters:
                sub_genes = self.curate_features(
                    target_adata=subset, group_col="leiden_fine", target_group=sub_cl,
                    coarse_col="major_type", n_top=n_top, use_hvg=True, batch_key=batch_key
                )

                sub_res = self.annotate(sub_genes, cell_type=major_name, context=base_context)
                sub_identity = sub_res.get("cell_type", "Unknown")

                if sub_identity in {"Error", "ParseError"}:
                    logger.error(
                        f"  Fine cluster {sub_cl} in coarse {major_cl} ('{major_name}'): "
                        f"annotation failed (cell_type={sub_identity}); "
                        "falling back to major label."
                    )
                    sub_identity = major_name

                cells_in_sub = subset.obs[subset.obs["leiden_fine"] == sub_cl].index
                fine_results.loc[cells_in_sub] = sub_identity

                logger.info(f"    Fine {sub_cl} -> {sub_identity}")
                adata.uns["llm_reasoning"][f"Fine_{major_cl}_{sub_cl}"] = sub_res
                time.sleep(2)

        adata.obs["fine_type"] = fine_results
        logger.info("Hierarchical Annotation Complete.")

        return adata
