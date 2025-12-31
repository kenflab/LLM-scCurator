# llm_sc_curator/utils.py
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import pandas as pd


DEFAULT_KEYS = ("cell_type", "confidence", "reasoning")


def ensure_json_result(x: Any) -> Dict[str, Any]:
    """
    Normalize LLM outputs into a stable dictionary schema.

    LLM backends may return a dict (JSON), a plain string, or unexpected types.
    This helper enforces a minimal, consistent schema so downstream code can
    reliably build tables, logs, and per-cell annotations.

    Parameters
    ----------
    x : Any
        Raw output from an LLM backend. Expected types include:
        - dict : a JSON object with keys like "cell_type", "confidence", "reasoning"
        - str  : a plain label string (treated as low-confidence)
        Other types are treated as errors and stringified into the "reasoning" field.

    Returns
    -------
    dict
        A dictionary guaranteed to contain:
        - cell_type : str
            Predicted label, or "Unknown"/"Error" on fallback.
        - confidence : {"High", "Medium", "Low"}
            Confidence bucket; defaults to "Low" if missing/invalid.
        - reasoning : str
            Brief explanation (may be empty).

        Any additional keys found in the input dict are preserved to support
        future extensions (e.g., "candidate_labels", "unknown_score").

    Notes
    -----
    This function is intentionally permissive: it never raises, and it prefers
    returning a best-effort normalized object to keep cluster loops running.
    """
    if isinstance(x, dict):
        out = dict(x)  # keep extras for future-proofing
        out.setdefault("cell_type", "Unknown")
        out.setdefault("confidence", "Low")
        out.setdefault("reasoning", "")
        # normalize confidence
        if out.get("confidence") not in {"High", "Medium", "Low"}:
            out["confidence"] = "Low"
        return out

    if isinstance(x, str):
        return {"cell_type": x, "confidence": "Low", "reasoning": ""}

    return {"cell_type": "Error", "confidence": "Low", "reasoning": repr(x)}


def export_cluster_annotation_table(
    adata,
    cluster_col: str,
    cluster_results: Dict[str, Dict[str, Any]],
    genes_by_cluster: Optional[Dict[str, Sequence[str]]] = None,
    prefix: str = "Curated",
) -> pd.DataFrame:
    """
    Build a cluster-level annotation table (DataFrame) from per-cluster LLM outputs.

    This utility converts JSON-like outputs (one result per cluster) into a tidy
    cluster summary table suitable for:
    - CSV export and sharing with non-bioinformatics users
    - downstream plotting and reporting (e.g., cluster composition summaries)
    - reproducible logging of label, confidence, reasoning, and the gene list used

    Parameters
    ----------
    adata : AnnData-like
        Object with `adata.obs` containing a cluster column.
        Only `adata.obs[cluster_col]` is required.
    cluster_col : str
        Column name in `adata.obs` containing cluster IDs (e.g., "seurat_clusters").
        Cluster IDs are coerced to string.
    cluster_results : dict
        Mapping: cluster_id (str) -> result dict.
        Each result dict is expected to include at least:
        - "cell_type"
        - "confidence"
        - "reasoning"
        Extra keys are allowed and will be appended as additional columns.
    genes_by_cluster : dict, optional
        Mapping: cluster_id (str) -> list/sequence of genes used for that cluster.
        If provided, genes are serialized as a semicolon-separated string.
    prefix : str, default="Curated"
        Prefix used to namespace output columns, e.g.:
        - Curated_CellType
        - Curated_Confidence
        - Curated_Reasoning
        - Curated_Genes

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per cluster and at minimum:
        - {cluster_col} : str
        - n_cells : int
        - {prefix}_CellType : str
        - {prefix}_Confidence : str
        - {prefix}_Reasoning : str
        - {prefix}_Genes : str

        If `cluster_results` contains extra keys beyond
        {"cell_type","confidence","reasoning"}, they are included as:
        - {prefix}_<extra_key>

    Notes
    -----
    - This function does not call the LLM. It only formats and aggregates results.
    - Stability: missing clusters in `cluster_results` are filled with "Unknown"/"Low".
    - Extensibility: extra keys (e.g., V2 "unknown_score") are kept as columns.
    """
    clusters = adata.obs[cluster_col].astype(str)
    unique_clusters = sorted(clusters.unique())

    # Determine extra keys across clusters (for V2 extensibility)
    extra_keys = set()
    for c in unique_clusters:
        r = cluster_results.get(str(c), {})
        if isinstance(r, dict):
            extra_keys |= set(r.keys())
    for k in DEFAULT_KEYS:
        extra_keys.discard(k)

    rows = []
    for c in unique_clusters:
        c_str = str(c)
        n_cells = int((clusters == c_str).sum())

        r_norm = ensure_json_result(cluster_results.get(c_str, {}))
        cell_type = str(r_norm.get("cell_type", "Unknown"))
        conf = str(r_norm.get("confidence", "Low"))
        reason = str(r_norm.get("reasoning", ""))

        genes = ""
        if genes_by_cluster is not None:
            g = genes_by_cluster.get(c_str)
            if g:
                genes = ";".join(map(str, g))

        row = {
            cluster_col: c_str,
            "n_cells": n_cells,
            f"{prefix}_CellType": cell_type,
            f"{prefix}_Confidence": conf,
            f"{prefix}_Reasoning": reason,
            f"{prefix}_Genes": genes,
        }

        for k in sorted(extra_keys):
            row[f"{prefix}_{k}"] = r_norm.get(k, None)

        rows.append(row)

    return pd.DataFrame(rows)


def apply_cluster_map_to_cells(
    adata,
    cluster_col: str,
    df_cluster: pd.DataFrame,
    label_col: str,
    new_col: str = "Curated_CellType",
    unknown: str = "Unknown",
):
    """
    Add per-cell labels to `adata.obs` by mapping cluster IDs to cluster-level labels.

    This is the standard "cluster → cell" propagation step:
    - users annotate each cluster once
    - the annotation is assigned to all cells in that cluster

    Parameters
    ----------
    adata : AnnData-like
        Object with `adata.obs` containing a cluster column.
        The function modifies `adata.obs` in-place by adding `new_col`.
    cluster_col : str
        Column name in `adata.obs` containing cluster IDs (e.g., "seurat_clusters").
        Cluster IDs are coerced to string.
    df_cluster : pandas.DataFrame
        Cluster-level annotation table (typically produced by
        `export_cluster_annotation_table`).
        Must contain columns: `[cluster_col, label_col]`.
    label_col : str
        Column in `df_cluster` containing the label to map (e.g., "Curated_CellType").
    new_col : str, default="Curated_CellType"
        Name of the new per-cell column to create in `adata.obs`.
    unknown : str, default="Unknown"
        Fallback label used when a cluster has no entry in `df_cluster`.

    Returns
    -------
    AnnData-like
        The input `adata` with `adata.obs[new_col]` added/updated.

    Notes
    -----
    - This function is intentionally minimal: it performs a direct mapping.
    - If you need label harmonization (synonyms / wording), run `harmonize_labels`
      after mapping, to keep ontology decisions explicit and user-controlled.
    """
    cluster_to_label = (
        df_cluster[[cluster_col, label_col]]
        .dropna()
        .astype({cluster_col: "string", label_col: "string"})
        .set_index(cluster_col)[label_col]
        .to_dict()
    )
    adata.obs[new_col] = (
        adata.obs[cluster_col].astype(str).map(cluster_to_label).fillna(unknown)
    )
    return adata


def harmonize_labels(
    adata,
    col: str,
    mapping: Optional[Dict[str, str]] = None,
    new_col: Optional[str] = None,
    keep_raw: bool = True,
):
    """
    Harmonize label wording using an explicit user-provided mapping dictionary.

    LLM-generated labels may vary in wording (synonyms, punctuation, long descriptors).
    This function applies an explicit mapping to standardize labels for plots and
    downstream summaries, while keeping the raw labels for transparency.

    Parameters
    ----------
    adata : AnnData-like
        Object with `adata.obs[col]` present. The function updates `adata.obs`.
    col : str
        Column name in `adata.obs` containing raw labels to harmonize.
    mapping : dict or None, default=None
        Mapping dict: raw_label -> standardized_label.
        If None, this function is a no-op and returns `adata`.
    new_col : str or None, default=None
        Output column name. If None, defaults to `f"{col}_harmonized"`.
    keep_raw : bool, default=True
        If True, store the original labels in `adata.obs[f"{col}_raw"]`
        before harmonization.

    Returns
    -------
    AnnData-like
        The input `adata` with harmonized labels added as a categorical column.

    Notes
    -----
    - This is intentionally a thin wrapper: ontology decisions stay outside the core
      engine and are encoded explicitly in `mapping`.
    - For safety and reproducibility, this function does not attempt fuzzy matching.
      If desired, fuzzy suggestions can be implemented as a separate helper that
      proposes candidates without auto-replacing.
    """
    if mapping is None:
        return adata

    if new_col is None:
        new_col = f"{col}_harmonized"

    if keep_raw:
        adata.obs[f"{col}_raw"] = adata.obs[col].astype("string")

    adata.obs[new_col] = adata.obs[col].replace(mapping).astype("category")
    return adata
