#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from llm_sc_curator import LLMscCurator
from llm_sc_curator.backends import BaseLLMBackend
from llm_sc_curator.noise_lists import NOISE_PATTERNS, NOISE_LISTS
from benchmarks.gt_mappings import get_cd8_ground_truth


warnings.filterwarnings("ignore")


# =============================================================================
# Display / figure style
# =============================================================================

COLORS = {
    "Standard": "#5DA5DA",
    "LLM-scCurator": "#D62728",
}

FULL_CORE_COLOR = COLORS["LLM-scCurator"]
LOO_COLOR = "#BDBDBD"
EDGE_COLOR = "black"

VARIANT_ORDER = [
    "full_core",
    "minus_curated_noise_mask",
    "minus_low_gini_suppression",
    "minus_high_gini_rescue",
    "minus_sentinel_retention",
    "minus_cross_lineage_filter",
]

VARIANT_LABELS = {
    "full_core": "full_core",
    "minus_curated_noise_mask": "- curated\nnoise mask",
    "minus_low_gini_suppression": "- low-Gini\nsuppression",
    "minus_high_gini_rescue": "- high-Gini\nrescue",
    "minus_sentinel_retention": "- sentinel\nretention",
    "minus_cross_lineage_filter": "- cross-lineage\nfilter",
}

COMPONENT_REMOVED = {
    "full_core": "None",
    "minus_curated_noise_mask": "Curated biological-noise masking",
    "minus_low_gini_suppression": "Low-Gini housekeeping suppression",
    "minus_high_gini_rescue": "High-Gini rare-marker rescue / candidate augmentation",
    "minus_sentinel_retention": "Canonical / sentinel marker retention",
    "minus_cross_lineage_filter": "Cross-lineage leakage filtering",
}


CD8_MARKER_DB = {
    "CD8_Naive": {
        "IL7R", "CCR7", "LEF1", "TCF7", "SELL", "MAL", "LTB", "KLF2"
    },
    "CD8_EffectorMemory": {
        "GZMK", "LTB", "AQP3", "IL7R", "CXCR4", "ANXA1", "ZFP36L2", "DUSP2"
    },
    "CD8_Effector": {
        "CCL5", "NKG7", "PRF1", "GZMB", "CX3CR1", "KLRG1", "FGFBP2", "GNLY"
    },
    "CD8_Exhausted": {
        "CXCL13", "CTLA4", "TIGIT", "HAVCR2", "PDCD1", "ENTPD1", "TOX", "TNFRSF9"
    },
    "CD8_ISG": {
        "IFIT1", "ISG15", "MX1", "STAT1", "OAS1", "IFI6"
    },
    "CD8_MAIT": {
        "SLC4A10", "KLRB1", "CXCR6"
    },
    "CD8_Cycling": {
        "MKI67", "TOP2A", "CDK1", "BIRC5", "PCNA", "TYMS"
    },
    "CD8_NK_Killer": {
        "NKG7", "GNLY", "PRF1", "GZMB", "FGFBP2"
    },
}


CROSS_LINEAGE_LEAKAGE_GENES = {
    # B / plasma
    "MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "IGHG1", "IGHG3", "IGKC",
    # Myeloid
    "LYZ", "LST1", "S100A8", "S100A9", "FCGR3A", "FCN1", "MS4A7", "CST3",
    # Epithelial / tumor
    "EPCAM", "KRT8", "KRT18", "KRT19", "KRT5", "KRT14", "MUC1",
    # Endothelial / stromal
    "PECAM1", "VWF", "KDR", "COL1A1", "COL1A2", "DCN", "LUM", "ACTA2", "RGS5",
    # Platelet / erythroid
    "PPBP", "PF4", "HBA1", "HBA2", "HBB",
    # CD4 helper leakage
    "CD4", "IL2RA", "FOXP3", "CCR4", "CCR6",
}


class NullBackend(BaseLLMBackend):
    def generate(self, prompt: str, json_mode: bool = False, **kwargs) -> str:
        raise RuntimeError("NullBackend does not support annotation calls.")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
        }
    )


def find_default_input() -> Path:
    candidates = [
        Path("paper/gb_resubmission/input/cd8_benchmark_data.h5ad"),
        Path("gb_resubmission/input/cd8_benchmark_data.h5ad"),
        Path("input/cd8_benchmark_data.h5ad"),
        Path("../input/cd8_benchmark_data.h5ad"),
        Path("/work/paper/gb_resubmission/input/cd8_benchmark_data.h5ad"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find cd8_benchmark_data.h5ad. "
        "Please pass --input-h5ad /path/to/cd8_benchmark_data.h5ad"
    )


def _to_1d_array(x):
    if issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def ordered_unique(items):
    seen = set()
    out = []
    for x in items:
        x = str(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


COMPILED_REGEX_NOISE = {
    name: re.compile(pattern)
    for name, pattern in NOISE_PATTERNS.items()
}

CURATED_NOISE_GENE_SET = set()
for _, genes in NOISE_LISTS.items():
    CURATED_NOISE_GENE_SET.update(map(str, genes))


def is_regex_noise_gene(gene: str) -> bool:
    g = str(gene)
    return any(p.search(g) for p in COMPILED_REGEX_NOISE.values())


def is_curated_noise_gene(gene: str) -> bool:
    return str(gene) in CURATED_NOISE_GENE_SET


def is_any_noise_gene(gene: str) -> bool:
    return is_regex_noise_gene(gene) or is_curated_noise_gene(gene)


def is_cross_lineage_gene(gene: str) -> bool:
    return str(gene) in CROSS_LINEAGE_LEAKAGE_GENES


def build_de_table(
    adata,
    group_col,
    cluster_name,
    min_target_mean=0.02,
    min_delta_mean=0.02,
    min_logfc=0.2,
    min_target_pct=0.02,
    min_delta_pct=0.02,
):
    tmp = "__tmp_binary__"
    adata.obs[tmp] = "Rest"
    adata.obs.loc[adata.obs[group_col].astype(str) == str(cluster_name), tmp] = "Target"

    sc.tl.rank_genes_groups(
        adata,
        groupby=tmp,
        groups=["Target"],
        reference="Rest",
        method="wilcoxon",
        use_raw=False,
    )

    de_df_raw = sc.get.rank_genes_groups_df(adata, group="Target").copy()

    target_mask = adata.obs[tmp] == "Target"
    rest_mask = adata.obs[tmp] == "Rest"

    X = adata.X
    if issparse(X):
        X_target = X[target_mask.values, :]
        X_rest = X[rest_mask.values, :]
        target_mean = np.asarray(X_target.mean(axis=0)).ravel()
        rest_mean = np.asarray(X_rest.mean(axis=0)).ravel()
        target_pct = np.asarray((X_target > 0).mean(axis=0)).ravel()
        rest_pct = np.asarray((X_rest > 0).mean(axis=0)).ravel()
    else:
        X_target = X[target_mask.values, :]
        X_rest = X[rest_mask.values, :]
        target_mean = X_target.mean(axis=0)
        rest_mean = X_rest.mean(axis=0)
        target_pct = (X_target > 0).mean(axis=0)
        rest_pct = (X_rest > 0).mean(axis=0)

    expr_stats = pd.DataFrame(
        {
            "names": adata.var_names.astype(str),
            "target_mean": target_mean,
            "rest_mean": rest_mean,
            "target_pct": target_pct,
            "rest_pct": rest_pct,
        }
    )

    de_df = de_df_raw.merge(expr_stats, on="names", how="left")
    de_df["delta_mean"] = de_df["target_mean"] - de_df["rest_mean"]
    de_df["delta_pct"] = de_df["target_pct"] - de_df["rest_pct"]

    eff_mask = (
        (de_df["target_mean"] >= min_target_mean)
        & (de_df["delta_mean"] >= min_delta_mean)
        & (de_df["target_pct"] >= min_target_pct)
        & (de_df["delta_pct"] >= min_delta_pct)
    )

    if "logfoldchanges" in de_df.columns:
        eff_mask &= de_df["logfoldchanges"].fillna(0) >= min_logfc

    de_df_filtered = de_df.loc[eff_mask].copy()

    adata.obs.drop(columns=[tmp], inplace=True, errors="ignore")
    return de_df_raw, de_df_filtered


def ensure_hvgs(adata) -> None:
    if "highly_variable" not in adata.var.columns:
        if "counts" in adata.layers:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=2000,
                flavor="seurat_v3",
                layer="counts",
                subset=False,
            )
        else:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=2000,
                flavor="seurat",
                subset=False,
            )


def ensure_global_gini_stats(curator, adata, mean_floor=0.01, gini_q_low=0.01, gini_q_high=0.90, low_gini_cap=0.15):
    if curator.masker is None:
        curator.set_global_context(adata)

    if getattr(curator.masker, "gene_stats", None) is None:
        curator.masker.calculate_gene_stats()

    gs = curator.masker.gene_stats.copy()

    if "gene" in gs.columns:
        gs = gs.set_index("gene", drop=False)
    elif "names" in gs.columns:
        gs = gs.set_index("names", drop=False)

    gs.index = gs.index.astype(str)

    if "gini" not in gs.columns or "mean" not in gs.columns:
        raise ValueError("gene_stats must contain 'gini' and 'mean' columns.")

    gs = gs.replace([np.inf, -np.inf], np.nan)
    gs = gs.dropna(subset=["gini", "mean"])

    valid_low = gs.loc[gs["mean"] >= mean_floor, "gini"].dropna()
    if len(valid_low) == 0:
        low_thr = low_gini_cap
    else:
        low_thr = min(float(np.quantile(valid_low, gini_q_low)), low_gini_cap)

    valid_high = gs["gini"].dropna()
    if len(valid_high) == 0:
        high_thr = np.nan
    else:
        high_thr = float(np.quantile(valid_high, gini_q_high))

    return gs, low_thr, high_thr


def is_low_gini(gene, gene_stats, low_gini_thr) -> bool:
    g = str(gene)
    if g not in gene_stats.index:
        return False
    val = gene_stats.at[g, "gini"]
    if pd.isna(val):
        return False
    return float(val) <= low_gini_thr


def is_high_gini(gene, gene_stats, high_gini_thr) -> bool:
    g = str(gene)
    if g not in gene_stats.index or pd.isna(high_gini_thr):
        return False
    val = gene_stats.at[g, "gini"]
    if pd.isna(val):
        return False
    return float(val) >= high_gini_thr


def select_full_core_like_genes(
    de_df_raw,
    de_df_filtered,
    gt_label,
    gene_stats,
    low_gini_thr,
    high_gini_thr,
    n_top=50,
    oversample=1000,
    disabled_components=None,
):
    """
    Backend-free diagnostic selector that mirrors the major feature-level operations
    bundled in full_core. It is used only for leave-one-component-out diagnostics.
    """
    disabled_components = set(disabled_components or [])

    raw_pool = de_df_raw["names"].astype(str).head(oversample).tolist()
    if de_df_filtered is not None and not de_df_filtered.empty:
        filtered_pool = de_df_filtered["names"].astype(str).head(oversample).tolist()
    else:
        filtered_pool = raw_pool

    pool = list(filtered_pool)

    # High-Gini rare-marker rescue / candidate augmentation.
    if "high_gini_rescue" not in disabled_components:
        high_gini_candidates = [
            g for g in raw_pool
            if is_high_gini(g, gene_stats, high_gini_thr)
        ]
        pool = ordered_unique(pool + high_gini_candidates)

    canonical_markers = set(CD8_MARKER_DB.get(str(gt_label), set()))
    sentinel_candidates = [
        g for g in raw_pool
        if g in canonical_markers
    ]

    selected = []

    for g in pool:
        if is_regex_noise_gene(g):
            continue

        if "curated_noise_mask" not in disabled_components:
            if is_curated_noise_gene(g):
                continue

        if "low_gini_suppression" not in disabled_components:
            if is_low_gini(g, gene_stats, low_gini_thr):
                continue

        if "cross_lineage_filter" not in disabled_components:
            if is_cross_lineage_gene(g):
                continue

        selected.append(g)

    # Sentinel retention is applied after masking so that canonical markers can be
    # protected from over-filtering, especially for functional states.
    if "sentinel_retention" not in disabled_components:
        selected = ordered_unique(sentinel_candidates + selected)

    # Conservative fill to keep list length comparable.
    fill_pool = []
    for g in raw_pool:
        if is_regex_noise_gene(g):
            continue
        if "cross_lineage_filter" not in disabled_components and is_cross_lineage_gene(g):
            continue
        fill_pool.append(g)

    selected = ordered_unique(selected + fill_pool)
    return selected[:n_top]


def fraction_or_nan(values):
    values = list(values)
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def metric_noise_fraction(genes):
    return fraction_or_nan([is_any_noise_gene(g) for g in genes])


def metric_low_gini_fraction(genes, gene_stats, low_gini_thr):
    return fraction_or_nan([is_low_gini(g, gene_stats, low_gini_thr) for g in genes])


def metric_high_gini_fraction(genes, gene_stats, high_gini_thr):
    return fraction_or_nan([is_high_gini(g, gene_stats, high_gini_thr) for g in genes])


def metric_canonical_recall(genes, gt_label, var_names):
    canonical = set(CD8_MARKER_DB.get(str(gt_label), set()))
    canonical = canonical.intersection(set(map(str, var_names)))
    if len(canonical) == 0:
        return np.nan
    return float(len(set(genes).intersection(canonical)) / len(canonical))


def metric_overlap_with_full_core(genes, full_core_genes, n_top=50):
    if len(full_core_genes) == 0:
        return np.nan
    return float(len(set(genes).intersection(set(full_core_genes))) / min(n_top, len(full_core_genes)))


def metric_jaccard_with_full_core(genes, full_core_genes):
    a = set(genes)
    b = set(full_core_genes)
    if len(a | b) == 0:
        return np.nan
    return float(len(a & b) / len(a | b))


def write_tables_to_excel(letter_xlsx: Path, summary: pd.DataFrame, cluster_metrics: pd.DataFrame) -> None:
    if not letter_xlsx.exists():
        raise FileNotFoundError(f"LetterTables file not found: {letter_xlsx}")

    with pd.ExcelWriter(
        letter_xlsx,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        summary.to_excel(writer, sheet_name="L9_component_ablation_summary", index=False)
        cluster_metrics.to_excel(writer, sheet_name="L9_component_ablation_cluster_metrics", index=False)


def make_summary(cluster_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Biological_Noise_Fraction",
        "Low_Gini_Fraction",
        "High_Gini_Fraction",
        "Canonical_Marker_Recall",
        "Overlap_With_Full_Core",
        "Jaccard_With_Full_Core",
        "N_Genes",
    ]

    rows = []
    for variant in VARIANT_ORDER:
        sub = cluster_metrics[cluster_metrics["Variant"] == variant].copy()
        if sub.empty:
            continue

        row = {
            "Dataset": "CD8 T",
            "Variant": variant,
            "Display_Label": VARIANT_LABELS[variant].replace("\n", " "),
            "Component_Removed": COMPONENT_REMOVED[variant],
            "N_Clusters": int(sub["Cluster_ID"].nunique()),
        }
        for col in metric_cols:
            row[col] = float(pd.to_numeric(sub[col], errors="coerce").mean())

        rows.append(row)

    summary = pd.DataFrame(rows)

    for col in [
        "Biological_Noise_Fraction",
        "Low_Gini_Fraction",
        "High_Gini_Fraction",
        "Canonical_Marker_Recall",
        "Overlap_With_Full_Core",
        "Jaccard_With_Full_Core",
    ]:
        summary[col.replace("_Fraction", "_Pct").replace("_Recall", "_Recall_Pct").replace("Overlap_With_Full_Core", "Overlap_With_Full_Core_Pct").replace("Jaccard_With_Full_Core", "Jaccard_With_Full_Core_Pct")] = summary[col] * 100

    return summary



def _prepare_L9_summary_for_plot(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["Variant"] = pd.Categorical(summary["Variant"], VARIANT_ORDER, ordered=True)
    summary = summary.sort_values("Variant")
    return summary


def _metric_ylim(metric: str, vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0

    if metric in ["Biological_Noise_Fraction", "Low_Gini_Fraction"]:
        return 0.0, max(2.0, vmax + 0.8)
    if metric == "Canonical_Marker_Recall":
        return 0.0, 85.0
    return 0.0, 108.0


def _plot_L9_metric_panel(
    ax,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    panel_label: str | None = None,
) -> None:
    x = np.arange(len(summary))
    labels = [VARIANT_LABELS[str(v)] for v in summary["Variant"]]
    colors = [
        FULL_CORE_COLOR if str(v) == "full_core" else LOO_COLOR
        for v in summary["Variant"]
    ]
    vals = summary[metric].astype(float).values * 100

    ymin, ymax = _metric_ylim(metric, vals)
    label_pad = max((ymax - ymin) * 0.025, 0.05)

    ax.bar(
        x,
        vals,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
    )

    for i, v in enumerate(vals):
        ax.text(
            i,
            v + label_pad,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if panel_label:
        ax.text(
            -0.12,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="left",
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Genes (%)")
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, outdir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_component_ablation(summary: pd.DataFrame, outdir: Path) -> None:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = _prepare_L9_summary_for_plot(summary)

    panels = [
        (
            "a",
            "Biological_Noise_Fraction",
            "Biological-noise\nfraction (%)",
            "ReviewerFig_L9a_biological_noise_fraction",
        ),
        (
            "b",
            "Low_Gini_Fraction",
            "Low-Gini\nfraction (%)",
            "ReviewerFig_L9b_low_gini_fraction",
        ),
        (
            "c",
            "Canonical_Marker_Recall",
            "Canonical marker\nrecall (%)",
            "ReviewerFig_L9c_canonical_marker_recall",
        ),
        (
            "d",
            "Overlap_With_Full_Core",
            "Overlap with\nfull_core (%)",
            "ReviewerFig_L9d_overlap_with_full_core",
        ),
    ]

    # -------------------------------------------------------------------------
    # Combined 2x2 reviewer-facing figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=300)
    axes = axes.ravel()

    for ax, (panel_label, metric, title, _) in zip(axes, panels):
        _plot_L9_metric_panel(
            ax=ax,
            summary=summary,
            metric=metric,
            title=title,
            panel_label=panel_label,
        )

    fig.suptitle(
        "Leave-one-component-out feature-level diagnostic of the full_core module",
        fontsize=16,
        y=1.02,
    )

    _save_figure(
        fig,
        outdir,
        "ReviewerFig_L9_component_ablation_feature_diagnostics",
    )

    # -------------------------------------------------------------------------
    # Individual panel figures
    # -------------------------------------------------------------------------
    for panel_label, metric, title, stem in panels:
        fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
        _plot_L9_metric_panel(
            ax=ax,
            summary=summary,
            metric=metric,
            title=title,
            panel_label=panel_label,
        )
        _save_figure(fig, outdir, stem)

    print("Wrote L9 combined and individual panel figures:")
    print(f"  {outdir / 'ReviewerFig_L9_component_ablation_feature_diagnostics.pdf'}")
    print(f"  {outdir / 'ReviewerFig_L9_component_ablation_feature_diagnostics.png'}")
    for _, _, _, stem in panels:
        print(f"  {outdir / (stem + '.pdf')}")
        print(f"  {outdir / (stem + '.png')}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--letter-xlsx", default="paper/revision/LetterTables.xlsx")
    parser.add_argument("--csv-outdir", default="paper/revision_tables")
    parser.add_argument("--fig-outdir", default="paper/revision_figures")
    parser.add_argument("--group-col", default="meta.cluster")
    parser.add_argument("--n-top", type=int, default=50)
    parser.add_argument("--oversample", type=int, default=1000)
    parser.add_argument("--round", type=int, default=4)
    args = parser.parse_args()

    input_h5ad = Path(args.input_h5ad) if args.input_h5ad else find_default_input()
    letter_xlsx = Path(args.letter_xlsx)
    csv_outdir = Path(args.csv_outdir)
    fig_outdir = Path(args.fig_outdir)

    csv_outdir.mkdir(parents=True, exist_ok=True)
    fig_outdir.mkdir(parents=True, exist_ok=True)

    print(f"Input h5ad: {input_h5ad}")
    print(f"LetterTables: {letter_xlsx}")

    adata = sc.read_h5ad(input_h5ad)
    ensure_hvgs(adata)

    if args.group_col not in adata.obs.columns:
        raise ValueError(f"group_col not found in adata.obs: {args.group_col}")

    adata.obs["Ground_Truth"] = adata.obs[args.group_col].astype(str).apply(get_cd8_ground_truth)

    curator = LLMscCurator(backend=NullBackend())
    curator.set_global_context(adata)
    gene_stats, low_gini_thr, high_gini_thr = ensure_global_gini_stats(curator, adata)

    print(f"LOW_GINI_THR : {low_gini_thr:.4f}")
    print(f"HIGH_GINI_THR: {high_gini_thr:.4f}")

    exclude_gt = {"CD8_Other", "Other", "Unknown"}
    cluster_list = sorted(adata.obs[args.group_col].astype(str).unique())
    cluster_list = [
        c for c in cluster_list
        if get_cd8_ground_truth(c) not in exclude_gt
    ]

    print(f"N clusters: {len(cluster_list)}")

    rows = []
    gene_rows = []

    disabled_by_variant = {
        "full_core": set(),
        "minus_curated_noise_mask": {"curated_noise_mask"},
        "minus_low_gini_suppression": {"low_gini_suppression"},
        "minus_high_gini_rescue": {"high_gini_rescue"},
        "minus_sentinel_retention": {"sentinel_retention"},
        "minus_cross_lineage_filter": {"cross_lineage_filter"},
    }

    for i, cluster_id in enumerate(cluster_list, start=1):
        gt = get_cd8_ground_truth(cluster_id)
        print(f"[{i}/{len(cluster_list)}] {cluster_id} -> {gt}")

        de_df_raw, de_df_filtered = build_de_table(
            adata=adata,
            group_col=args.group_col,
            cluster_name=cluster_id,
        )

        genes_by_variant = {}
        for variant in VARIANT_ORDER:
            genes = select_full_core_like_genes(
                de_df_raw=de_df_raw,
                de_df_filtered=de_df_filtered,
                gt_label=gt,
                gene_stats=gene_stats,
                low_gini_thr=low_gini_thr,
                high_gini_thr=high_gini_thr,
                n_top=args.n_top,
                oversample=args.oversample,
                disabled_components=disabled_by_variant[variant],
            )
            genes_by_variant[variant] = genes

        full_core_genes = genes_by_variant["full_core"]

        for variant in VARIANT_ORDER:
            genes = genes_by_variant[variant]

            rows.append(
                {
                    "Dataset": "CD8 T",
                    "Cluster_ID": cluster_id,
                    "Ground_Truth": gt,
                    "Variant": variant,
                    "Display_Label": VARIANT_LABELS[variant].replace("\n", " "),
                    "Component_Removed": COMPONENT_REMOVED[variant],
                    "N_Genes": len(genes),
                    "Biological_Noise_Fraction": metric_noise_fraction(genes),
                    "Low_Gini_Fraction": metric_low_gini_fraction(genes, gene_stats, low_gini_thr),
                    "High_Gini_Fraction": metric_high_gini_fraction(genes, gene_stats, high_gini_thr),
                    "Canonical_Marker_Recall": metric_canonical_recall(genes, gt, adata.var_names),
                    "Overlap_With_Full_Core": metric_overlap_with_full_core(genes, full_core_genes, n_top=args.n_top),
                    "Jaccard_With_Full_Core": metric_jaccard_with_full_core(genes, full_core_genes),
                    "Genes": ";".join(genes),
                }
            )

            for rank, gene in enumerate(genes, start=1):
                gene_rows.append(
                    {
                        "Dataset": "CD8 T",
                        "Cluster_ID": cluster_id,
                        "Ground_Truth": gt,
                        "Variant": variant,
                        "Rank": rank,
                        "Gene": gene,
                        "Is_Regex_Noise": is_regex_noise_gene(gene),
                        "Is_Curated_Noise": is_curated_noise_gene(gene),
                        "Is_Any_Noise": is_any_noise_gene(gene),
                        "Is_Low_Gini": is_low_gini(gene, gene_stats, low_gini_thr),
                        "Is_High_Gini": is_high_gini(gene, gene_stats, high_gini_thr),
                        "Is_Cross_Lineage": is_cross_lineage_gene(gene),
                        "Is_Canonical_Marker": gene in CD8_MARKER_DB.get(gt, set()),
                        "Gini": float(gene_stats.at[gene, "gini"]) if gene in gene_stats.index and pd.notna(gene_stats.at[gene, "gini"]) else np.nan,
                    }
                )

    cluster_metrics = pd.DataFrame(rows)
    gene_level = pd.DataFrame(gene_rows)
    summary = make_summary(cluster_metrics)

    float_cols_summary = summary.select_dtypes(include=["float"]).columns
    float_cols_cluster = cluster_metrics.select_dtypes(include=["float"]).columns
    float_cols_gene = gene_level.select_dtypes(include=["float"]).columns

    summary[float_cols_summary] = summary[float_cols_summary].round(args.round)
    cluster_metrics[float_cols_cluster] = cluster_metrics[float_cols_cluster].round(args.round)
    gene_level[float_cols_gene] = gene_level[float_cols_gene].round(args.round)


    cluster_metrics.to_csv(csv_outdir / "L9_component_ablation_cluster_metrics.csv", index=False)
    gene_level.to_csv(csv_outdir / "L9_component_ablation_gene_level.csv", index=False)
    summary.to_csv(csv_outdir / "L10_component_ablation_summary.csv", index=False)

    write_tables_to_excel(letter_xlsx, summary, cluster_metrics)
    plot_component_ablation(summary, fig_outdir)

    print("Wrote:")
    print(f"  {csv_outdir / 'L9_component_ablation_cluster_metrics.csv'}")
    print(f"  {csv_outdir / 'L9_component_ablation_gene_level.csv'}")
        print(f"  {csv_outdir / 'L10_component_ablation_summary.csv'}")
    print(f"  {letter_xlsx} sheets: L10_component_ablation_summary, L9_component_ablation_cluster_metrics")
    print(f"  {fig_outdir / 'ReviewerFig_L9_component_ablation_feature_diagnostics.pdf'}")
    print(f"  {fig_outdir / 'ReviewerFig_L9_component_ablation_feature_diagnostics.png'}")


if __name__ == "__main__":
    main()
