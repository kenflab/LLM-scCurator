#!/usr/bin/env python3
"""Production-path CD8 component ablation. No model calls are made."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

# Always use the checkout containing this script, not an older installed package.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
import scanpy as sc
from llm_sc_curator import LLMscCurator
from llm_sc_curator.backends import BaseLLMBackend
from llm_sc_curator.noise_lists import NOISE_LISTS, NOISE_PATTERNS

COMPONENTS = {
    "curated_noise_mask": "Curated list-based masking",
    "low_gini_suppression": "Low-Gini suppression",
    "high_gini_rescue": "High-Gini candidate augmentation",
    "sentinel_retention": "Canonical/sentinel retention",
    "cross_lineage_filter": "Expression-based cross-lineage filtering",
}

# Evaluation reference only. Never supplied to curate_features().
CD8_MARKER_DB = {
    "CD8_Naive": {"IL7R", "CCR7", "LEF1", "TCF7", "SELL", "MAL", "LTB", "KLF2"},
    "CD8_EffectorMemory": {"GZMK", "LTB", "AQP3", "IL7R", "CXCR4", "ANXA1", "ZFP36L2", "DUSP2"},
    "CD8_Effector": {"CCL5", "NKG7", "PRF1", "GZMB", "CX3CR1", "KLRG1", "FGFBP2", "GNLY"},
    "CD8_Exhausted": {"CXCL13", "CTLA4", "TIGIT", "HAVCR2", "PDCD1", "ENTPD1", "TOX", "TNFRSF9"},
    "CD8_ISG": {"IFIT1", "ISG15", "MX1", "STAT1", "OAS1", "IFI6"},
    "CD8_MAIT": {"SLC4A10", "KLRB1", "CXCR6"},
    "CD8_Cycling": {"MKI67", "TOP2A", "CDK1", "BIRC5", "PCNA", "TYMS"},
    "CD8_NK_Killer": {"NKG7", "GNLY", "PRF1", "GZMB", "FGFBP2"},
}


class NoInferenceBackend(BaseLLMBackend):
    def generate(self, prompt, json_mode=False):
        raise RuntimeError("This analysis performs marker selection only.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_input():
    for name in ("paper/gb_resubmission/input/cd8_benchmark_data.h5ad",
                 "gb_resubmission/input/cd8_benchmark_data.h5ad",
                 "input/cd8_benchmark_data.h5ad", "../input/cd8_benchmark_data.h5ad"):
        path = REPO / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Provide --input-h5ad with the existing CD8 benchmark H5AD path.")


def prepare_selection_data(adata, group_col, coarse_col=None, batch_key=None):
    """Remove annotation columns and expose opaque cluster IDs to selection."""
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique.")
    if group_col not in adata.obs or adata.obs[group_col].isna().any():
        raise ValueError(f"Missing or incomplete cluster column: {group_col}")
    if coarse_col and (coarse_col == group_col or re.search(r"ground.?truth|fine.?state|gt_label", coarse_col, re.I)):
        raise ValueError("--coarse-col must identify independently supplied broad context, not the scored state.")
    if batch_key and (batch_key == group_col or re.search(r"ground.?truth|fine.?state|gt_label", batch_key, re.I)):
        raise ValueError("--batch-key must identify technical batches, not benchmark labels.")
    if "curation_cluster" in (coarse_col, batch_key):
        raise ValueError("curation_cluster is reserved for opaque cluster identifiers.")
    original = adata.obs[group_col].astype(str)
    cluster_ids = sorted(original.unique())
    codes = {name: f"cluster_{i:03d}" for i, name in enumerate(cluster_ids)}
    columns = {}
    for name in (coarse_col, batch_key):
        if name:
            if name not in adata.obs or adata.obs[name].isna().any():
                raise ValueError(f"Missing or incomplete metadata column: {name}")
            columns[name] = adata.obs[name].copy()
    columns["curation_cluster"] = original.map(codes).astype("category")
    adata.obs = pd.DataFrame(columns, index=adata.obs_names)
    adata.raw = None
    adata.uns = {k: v for k, v in adata.uns.items() if k == "log1p"}
    return adata, codes


def prepare_context(context, coarse_col):
    if not context.var_names.is_unique or not context.obs_names.is_unique:
        raise ValueError("Global context cell and gene identifiers must be unique.")
    if coarse_col not in context.obs or context.obs[coarse_col].isna().any():
        raise ValueError(f"Global context must contain complete broad labels in {coarse_col!r}.")
    context.obs = context.obs[[coarse_col]].copy()
    context.raw = None
    context.uns = {k: v for k, v in context.uns.items() if k == "log1p"}
    return context


def cross_lineage_status(local, context, coarse_col):
    if not coarse_col:
        return "not_assessed_no_broad_context"
    counts = context.obs[coarse_col].value_counts()
    for _, cells in local.obs.groupby("curation_cluster", observed=True):
        labels = cells[coarse_col].astype(str).unique()
        if len(labels) != 1:
            raise ValueError("Each target cluster must have one supplied broad context label.")
        if labels[0] not in set(counts.index.astype(str)):
            raise ValueError("Target broad label is absent from the global context.")
        if not any(str(k) != labels[0] and n >= 10 for k, n in counts.items()):
            return "not_assessed_no_eligible_contrasting_lineage"
    return "assessed"


def select_lists(adata, context, codes, n_top, n_candidates, coarse_col=None, batch_key=None):
    """Selection phase: neither reference labels nor reference marker sets enter."""
    curator = LLMscCurator(backend=NoInferenceBackend())
    curator.set_global_context(context)
    curator.masker.calculate_gene_stats()
    cross_status = cross_lineage_status(adata, context, coarse_col)
    effective_coarse = coarse_col if cross_status == "assessed" else None
    variants = {"full_core": ()}
    variants.update({f"minus_{c}": (c,) for c in COMPONENTS
                     if c != "cross_lineage_filter" or cross_status == "assessed"})
    selected = {}
    for index, (original_id, opaque_id) in enumerate(codes.items(), 1):
        print(f"[{index}/{len(codes)}] {original_id}", flush=True)
        for variant, disabled in variants.items():
            genes = curator.curate_features(
                target_adata=adata, group_col="curation_cluster", target_group=opaque_id,
                n_top=n_top, n_candidates=n_candidates, use_hvg=True, use_statistics=True,
                coarse_col=effective_coarse, batch_key=batch_key,
                disabled_components=disabled,
            )
            if not genes or len(genes) > n_top or len(genes) != len(set(genes)):
                raise ValueError(f"Invalid selected gene list: {original_id}/{variant}")
            selected[(original_id, variant)] = list(genes)
    return selected, curator, cross_status


def evaluate_lists(selected, curator, var_names, ground_truth_by_cluster):
    """Evaluation phase: reference categories enter only after lists are frozen."""
    stats = curator.masker.gene_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["gini", "mean"])
    eligible = stats.loc[stats["mean"] >= 0.01, "gini"]
    low_cut = min(float(eligible.quantile(0.01)), 0.15) if len(eligible) else -1.0
    high_cut = float(stats["gini"].quantile(0.90))
    low_genes = set(stats.index[(stats["mean"] >= 0.01) & (stats["gini"] < low_cut)])
    high_genes = set(stats.index[stats["gini"] >= high_cut])
    rescue_candidates = curator._get_high_gini_genes()
    noise_lists = set().union(*[set(g) for g in NOISE_LISTS.values()])
    patterns = [re.compile(p) for p in NOISE_PATTERNS.values()]
    rows, gene_rows = [], []
    for (cluster, variant), genes in selected.items():
        gt = ground_truth_by_cluster[cluster]
        if gt not in CD8_MARKER_DB:
            raise ValueError(f"No canonical evaluation set for {cluster}: {gt}")
        canonical = CD8_MARKER_DB[gt] & set(var_names)
        gene_set = set(genes)
        baseline = set(selected[(cluster, "full_core")])
        flags = []
        for rank, gene in enumerate(genes, 1):
            regex_noise = any(p.match(gene) for p in patterns)
            item = {
                "Cluster_ID": cluster, "Ground_Truth": gt, "Variant": variant,
                "Rank": rank, "Gene": gene,
                "Is_Regex_Noise": bool(regex_noise), "Is_Curated_Noise": gene in noise_lists,
                "Is_Any_Noise": bool(regex_noise or gene in noise_lists),
                "Is_Low_Gini": gene in low_genes, "Is_High_Gini": gene in high_genes,
                "Is_High_Gini_Rescue_Candidate": gene in rescue_candidates,
                "Is_Canonical_Marker": gene in canonical,
                "Gini": float(stats.loc[gene, "gini"]) if gene in stats.index else np.nan,
            }
            flags.append(item)
            gene_rows.append(item)
        rows.append({
            "Cluster_ID": cluster, "Ground_Truth": gt, "Variant": variant,
            "N_Genes": len(genes), "N_Eligible_Canonical_Markers": len(canonical),
            "Biological_Noise_Fraction": np.mean([r["Is_Any_Noise"] for r in flags]),
            "Low_Gini_Fraction": np.mean([r["Is_Low_Gini"] for r in flags]),
            "High_Gini_Fraction": np.mean([r["Is_High_Gini"] for r in flags]),
            "High_Gini_Rescue_Candidate_Fraction": np.mean([r["Is_High_Gini_Rescue_Candidate"] for r in flags]),
            "Canonical_Marker_Recall": len(gene_set & canonical) / len(canonical) if canonical else np.nan,
            "Overlap_With_Full_Core": len(gene_set & baseline) / len(baseline),
            "Jaccard_With_Full_Core": len(gene_set & baseline) / len(gene_set | baseline),
            "Genes": ";".join(genes),
        })
    return pd.DataFrame(rows), pd.DataFrame(gene_rows), {"low_gini_cutoff": low_cut, "high_gini_cutoff": high_cut}


def summarize(metrics, cross_status):
    values = ["Biological_Noise_Fraction", "Low_Gini_Fraction", "High_Gini_Fraction",
              "High_Gini_Rescue_Candidate_Fraction", "Canonical_Marker_Recall",
              "Overlap_With_Full_Core", "Jaccard_With_Full_Core"]
    rows = []
    for variant, sub in metrics.groupby("Variant", sort=False):
        rows.append({"Variant": variant, "Status": "assessed", "N_Clusters": len(sub),
                     "N_Clusters_With_Canonical_Recall": int(sub.Canonical_Marker_Recall.notna().sum()),
                     "Min_N_Genes": int(sub.N_Genes.min()), "Max_N_Genes": int(sub.N_Genes.max()),
                     **{col: float(sub[col].mean()) for col in values}})
    if cross_status != "assessed":
        rows.append({"Variant": "minus_cross_lineage_filter", "Status": cross_status, "N_Clusters": 0})
    return pd.DataFrame(rows)


def plot_summary(summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["full_core", "− curated\nmask", "− low-Gini\nsuppression", "− high-Gini\nrescue",
              "− canonical/\nsentinel", "− cross-lineage\nfilter"]
    panels = [("Biological_Noise_Fraction", "Biological-noise fraction (%)"),
              ("Low_Gini_Fraction", "Low-Gini fraction (%)"),
              ("Canonical_Marker_Recall", "Canonical marker recall (%)"),
              ("Overlap_With_Full_Core", "Overlap with full_core (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for letter, ax, (metric, title) in zip("abcd", axes.flat, panels):
        values = summary[metric].to_numpy(float) * 100
        ax.bar(range(len(summary)), values, color=["#D62728"] + ["#BDBDBD"] * (len(summary) - 1))
        ax.set_xticks(range(len(summary)), labels, fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        finite = values[np.isfinite(values)]
        ax.set_ylim(0, max(1, float(finite.max()) * 1.15) if len(finite) else 1)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(i, 0, "N/A", ha="center", va="bottom", fontsize=8)
    fig.savefig(outdir / "Figure_L2.pdf")
    fig.savefig(outdir / "Figure_L2.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-h5ad", type=Path)
    p.add_argument("--group-col", default="meta.cluster")
    p.add_argument("--batch-key")
    p.add_argument("--global-h5ad", type=Path)
    p.add_argument("--coarse-col", help="Existing broad-context column; never a benchmark fine-state column")
    p.add_argument("--expected-clusters", type=int, default=17)
    p.add_argument("--n-top", type=int, default=50)
    p.add_argument("--n-candidates", type=int, default=500)
    p.add_argument("--outdir", type=Path, default=REPO / "paper/revision_tables/cd8_component_ablation_v2")
    args = p.parse_args()
    source = args.input_h5ad or default_input()
    if args.outdir.exists():
        p.error("Output directory exists; choose a new --outdir.")
    if args.global_h5ad and not args.coarse_col:
        p.error("--global-h5ad requires --coarse-col.")
    if min(args.n_top, args.n_candidates, args.expected_clusters) <= 0:
        p.error("Marker budgets and expected cluster count must be positive.")
    source = source.resolve()
    adata, codes = prepare_selection_data(sc.read_h5ad(source), args.group_col, args.coarse_col, args.batch_key)
    if len(codes) != args.expected_clusters:
        p.error(f"Found {len(codes)} clusters, expected {args.expected_clusters}; verify the frozen benchmark input.")
    context = prepare_context(sc.read_h5ad(args.global_h5ad), args.coarse_col) if args.global_h5ad else adata
    if not set(adata.var_names).issubset(set(context.var_names)):
        p.error("Global context must contain every gene in the target dataset.")
    args.outdir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=args.outdir / "selection.log", level=logging.INFO)
    selected, curator, cross_status = select_lists(
        adata, context, codes, args.n_top, args.n_candidates, args.coarse_col, args.batch_key,
    )
    # Freeze every list before importing or applying the ground-truth mapping.
    marker_file = args.outdir / "marker_inputs.csv"
    pd.DataFrame([{"Cluster_ID": c, "Variant": v, "N_Genes": len(g), "Genes": ";".join(g)}
                  for (c, v), g in selected.items()]).to_csv(marker_file, index=False)
    frozen_sha = sha256(marker_file)
    from benchmarks.gt_mappings import get_cd8_ground_truth
    ground_truth = {c: get_cd8_ground_truth(c) for c in codes}
    metrics, genes, cutoffs = evaluate_lists(selected, curator, adata.var_names, ground_truth)
    summary = summarize(metrics, cross_status)
    metrics.to_csv(args.outdir / "L9A_component_ablation_cluster_metrics.csv", index=False)
    genes.to_csv(args.outdir / "L9B_component_ablation_gene_records.csv", index=False)
    summary.to_csv(args.outdir / "L10_component_ablation_summary.csv", index=False)
    plot_summary(summary, args.outdir)
    assert frozen_sha == sha256(marker_file), "Evaluation modified the frozen marker file."
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_h5ad": str(source), "input_sha256": sha256(source),
        "global_h5ad": str(args.global_h5ad) if args.global_h5ad else None,
        "global_sha256": sha256(args.global_h5ad) if args.global_h5ad else None,
        "git_commit": git.stdout.strip() if git.returncode == 0 else "unavailable",
        "code_sha256": {str(path.relative_to(REPO)): sha256(path) for path in (
            Path(__file__), REPO / "llm_sc_curator/core.py", REPO / "llm_sc_curator/masking.py",
            REPO / "llm_sc_curator/noise_lists.py", REPO / "benchmarks/gt_mappings.py")},
        "scanpy_version": sc.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "selection_obs_columns": list(adata.obs.columns), "opaque_cluster_ids_used": True,
        "reference_labels_used_in_selection": False, "model_calls": 0,
        "marker_file_sha256": frozen_sha, "all_lists_frozen_before_evaluation": True,
        "n_clusters": len(codes), "n_assessed_variants": metrics.Variant.nunique(),
        "marker_budget": args.n_top, "candidate_budget": args.n_candidates,
        "group_col": args.group_col, "batch_key": args.batch_key,
        "coarse_col": args.coarse_col, "cross_lineage_status": cross_status,
        "global_context": curator._global_context_config,
        "definitions": {
            "canonical_recall": "Recovered reference markers / reference markers present in the input feature space; evaluation only",
            "fractions": "Per-cluster counts / actual selected list length; unweighted cluster means",
            "low_gini": "Production low-Gini eligibility: mean >= 0.01 and Gini < min(eligible 1st percentile, 0.15)",
            "high_gini": "Gini >= global 90th percentile; additional mean limits recorded in the rescue-candidate metric",
            "sentinel_ablation": "Disable lineage candidate augmentation, built-in lineage/proliferation whitelist, cell-cycle sentinel exemptions, and module sentinel rescue",
            "noise_categories": "Membership in regex or curated noise classes; protected sentinels may belong to these classes",
            "short_lists": "Production lists retained as returned; no diagnostic padding",
        }, **cutoffs,
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved: {args.outdir.resolve()}")
    print(f"Cross-lineage comparison: {cross_status}")


if __name__ == "__main__":
    main()
