#!/usr/bin/env python3
"""Render Figure 3a–c and Source Data from public primary-benchmark metrics.

No LLM calls or label parsing are performed here. Generate both input CSVs
with paper/revision_metrics/make_l4_l5_letter_tables.py before running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

DATASETS = ["CD8 T", "CD4 T", "MSC", "Mouse B"]
COUNTS = dict(zip(DATASETS, [17, 22, 8, 5]))
METHODS = ["Standard", "Curated", "CellTypist", "SingleR", "Azimuth"]
LABELS = ["Standard", "Full pipeline", "CellTypist", "SingleR", "Azimuth"]
COLORS = ["#42A5F5", "#D32F2F", "#66BB6A", "#7E57C2", "#FFB74D"]
METRICS = ["Mean_S_anno", "Exact_State_Agreement", "Exact_Match_Accuracy",
           "Major_Lineage_Accuracy", "Ontology_Consistent_Accuracy_Sanno_ge_0.5",
           "Low_Consistency_Rate_Sanno_lt_0.5"]
MISSING = {("Mouse B", "CellTypist"), ("Mouse B", "Azimuth")}


def bootstrap_ci(values, n_boot=20000, seed=42):
    """Percentile 95% CI of a mean; resample clusters within each method/task."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires nonempty, finite cluster scores")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        size = min(1000, n_boot - start)
        means[start:start + size] = rng.choice(values, (size, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def build_source_data(metrics_csv, audit_csv, n_boot=20000, seed=42):
    l4 = pd.read_csv(metrics_csv)
    audit = pd.read_csv(audit_csv, keep_default_na=False)
    required = {"Dataset", "Method", "N", *METRICS}
    if not required.issubset(l4.columns):
        raise ValueError(f"L4 missing columns: {sorted(required - set(l4.columns))}")
    required = {"Dataset", "Method", "Cluster_ID", "GT_Major", "GT_State",
                "Pred_Major", "Pred_State", "Public_Sanno", "Recorded_Sanno"}
    if not required.issubset(audit.columns):
        raise ValueError(f"Audit missing columns: {sorted(required - set(audit.columns))}")
    if l4.duplicated(["Dataset", "Method"]).any() or audit.duplicated(["Dataset", "Method", "Cluster_ID"]).any():
        raise ValueError("Duplicate dataset/method or cluster records")
    expected = {(d, m) for d in DATASETS for m in METHODS} - MISSING
    for name, df in [("L4", l4), ("audit", audit)]:
        observed = set(zip(df.Dataset, df.Method))
        if observed != expected:
            raise ValueError(f"Unexpected {name} groups: missing={expected-observed}, extra={observed-expected}")
    audit["Public_Sanno"] = pd.to_numeric(audit.Public_Sanno, errors="raise")
    audit["Recorded_Sanno"] = pd.to_numeric(audit.Recorded_Sanno, errors="raise")
    scores = audit.Public_Sanno.to_numpy()
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Invalid audit scores")
    if not np.allclose(scores, audit.Recorded_Sanno, rtol=0, atol=1e-8):
        raise ValueError("Public and recorded Sanno differ")
    lookup = l4.set_index(["Dataset", "Method"])
    rows = []
    for dataset in DATASETS:
        reference_ids = set(audit.loc[(audit.Dataset == dataset) & (audit.Method == "Standard"), "Cluster_ID"])
        for method, label in zip(METHODS, LABELS):
            row = {"Dataset": dataset, "Method": method, "Display_Method": label}
            if (dataset, method) in MISSING:
                row.update({"Evaluated": False, "N": pd.NA})
                row.update({key: np.nan for key in METRICS + ["CI_Low", "CI_High"]})
            else:
                group = audit[(audit.Dataset == dataset) & (audit.Method == method)]
                if len(group) != COUNTS[dataset] or set(group.Cluster_ID) != reference_ids:
                    raise ValueError(f"Incomplete cluster coverage: {dataset}/{method}")
                score = group.Public_Sanno
                major = group.GT_Major.eq(group.Pred_Major)
                state = group.GT_State.eq(group.Pred_State)
                calculated = dict(zip(METRICS, [score.mean(), state.mean(), (major & state).mean(),
                                                major.mean(), score.ge(0.5).mean(), score.lt(0.5).mean()]))
                recorded = lookup.loc[(dataset, method)]
                if recorded.N != len(group):
                    raise ValueError(f"L4 N mismatch: {dataset}/{method}")
                for key, value in calculated.items():
                    if not np.isclose(float(recorded[key]), value, rtol=0, atol=1e-10):
                        raise ValueError(f"L4/audit mismatch: {dataset}/{method}/{key}; use full-precision CSVs")
                lo, hi = bootstrap_ci(score, n_boot, seed)
                row.update({"Evaluated": True, "N": len(group), **{key: float(recorded[key]) for key in METRICS},
                            "CI_Low": float(lo), "CI_High": float(hi)})
            rows.append(row)
    result = pd.DataFrame(rows)
    result["N"] = result.N.astype("Int64")
    return result


def heatmap(ax, source, metric, panel_label):
    matrix = (
        source.pivot(index="Dataset", columns="Method", values=metric)
        .reindex(index=DATASETS, columns=METHODS)
        * 100
    )
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")

    image = ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy()),
        vmin=0,
        vmax=100,
        cmap=cmap,
        aspect="auto",
    )

    ax.set_xticks(range(5), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(4), ["CD8", "CD4", "MSC", "Mouse B"])
    # ax.set_title(panel_label, loc="left", fontsize=10, fontweight="bold")

    for i in range(4):
        for j in range(5):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                "NA" if pd.isna(value) else f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if pd.notna(value) and value >= 60 else "black",
                fontsize=9,
            )

    return image


def render(source, outdir):
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    })

    # 縦方向を少し広げる
    fig = plt.figure(figsize=(12.1, 6.0), layout="constrained")

    # hspaceで棒グラフとヒートマップの間隔を明示的に確保
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.15, 0.85],
        hspace=0.22,
    )
    top_axes = []
    for index, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(grid[0, index])
        top_axes.append(ax)
        sub = source[source.Dataset.eq(dataset)].set_index("Method")
        for x, method in enumerate(METHODS):
            row = sub.loc[method]
            if not row.Evaluated:
                ax.text(x, 3, "NA", ha="center")
                continue
            mean, lo, hi = 100 * row.Mean_S_anno, 100 * row.CI_Low, 100 * row.CI_High
            ax.bar(x, mean, yerr=[[mean-lo], [hi-mean]], color=COLORS[x], edgecolor="black", linewidth=.5, capsize=2.5)
        ax.set(xlim=(-.6, 4.6), ylim=(0, 105), xticks=[], yticks=range(0, 101, 20), xlabel=["CD8", "CD4", "MSC", "Mouse B"][index])
        if index == 0:
            ax.set_ylabel(r"Mean $S_{anno}$ (%)")
            # ax.set_title("a", loc="left", fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
    ax_b, ax_c = fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:])

    heatmap(
        ax_b,
        source,
        METRICS[4],
        "b",
    )
    image = heatmap(
        ax_c,
        source,
        "Major_Lineage_Accuracy",
        "c",
    )

    fig.colorbar(
        image,
        ax=[ax_b, ax_c],
        shrink=.85,
        pad=.015,
        label="Accuracy (%)",
    )
    # Reserve a title line for the common method legend.
    fig.suptitle(" ", fontsize=20)
    fig.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS, LABELS)],
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=11,       
        handlelength=1.8,
        handleheight=1.2,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    fig.savefig(outdir / "Fig3a_c.pdf")
    fig.savefig(outdir / "Fig3a_c.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("paper/revision_figures/Fig3"))
    parser.add_argument("--source-data-dir", type=Path, default=Path("paper/source_data/figure_data/Fig3"))
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    source = build_source_data(args.metrics_csv, args.audit_csv, args.n_bootstrap, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.source_data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.source_data_dir / "inputs"
    snapshot_dir.mkdir(exist_ok=True)
    for original, name in [(args.metrics_csv, "L4_complementary_metrics.csv"),
                           (args.audit_csv, "L4_prediction_audit.csv")]:
        destination = snapshot_dir / name
        if original.resolve() != destination.resolve():
            shutil.copyfile(original, destination)
    source.to_csv(args.source_data_dir / "Fig3abc_data.csv", index=False, na_rep="NA")
    for panel, column in [("a", "Mean_S_anno"), ("b", METRICS[4]), ("c", "Major_Lineage_Accuracy")]:
        columns = ["Dataset", "Method", "Display_Method", "Evaluated", "N", column]
        if panel == "a":
            columns += ["CI_Low", "CI_High"]
        source[columns].to_csv(args.source_data_dir / f"Fig3{panel}_data.csv", index=False, na_rep="NA")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"inputs": {p.name: digest(p) for p in [args.metrics_csv, args.audit_csv]},
                "script_sha256": digest(Path(__file__)), "bootstrap_resamples": args.n_bootstrap,
                "seed": args.seed, "ci_method": "percentile 95%; clusters resampled within each dataset and method",
                "units": "Source Data fractions (0–1); plotted percentages (0–100)",
                "n_clusters": sum(COUNTS.values()), "n_predictions": int(source.N.sum()),
                "numpy_version": np.__version__, "pandas_version": pd.__version__,
                "matplotlib_version": matplotlib.__version__}
    (args.source_data_dir / "Fig3_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    render(source, args.outdir)
    print(f"Validated 52 clusters and {source.N.sum()} predictions; wrote Figure 3a–c to {args.outdir}")
    print(f"Source Data: {args.source_data_dir}; bootstrap={args.n_bootstrap}, seed={args.seed}")


if __name__ == "__main__":
    main()
