#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


COLORS = {
    "Standard": "#5DA5DA",
    "LLM-scCurator": "#D62728",
}

STANDARD_COLOR = COLORS["Standard"]
CURATED_COLOR = COLORS["LLM-scCurator"]
NEUTRAL_COLOR = "#BDBDBD"
WORSE_COLOR = "#7A7A7A"
ZERO_COLOR = "#333333"
EDGE_COLOR = "black"

DATASET_ORDER = ["CD8 T", "CD4 T", "MSC", "Mouse B"]
DATASET_LABELS = {
    "CD8 T": "CD8 T",
    "CD4 T": "CD4 T",
    "MSC": "MSC",
    "Mouse B": "Mouse B",
}

SUMMARY_ORDER = [
    "CD8 T",
    "CD4 T",
    "MSC",
    "Mouse B",
    "All tasks pooled",
    "All tasks excluding Mouse B",
]


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


def parse_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) != 0.0
    return str(x).strip().lower() in {"true", "t", "1", "1.0", "yes", "y"}


def canonicalize_dataset(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    low = s.lower()

    if low.startswith("note."):
        return np.nan
    if "cd8" in low:
        return "CD8 T"
    if "cd4" in low:
        return "CD4 T"
    if "msc" in low:
        return "MSC"
    if "mouse" in low and "b" in low:
        return "Mouse B"

    return np.nan


def load_l2_audit(letter_xlsx: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(letter_xlsx, sheet_name=sheet_name)
    print(f"Loaded {sheet_name}: shape={df.shape}")

    df = df.copy()
    df["Dataset"] = df["Dataset"].map(canonicalize_dataset)
    df = df[df["Dataset"].notna()].copy()

    required = ["Dataset", "Cluster_ID", "Ground_Truth", "Score_Standard", "Score_Curated"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required L2 columns: {missing}")

    df["Score_Standard"] = pd.to_numeric(df["Score_Standard"], errors="coerce")
    df["Score_Curated"] = pd.to_numeric(df["Score_Curated"], errors="coerce")
    df = df.dropna(subset=["Score_Standard", "Score_Curated"]).copy()

    # Do not use UsedInConfusion here. It is a confusion-matrix flag, not a benchmark inclusion flag.
    df["Diff"] = df["Score_Curated"] - df["Score_Standard"]
    df["Diff_Percentage_Points"] = df["Diff"] * 100

    eps = 1e-12
    df["Direction"] = np.where(
        df["Diff"] > eps,
        "Improved",
        np.where(df["Diff"] < -eps, "Worsened", "Unchanged"),
    )

    df["Dataset"] = pd.Categorical(df["Dataset"], DATASET_ORDER, ordered=True)
    df = df.sort_values(["Dataset", "Cluster_ID"]).reset_index(drop=True)

    print(f"After cleaning evaluable matched clusters: shape={df.shape}")
    print(df["Dataset"].value_counts().sort_index())

    return df


def bootstrap_ci_mean(diff, n_boot: int = 20000, seed: int = 42) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()

    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def paired_permutation_p(diff, n_perm: int = 50000, seed: int = 42) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return np.nan

    observed = abs(diff.mean())
    if np.isclose(observed, 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    n = len(diff)
    count = 0

    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        null_stat = abs((diff * signs).mean())
        if null_stat >= observed:
            count += 1

    # Plus-one correction.
    return float((count + 1) / (n_perm + 1))


def paired_wilcoxon_p(diff) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return np.nan
    if np.allclose(diff, 0.0):
        return 1.0

    try:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except TypeError:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def cohen_dz(diff) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) < 2:
        return np.nan

    sd = diff.std(ddof=1)
    mean = diff.mean()

    if np.isclose(sd, 0.0):
        return 0.0 if np.isclose(mean, 0.0) else np.nan

    return float(mean / sd)


def summarize_group(
    sub: pd.DataFrame,
    label: str,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    diff = sub["Diff"].to_numpy(dtype=float)

    ci_low, ci_high = bootstrap_ci_mean(diff, n_boot=n_boot, seed=seed)
    w_p = paired_wilcoxon_p(diff)
    perm_p = paired_permutation_p(diff, n_perm=n_perm, seed=seed)

    n_improved = int((diff > 1e-12).sum())
    n_worsened = int((diff < -1e-12).sum())
    n_unchanged = int((np.abs(diff) <= 1e-12).sum())

    return {
        "Benchmark": label,
        "N_Matched_Clusters": int(len(sub)),
        "Mean_Sanno_Standard_Pct": float(sub["Score_Standard"].mean() * 100),
        "Mean_Sanno_LLM_scCurator_Pct": float(sub["Score_Curated"].mean() * 100),
        "Mean_Delta_Sanno_PctPts": float(diff.mean() * 100),
        "Bootstrap_95CI_Lower_PctPts": float(ci_low * 100),
        "Bootstrap_95CI_Upper_PctPts": float(ci_high * 100),
        "Median_Delta_Sanno_PctPts": float(np.median(diff) * 100),
        "Wilcoxon_P": w_p,
        "Permutation_P": perm_p,
        "Cohen_dz": cohen_dz(diff),
        "N_Improved": n_improved,
        "N_Unchanged": n_unchanged,
        "N_Worsened": n_worsened,
        "Improved_Unchanged_Worsened": f"{n_improved}/{n_unchanged}/{n_worsened}",
    }


def build_L11_summary(df: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []

    for dataset in DATASET_ORDER:
        sub = df[df["Dataset"].astype(str) == dataset].copy()
        if not sub.empty:
            rows.append(summarize_group(sub, dataset, n_boot, n_perm, seed))

    rows.append(summarize_group(df, "All tasks pooled", n_boot, n_perm, seed))

    non_ceiling = df[df["Dataset"].astype(str) != "Mouse B"].copy()
    rows.append(
        summarize_group(
            non_ceiling,
            "All tasks excluding Mouse B",
            n_boot,
            n_perm,
            seed,
        )
    )

    out = pd.DataFrame(rows)
    out["Benchmark"] = pd.Categorical(out["Benchmark"], SUMMARY_ORDER, ordered=True)
    out = out.sort_values("Benchmark").reset_index(drop=True)
    out["Benchmark"] = out["Benchmark"].astype(str)

    return out


def build_L12_leave_one_task_out(df: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []

    base = summarize_group(df, "None (all tasks)", n_boot, n_perm, seed)
    base["Excluded_Task"] = "None"
    rows.append(base)

    for dataset in DATASET_ORDER:
        sub = df[df["Dataset"].astype(str) != dataset].copy()
        row = summarize_group(sub, f"Exclude {dataset}", n_boot, n_perm, seed)
        row["Excluded_Task"] = dataset
        rows.append(row)

    out = pd.DataFrame(rows)
    out["Direction_Consistent_With_All_Tasks"] = out["Mean_Delta_Sanno_PctPts"] >= 0
    return out


def build_cluster_diff_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Dataset",
        "Cluster_ID",
        "Ground_Truth",
        "Score_Standard",
        "Score_Curated",
        "Diff",
        "Diff_Percentage_Points",
        "Direction",
    ]
    optional_cols = [
        "Standard_Answer",
        "Curated_Answer",
        "Standard_CellType",
        "Curated_CellType",
        "GT_Major",
        "GT_State",
    ]

    keep = cols + [c for c in optional_cols if c in df.columns]
    out = df[keep].copy()
    out["Dataset"] = out["Dataset"].astype(str)
    return out


def write_tables_to_excel(
    letter_xlsx: Path,
    L11_summary: pd.DataFrame,
    cluster_diff: pd.DataFrame,
    L12_loo: pd.DataFrame,
) -> None:
    if not letter_xlsx.exists():
        raise FileNotFoundError(f"LetterTables file not found: {letter_xlsx}")

    with pd.ExcelWriter(
        letter_xlsx,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        L11_summary.to_excel(writer, sheet_name="L11_paired_statistical_robustness", index=False)
        cluster_diff.to_excel(writer, sheet_name="L11_cluster_paired_differences", index=False)
        L12_loo.to_excel(writer, sheet_name="L12_leave_one_task_out_robustness", index=False)


def savefig(fig, outdir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_l3a_paired_differences(ax, cluster_diff: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)

    x_positions = np.arange(len(DATASET_ORDER))
    for i, dataset in enumerate(DATASET_ORDER):
        sub = cluster_diff[cluster_diff["Dataset"].astype(str) == dataset].copy()
        y = sub["Diff_Percentage_Points"].to_numpy(dtype=float)
        jitter = rng.normal(0, 0.045, size=len(y))

        point_colors = np.where(
            y > 1e-12,
            CURATED_COLOR,
            np.where(y < -1e-12, WORSE_COLOR, NEUTRAL_COLOR),
        )

        ax.scatter(
            np.full(len(y), i) + jitter,
            y,
            s=42,
            c=point_colors,
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )

        mean = float(np.mean(y))
        ci_low, ci_high = bootstrap_ci_mean(y / 100.0, n_boot=20000, seed=42)
        ci_low *= 100
        ci_high *= 100

        ax.errorbar(
            i,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="o",
            color=CURATED_COLOR,
            ecolor=CURATED_COLOR,
            elinewidth=2,
            capsize=5,
            markersize=7,
            markeredgecolor=EDGE_COLOR,
            zorder=4,
        )

    ax.axhline(0, color=ZERO_COLOR, linestyle="--", linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], rotation=35, ha="right")
    ax.set_ylabel("Delta S$_{anno}$, percentage points\nLLM-scCurator minus Standard")
    ax.set_title("Paired cluster-level differences")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_l3b_forest(ax, L11_summary: pd.DataFrame) -> None:
    plot_rows = L11_summary.copy()
    plot_rows["Benchmark"] = pd.Categorical(plot_rows["Benchmark"], SUMMARY_ORDER, ordered=True)
    plot_rows = plot_rows.sort_values("Benchmark")
    plot_rows = plot_rows.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(plot_rows))
    x = plot_rows["Mean_Delta_Sanno_PctPts"].to_numpy(dtype=float)
    lo = plot_rows["Bootstrap_95CI_Lower_PctPts"].to_numpy(dtype=float)
    hi = plot_rows["Bootstrap_95CI_Upper_PctPts"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        y,
        xerr=[x - lo, hi - x],
        fmt="o",
        color=CURATED_COLOR,
        ecolor=CURATED_COLOR,
        elinewidth=2,
        capsize=5,
        markersize=7,
        markeredgecolor=EDGE_COLOR,
    )

    ax.axvline(0, color=ZERO_COLOR, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["Benchmark"])
    ax.set_xlabel("Mean delta S$_{anno}$, percentage points")
    ax.set_title("Bootstrap confidence intervals")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_l3c_direction_counts(ax, L11_summary: pd.DataFrame) -> None:
    plot_rows = L11_summary[L11_summary["Benchmark"].isin(DATASET_ORDER)].copy()
    plot_rows["Benchmark"] = pd.Categorical(plot_rows["Benchmark"], DATASET_ORDER, ordered=True)
    plot_rows = plot_rows.sort_values("Benchmark")

    x = np.arange(len(plot_rows))
    improved = plot_rows["N_Improved"].to_numpy(dtype=float)
    unchanged = plot_rows["N_Unchanged"].to_numpy(dtype=float)
    worsened = plot_rows["N_Worsened"].to_numpy(dtype=float)

    ax.bar(x, improved, color=CURATED_COLOR, edgecolor=EDGE_COLOR, linewidth=0.5, label="Improved")
    ax.bar(x, unchanged, bottom=improved, color=NEUTRAL_COLOR, edgecolor=EDGE_COLOR, linewidth=0.5, label="Unchanged")
    ax.bar(x, worsened, bottom=improved + unchanged, color=WORSE_COLOR, edgecolor=EDGE_COLOR, linewidth=0.5, label="Worsened")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_rows["Benchmark"], rotation=35, ha="right")
    ax.set_ylabel("Matched clusters")
    ax.set_title("Cluster-level direction counts")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_l3d_leave_one_task_out(ax, L12_loo: pd.DataFrame) -> None:
    plot_rows = L12_loo.copy()
    plot_rows = plot_rows.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(plot_rows))
    x = plot_rows["Mean_Delta_Sanno_PctPts"].to_numpy(dtype=float)
    lo = plot_rows["Bootstrap_95CI_Lower_PctPts"].to_numpy(dtype=float)
    hi = plot_rows["Bootstrap_95CI_Upper_PctPts"].to_numpy(dtype=float)

    labels = plot_rows["Excluded_Task"].replace({"None": "None\n(all tasks)"})

    ax.errorbar(
        x,
        y,
        xerr=[x - lo, hi - x],
        fmt="o",
        color=CURATED_COLOR,
        ecolor=CURATED_COLOR,
        elinewidth=2,
        capsize=5,
        markersize=7,
        markeredgecolor=EDGE_COLOR,
    )

    ax.axvline(0, color=ZERO_COLOR, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean delta S$_{anno}$, percentage points")
    ax.set_title("Leave-one-task-out robustness")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figures(
    L11_summary: pd.DataFrame,
    cluster_diff: pd.DataFrame,
    L12_loo: pd.DataFrame,
    fig_outdir: Path,
) -> None:
    set_style()
    fig_outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=300)
    axes = axes.ravel()

    plot_l3a_paired_differences(axes[0], cluster_diff)
    plot_l3b_forest(axes[1], L11_summary)
    plot_l3c_direction_counts(axes[2], L11_summary)
    plot_l3d_leave_one_task_out(axes[3], L12_loo)

    for label, ax in zip(["a", "b", "c", "d"], axes):
        ax.text(
            -0.14,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "Paired statistical robustness of Standard versus LLM-scCurator",
        fontsize=16,
        y=1.02,
    )

    savefig(fig, fig_outdir, "ReviewerFig_L3_statistical_robustness")

    individual = [
        ("ReviewerFig_L11a_paired_cluster_differences", plot_l3a_paired_differences, cluster_diff),
        ("ReviewerFig_L11b_bootstrap_ci_forest", plot_l3b_forest, L11_summary),
        ("ReviewerFig_L11c_direction_counts", plot_l3c_direction_counts, L11_summary),
        ("ReviewerFig_L12_leave_one_task_out", plot_l3d_leave_one_task_out, L12_loo),
    ]

    for stem, func, data in individual:
        fig, ax = plt.subplots(figsize=(6.6, 4.9), dpi=300)
        if stem == "ReviewerFig_L11a_paired_cluster_differences":
            func(ax, cluster_diff)
        elif stem == "ReviewerFig_L11b_bootstrap_ci_forest":
            func(ax, L11_summary)
        elif stem == "ReviewerFig_L11c_direction_counts":
            func(ax, L11_summary)
        else:
            func(ax, L12_loo)
        savefig(fig, fig_outdir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter-xlsx", default="paper/revision/LetterTables.xlsx")
    parser.add_argument("--audit-sheet", default="L2_per_cluster_audit")
    parser.add_argument("--csv-outdir", default="paper/revision_tables")
    parser.add_argument("--fig-outdir", default="paper/revision_figures")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--n-perm", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", type=int, default=4)
    args = parser.parse_args()

    letter_xlsx = Path(args.letter_xlsx)
    csv_outdir = Path(args.csv_outdir)
    fig_outdir = Path(args.fig_outdir)

    csv_outdir.mkdir(parents=True, exist_ok=True)
    fig_outdir.mkdir(parents=True, exist_ok=True)

    df = load_l2_audit(letter_xlsx, args.audit_sheet)

    L11_summary = build_L11_summary(df, args.n_boot, args.n_perm, args.seed)
    cluster_diff = build_cluster_diff_table(df)
    L12_loo = build_L12_leave_one_task_out(df, args.n_boot, args.n_perm, args.seed)

    for table in [L11_summary, cluster_diff, L12_loo]:
        float_cols = table.select_dtypes(include=["float"]).columns
        table[float_cols] = table[float_cols].round(args.round)

    L11_summary.to_csv(csv_outdir / "L11_paired_statistical_robustness.csv", index=False)
    cluster_diff.to_csv(csv_outdir / "L12_cluster_paired_differences.csv", index=False)
    L12_loo.to_csv(csv_outdir / "L13_leave_one_task_out_robustness.csv", index=False)

    write_tables_to_excel(letter_xlsx, L11_summary, cluster_diff, L12_loo)
    make_figures(L11_summary, cluster_diff, L12_loo, fig_outdir)

    print("Wrote:")
    print(f"  {csv_outdir / 'L11_paired_statistical_robustness.csv'}")
    print(f"  {csv_outdir / 'L12_cluster_paired_differences.csv'}")
    print(f"  {csv_outdir / 'L13_leave_one_task_out_robustness.csv'}")
    print(f"  {letter_xlsx} sheets:")
    print("    L11_paired_statistical_robustness")
    print("    L12_cluster_paired_differences")
    print("    L13_leave_one_task_out_robustness")
    print(f"  {fig_outdir / 'ReviewerFig_L3_statistical_robustness.pdf'}")
    print(f"  {fig_outdir / 'ReviewerFig_L3_statistical_robustness.png'}")


if __name__ == "__main__":
    main()
