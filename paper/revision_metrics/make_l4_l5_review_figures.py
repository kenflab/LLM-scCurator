#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_XLSX = Path("paper/revision/LetterTables.xlsx")
OUT_DIR = Path("paper/revision_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Standard": "#5DA5DA",
    "LLM-scCurator": "#D62728",
}

MARKERS = {
    "Standard": "s",
    "LLM-scCurator": "o",
}

LINESTYLES = {
    "Standard": "--",
    "LLM-scCurator": "-",
}

METHOD_LABELS = {
    "Standard": "Standard",
    "Curated": "LLM-scCurator",
}


STANDARD_COLOR = COLORS["Standard"]
CURATED_COLOR = COLORS["LLM-scCurator"]
REFERENCE_COLOR = "#8A8F98"
ZERO_COLOR = "#333333"


DATASET_ORDER = ["CD8 T", "CD4 T", "MSC", "Mouse B"]
SCHEME_ORDER = [
    "Default_task_specific",
    "Equal_0.5_0.5",
    "Lineage_heavy_0.8_0.2",
    "State_heavy_0.3_0.7",
    "Major_lineage_only",
    "Exact_state_only",
]

SCHEME_LABELS = {
    "Default_task_specific": "Default",
    "Equal_0.5_0.5": "Equal\n0.5/0.5",
    "Lineage_heavy_0.8_0.2": "Lineage-heavy\n0.8/0.2",
    "State_heavy_0.3_0.7": "State-heavy\n0.3/0.7",
    "Major_lineage_only": "Major-lineage\nonly",
    "Exact_state_only": "Exact-state\nonly",
}

METRIC_LABELS = {
    "Mean_S_anno": "Mean S$_{anno}$",
    "Exact_State_Agreement": "Exact state\nagreement",
    "Major_Lineage_Accuracy": "Major-lineage\naccuracy",
    "Ontology_Consistent_Accuracy_Sanno_ge_0.5": "Hierarchy-consistent\naccuracy",
    "Low_Consistency_Rate_Sanno_lt_0.5": "Low-consistency\nrate",
}


def set_style():
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


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = ~np.isnan(values) & ~np.isnan(weights)
    return np.sum(values[ok] * weights[ok]) / np.sum(weights[ok])


def savefig(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_l4_overall(l4):
    methods = ["Standard", "Curated"]
    metrics = [
        "Mean_S_anno",
        "Exact_State_Agreement",
        "Major_Lineage_Accuracy",
        "Ontology_Consistent_Accuracy_Sanno_ge_0.5",
        "Low_Consistency_Rate_Sanno_lt_0.5",
    ]

    rows = []
    for method in methods:
        sub = l4[l4["Method"] == method].copy()
        row = {"Method": method}
        for metric in metrics:
            row[metric] = weighted_mean(sub[metric], sub["N"])
        rows.append(row)

    overall = pd.DataFrame(rows)

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.2, 4.8))

    y_standard = overall.loc[overall["Method"] == "Standard", metrics].iloc[0].values * 100
    y_curated = overall.loc[overall["Method"] == "Curated", metrics].iloc[0].values * 100

    ax.bar(
        x - width / 2,
        y_standard,
        width,
        label=METHOD_LABELS["Standard"],
        color=STANDARD_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        y_curated,
        width,
        label=METHOD_LABELS["Curated"],
        color=CURATED_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )

    for i, (a, b) in enumerate(zip(y_standard, y_curated)):
        ax.text(i - width / 2, a + 1.5, f"{a:.1f}", ha="center", va="bottom", fontsize=10)
        ax.text(i + width / 2, b + 1.5, f"{b:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Clusters (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_title("Complementary metrics across 52 evaluable benchmark clusters")
    ax.legend(frameon=False, ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(fig, "ReviewerFig_L4_overall_complementary_metrics")


def figure_l4_by_dataset(l4):
    sub = l4[l4["Method"].isin(["Standard", "Curated"])].copy()
    sub["Dataset"] = pd.Categorical(sub["Dataset"], DATASET_ORDER, ordered=True)
    sub = sub.sort_values(["Dataset", "Method"])

    metrics = [
        ("Mean_S_anno", "Mean S$_{anno}$"),
        ("Ontology_Consistent_Accuracy_Sanno_ge_0.5", "Hierarchy-consistent accuracy"),
        ("Major_Lineage_Accuracy", "Major-lineage accuracy"),
        ("Low_Consistency_Rate_Sanno_lt_0.5", "Low-consistency rate"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.6), sharey=False)

    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(DATASET_ORDER))
        width = 0.36

        std = (
            sub[sub["Method"] == "Standard"]
            .set_index("Dataset")
            .reindex(DATASET_ORDER)[metric]
            .values
            * 100
        )
        cur = (
            sub[sub["Method"] == "Curated"]
            .set_index("Dataset")
            .reindex(DATASET_ORDER)[metric]
            .values
            * 100
        )

        ax.bar(
            x - width / 2,
            std,
            width,
            color=STANDARD_COLOR,
            edgecolor="black",
            linewidth=0.5,
            label=METHOD_LABELS["Standard"],
        )
        ax.bar(
            x + width / 2,
            cur,
            width,
            color=CURATED_COLOR,
            edgecolor="black",
            linewidth=0.5,
            label=METHOD_LABELS["Curated"],
        )

        ax.set_title(title)
        ax.set_ylim(0, 108)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_ORDER, rotation=35, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax is axes[0]:
            ax.set_ylabel("Clusters (%)")

    axes[-1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1))
    fig.suptitle("Dataset-level complementary metrics", y=1.04, fontsize=16)

    savefig(fig, "ReviewerFig_L4_by_dataset_standard_vs_curated")


def figure_l5_delta(l5):
    sub = l5.copy()
    sub["Dataset"] = pd.Categorical(sub["Dataset"], DATASET_ORDER, ordered=True)
    sub["Weighting_Scheme"] = pd.Categorical(
        sub["Weighting_Scheme"], SCHEME_ORDER, ordered=True
    )
    sub = sub.sort_values(["Weighting_Scheme", "Dataset"])

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    y_positions = np.arange(len(SCHEME_ORDER))
    offsets = {
        "CD8 T": -0.24,
        "CD4 T": -0.08,
        "MSC": 0.08,
        "Mouse B": 0.24,
    }
    markers = {
        "CD8 T": "o",
        "CD4 T": "s",
        "MSC": "^",
        "Mouse B": "D",
    }

    dataset_colors = {
        "CD8 T": "#0072B2",
        "CD4 T": "#009E73",
        "MSC": "#CC79A7",
        "Mouse B": "#E69F00",
    }

    for dataset in DATASET_ORDER:
        ds = sub[sub["Dataset"] == dataset].set_index("Weighting_Scheme").reindex(SCHEME_ORDER)
        x = ds["Mean_Difference_Curated_minus_Standard"].values * 100
        y = y_positions + offsets[dataset]

        ax.scatter(
            x,
            y,
            s=70,
            marker=markers[dataset],
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.5,
            label=dataset,
            zorder=3,
        )

    ax.axvline(0, color=ZERO_COLOR, linewidth=1.1, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SCHEME_LABELS[s] for s in SCHEME_ORDER])
    ax.set_xlabel("LLM-scCurator minus Standard, percentage points")
    ax.set_title("Sensitivity of Standard-versus-Curated comparison to S$_{anno}$ weighting")
    ax.legend(frameon=False, ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(fig, "ReviewerFig_L5_weight_sensitivity_delta_by_dataset")


def figure_l5_overall(l5):
    rows = []
    for scheme in SCHEME_ORDER:
        sub = l5[l5["Weighting_Scheme"] == scheme].copy()
        rows.append(
            {
                "Weighting_Scheme": scheme,
                "Mean_Standard": weighted_mean(sub["Mean_Standard"], sub["N"]),
                "Mean_Curated": weighted_mean(sub["Mean_Curated"], sub["N"]),
                "Delta": weighted_mean(
                    sub["Mean_Difference_Curated_minus_Standard"], sub["N"]
                ),
            }
        )

    overall = pd.DataFrame(rows)

    y = np.arange(len(SCHEME_ORDER))
    fig, ax = plt.subplots(figsize=(10.4, 5.4))

    std = overall["Mean_Standard"].values * 100
    cur = overall["Mean_Curated"].values * 100

    for i in range(len(SCHEME_ORDER)):
        ax.plot([std[i], cur[i]], [y[i], y[i]], color="#B0B0B0", linewidth=2, zorder=1)
        ax.scatter(std[i], y[i], s=75, color=STANDARD_COLOR, edgecolor="black", linewidth=0.5, label="Standard" if i == 0 else None, zorder=2)
        ax.scatter(cur[i], y[i], s=75, color=CURATED_COLOR, edgecolor="black", linewidth=0.5, label="LLM-scCurator" if i == 0 else None, zorder=3)

        delta = cur[i] - std[i]
        ax.text(
            max(std[i], cur[i]) + 1.0,
            y[i],
            f"{delta:+.1f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([SCHEME_LABELS[s] for s in SCHEME_ORDER])
    ax.set_xlabel("Mean score across evaluable clusters (%)")
    ax.set_xlim(45, 103)
    ax.set_title("Overall weighting sensitivity across 52 evaluable clusters")
    ax.legend(frameon=False, loc='upper left',  bbox_to_anchor=(1.05, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(fig, "ReviewerFig_L5_overall_standard_vs_curated")


def main():
    set_style()

    l4 = pd.read_excel(IN_XLSX, sheet_name="L4_complementary_metrics")
    l5 = pd.read_excel(IN_XLSX, sheet_name="L5_weight_sensitivity")

    figure_l4_overall(l4)
    figure_l4_by_dataset(l4)
    figure_l5_delta(l5)
    figure_l5_overall(l5)

    print(f"Wrote reviewer figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()


