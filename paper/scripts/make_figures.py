#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper/scripts/make_figures.py

Optional manuscript-facing figure renderer.

This script is intentionally notebook-free and does NOT require raw `.h5ad`.
It renders a small subset of panels directly from the versioned Source Data.

Inputs (versioned Source Data; default locations):
  - Fig. 2a:           paper/source_data/figure_data/Fig2/Fig2a-d_data.csv
                       (or paper/source_data/figure_data/Fig2a.csv, if present)
  - ED Fig. 2 (a–c):   paper/source_data/figure_data/EDFig2/EDFig2a_data.csv
                       paper/source_data/figure_data/EDFig2/EDFig2b_data.csv
                       paper/source_data/figure_data/EDFig2/EDFig2c_data.csv
                       (per-cluster scored tables; confusion matrices are computed at plot time)

Outputs (rendered figures; not versioned by default):
  - paper/figures/*.pdf (+ png)

Notes
- Most review/verification can be done directly from paper/source_data/, indexed by paper/FIGURE_MAP.csv.
- This renderer is provided as a convenience for reviewers/users who want ready-to-view PDFs.
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LOG = logging.getLogger("paper.make_figures")


# -------------------------
# Style
# -------------------------
def set_nature_rcparams() -> None:
    """A minimal, publication-friendly Matplotlib style."""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["axes.linewidth"] = 1.0


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Fig 2a (bars across datasets)
# -------------------------
PIPELINE_ORDER_FIG2A = ["Standard", "Curated", "CellTypist", "SingleR", "Azimuth"]

PIPELINE_LABELS = {
    "Standard":   "Standard\n(Top DE)",
    "Curated":    "LLM-scCurator\n(Feature distillation)",
    "CellTypist": "CellTypist\n(Supervised)",
    "SingleR":    "SingleR\n(Reference)",
    "Azimuth":    "Azimuth\n(Reference)",
}

PIPELINE_COLORS = {
    "Standard":   "#42A5F5",
    "Curated":    "#D32F2F",
    "CellTypist": "#66BB6A",
    "SingleR":    "#7E57C2",
    "Azimuth":    "#FFB74D",
}

DATASET_ORDER_DEFAULT = ["cd8", "cd4", "msc", "mouse_b"]
DATASET_TITLES = {"cd8": "CD8", "cd4": "CD4", "msc": "MSC", "mouse_b": "MOUSE_B"}


def _normalize_fig2a_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a Fig2a table to the columns expected by `plot_fig2a_bars`.

    Expected columns (case-sensitive):
      - Dataset_Tag (e.g., cd8)
      - Pipeline    (e.g., Standard)
      - Mean_pct, CI_Low_pct, CI_High_pct

    We keep this permissive to accommodate small format drifts.
    """
    x = df.copy()

    # optional: panel filtering
    for c in ["Panel", "panel", "FIG", "Figure"]:
        if c in x.columns:
            # accept Fig2a or Fig2a-d naming
            mask = x[c].astype(str).str.contains("Fig2a", case=False, regex=False)
            if mask.any():
                x = x.loc[mask].copy()
            break

    # map possible dataset tag columns
    if "Dataset_Tag" not in x.columns:
        for c in ["dataset_tag", "dataset", "DatasetTag"]:
            if c in x.columns:
                x["Dataset_Tag"] = x[c].astype(str)
                break

    # map possible dataset display column
    if "Dataset" not in x.columns and "Dataset_Tag" in x.columns:
        x["Dataset"] = x["Dataset_Tag"].map(DATASET_TITLES).fillna(x["Dataset_Tag"])

    # pipeline col
    if "Pipeline" not in x.columns:
        for c in ["pipeline", "Method", "method"]:
            if c in x.columns:
                x["Pipeline"] = x[c].astype(str)
                break

    # percentages
    if "Mean_pct" not in x.columns and "Mean" in x.columns:
        x["Mean_pct"] = x["Mean"].astype(float) * 100.0
    if "CI_Low_pct" not in x.columns and "CI_Low" in x.columns:
        x["CI_Low_pct"] = x["CI_Low"].astype(float) * 100.0
    if "CI_High_pct" not in x.columns and "CI_High" in x.columns:
        x["CI_High_pct"] = x["CI_High"].astype(float) * 100.0

    needed = {"Dataset_Tag", "Pipeline", "Mean_pct", "CI_Low_pct", "CI_High_pct"}
    missing = [c for c in sorted(needed) if c not in x.columns]
    if missing:
        raise ValueError(f"Fig2a input table missing columns: {missing}. Columns present: {list(x.columns)}")

    return x


def load_fig2a_table(fig_data_dir: Path) -> pd.DataFrame:
    """Load Fig2a summary data from versioned Source Data."""
    # Preferred: a dedicated Fig2a table (if present)
    cand1 = fig_data_dir / "Fig2a.csv"
    if cand1.exists():
        LOG.info("Loading Fig2a table: %s", cand1)
        return _normalize_fig2a_like(pd.read_csv(cand1))

    # Common layout in this repo: Fig2/Fig2a-d_data.csv
    cand2 = fig_data_dir / "Fig2" / "Fig2a-d_data.csv"
    if cand2.exists():
        LOG.info("Loading Fig2a-d table: %s", cand2)
        return _normalize_fig2a_like(pd.read_csv(cand2))

    raise FileNotFoundError(
        "Could not find Fig2a Source Data. Tried: " + ", ".join([str(cand1), str(cand2)])
    )


def plot_fig2a_bars(summary_df: pd.DataFrame, out_pdf: Path, out_png: Path, dataset_order: Sequence[str]) -> None:
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(16, 5), dpi=300, sharey=True)
    if len(dataset_order) == 1:
        axes = [axes]

    for ax, tag in zip(axes, dataset_order):
        sub = summary_df[summary_df["Dataset_Tag"].astype(str) == tag].copy()
        if sub.empty:
            ax.axis("off")
            continue

        # keep pipeline order but only present ones
        present = [p for p in PIPELINE_ORDER_FIG2A if p in set(sub["Pipeline"])]

        sub = sub.set_index("Pipeline").loc[present].reset_index()

        x = np.arange(len(sub))
        means = sub["Mean_pct"].astype(float).values
        yerr = np.vstack([
            means - sub["CI_Low_pct"].astype(float).values,
            sub["CI_High_pct"].astype(float).values - means
        ])

        colors = [PIPELINE_COLORS[p] for p in sub["Pipeline"]]
        ax.bar(x, means, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([PIPELINE_LABELS[p] for p in sub["Pipeline"]], rotation=90, fontsize=12)
        ax.set_ylim(0, 110)
        ax.set_title(DATASET_TITLES.get(tag, tag), fontsize=14, fontweight="bold")

    axes[0].set_ylabel("Annotation accuracy (%)\n(mean hierarchical score)", fontsize=13)
    axes[0].tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    LOG.info("Wrote: %s", out_pdf)


# -------------------------
# ED Fig 2 confusion matrices (computed from per-cluster scored tables)
# -------------------------
def _ensure_repo_imports(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root.resolve()))


def choose_prediction_column(df: pd.DataFrame, pipeline: str) -> str:
    # expected: Standard_Answer / Curated_Answer / CellTypist_Answer etc.
    cand = [f"{pipeline}_Answer", f"{pipeline}_Subtype_Clean", f"{pipeline}_CellType"]
    for c in cand:
        if c in df.columns:
            return c
    raise ValueError(f"No prediction column found for pipeline={pipeline}. Tried: {cand}")


def compute_confusion_mean_score(
    df: pd.DataFrame,
    pipeline: str,
    cfg: Any,
    state_order: List[str],
) -> Dict[str, Any]:
    """
    Confusion matrix where each cell contains mean hierarchical score (0-1)
    for (GT_State, Pred_State) pairs, using Score_{pipeline}.
    """
    pred_col = choose_prediction_column(df, pipeline)
    score_col = f"Score_{pipeline}"
    if score_col not in df.columns:
        raise ValueError(f"{score_col} not found.")

    # filter
    if "UsedInConfusion" in df.columns:
        df_used = df[df["UsedInConfusion"] == True].copy()  # noqa: E712
    else:
        # fallback: exclude default state if available
        default_state = getattr(cfg, "default_state", "Other")
        df_used = df[df["GT_State"].astype(str) != str(default_state)].copy()

    from benchmarks.hierarchical_scoring import _parse_state_generic  # requires repo imports

    labels = state_order[:]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    cm_sum = np.zeros((len(labels), len(labels)), dtype=float)
    cm_count = np.zeros((len(labels), len(labels)), dtype=int)

    gt_states: List[str] = []
    pr_states: List[str] = []

    for _, row in df_used.iterrows():
        gt = str(row.get("GT_State", "Other"))
        pred = _parse_state_generic(str(row[pred_col]), cfg)
        score = float(row[score_col])

        gt_lab = gt if gt in label_to_idx else "Other"
        pr_lab = pred if pred in label_to_idx else "Other"

        i = label_to_idx[gt_lab]
        j = label_to_idx[pr_lab]
        cm_sum[i, j] += score
        cm_count[i, j] += 1

        gt_states.append(gt_lab)
        pr_states.append(pr_lab)

    with np.errstate(invalid="ignore", divide="ignore"):
        cm_mean = np.where(cm_count > 0, cm_sum / cm_count, 0.0)

    n = len(gt_states)
    n_correct = sum(1 for g, p in zip(gt_states, pr_states) if g == p)
    state_acc = 100.0 * n_correct / n if n > 0 else 0.0
    hier_mean = df_used[score_col].dropna().astype(float).mean() * 100.0

    return {
        "labels": labels,
        "cm": cm_mean,
        "state_acc": state_acc,
        "n_correct": n_correct,
        "n": n,
        "hier_mean": hier_mean,
        "counts": cm_count,
    }


def plot_confusion_panels(
    df: pd.DataFrame,
    cfg: Any,
    state_order: List[str],
    dataset_title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    pipelines = ["Standard", "Curated", "CellTypist"]
    titles = {"Standard": "Standard pipeline", "Curated": "LLM-scCurator", "CellTypist": "CellTypist"}

    cmaps = {"Standard": "Blues", "Curated": "Reds", "CellTypist": "Greens"}
    title_colors = {"Standard": "#1976D2", "Curated": "#D32F2F", "CellTypist": "#388E3C"}

    results = {p: compute_confusion_mean_score(df, p, cfg, state_order) for p in pipelines}

    fig, axes = plt.subplots(1, len(pipelines), figsize=(18, 6), dpi=300)
    for ax, p in zip(axes, pipelines):
        res = results[p]
        labels = res["labels"]
        cm = res["cm"]
        counts = res["counts"]

        ax.imshow(cm, vmin=0.0, vmax=1.0, cmap=cmaps[p])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # show mean score; N is implicit in counts
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10)

        title = (
            f"{titles[p]}\n"
            f"State acc: {res['state_acc']:.1f}% ({res['n_correct']}/{res['n']})\n"
            f"Hierarchical score: {res['hier_mean']:.1f}%"
        )
        ax.set_title(title, fontsize=14, fontweight="bold", color=title_colors[p])

        if ax is axes[0]:
            ax.set_ylabel("True state", fontsize=12, fontweight="bold")
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Predicted state", fontsize=12, fontweight="bold")

    plt.suptitle(f"{dataset_title} benchmark", fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote: %s", out_pdf)


def load_edfig2_tables(fig_data_dir: Path) -> Dict[str, Path]:
    ed_dir = fig_data_dir / "EDFig2"
    out = {
        "cd8": ed_dir / "EDFig2a_data.csv",
        "cd4": ed_dir / "EDFig2b_data.csv",
        "msc": ed_dir / "EDFig2c_data.csv",
    }
    return {k: v for k, v in out.items() if v.exists()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-data-dir", default="paper/source_data/figure_data", type=str)
    ap.add_argument("--figdir", default="paper/figures", type=str)
    ap.add_argument("--dataset-order", default=",".join(DATASET_ORDER_DEFAULT), type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--make-fig2a", action="store_true")
    ap.add_argument("--make-confusions", action="store_true")
    ap.add_argument("--repo-root", default=".", type=str, help="Repo root for importing benchmarks/* configs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    set_nature_rcparams()

    fig_data_dir = Path(args.figure_data_dir).resolve()
    figdir = Path(args.figdir).resolve()
    safe_mkdir(figdir)

    dataset_order = [x.strip() for x in args.dataset_order.split(",") if x.strip()]

    if args.make_fig2a:
        summary = load_fig2a_table(fig_data_dir)

        out_pdf = figdir / "Fig2a.pdf"
        out_png = figdir / "Fig2a.png"
        plot_fig2a_bars(summary, out_pdf=out_pdf, out_png=out_png, dataset_order=dataset_order)

    if args.make_confusions:
        _ensure_repo_imports(Path(args.repo_root))

        # dataset-specific state orders and cfg objects
        from benchmarks.cd8_config import CD8_HIER_CFG
        from benchmarks.cd4_config import CD4_HIER_CFG
        from benchmarks.caf_config import CAF_HIER_CFG
        # mouse_b confusions are not part of ED Fig 2 in this repo; keep config for completeness
        from benchmarks.mouse_b_config import MOUSE_B_CFG  # noqa: F401

        conf_cfg = {
            "cd8": ("CD8 T cell", CD8_HIER_CFG, ["Naive", "EffMem", "Exhausted", "Resident", "MAIT", "ISG", "Cycling", "Other"]),
            "cd4": ("CD4 T cell", CD4_HIER_CFG, ["Naive", "EffMem", "Exhausted", "Treg", "Tfh", "Th17", "ISG", "Cycling", "Other"]),
            "msc": ("MSC/CAF", CAF_HIER_CFG, ["iCAF", "myCAF", "PVL", "Cycling", "Endothelial", "Other"]),
        }

        ed_tables = load_edfig2_tables(fig_data_dir)
        if not ed_tables:
            raise RuntimeError(f"No EDFig2*_data.csv found under: {fig_data_dir / 'EDFig2'}")

        for tag in dataset_order:
            if tag not in ed_tables or tag not in conf_cfg:
                continue
            title, cfg, order = conf_cfg[tag]
            df = pd.read_csv(ed_tables[tag])

            # name outputs by panel
            panel = {"cd8": "EDFig2a", "cd4": "EDFig2b", "msc": "EDFig2c"}[tag]
            out_pdf = figdir / f"{panel}_confusion.pdf"
            out_png = figdir / f"{panel}_confusion.png"
            plot_confusion_panels(df, cfg=cfg, state_order=order, dataset_title=title, out_pdf=out_pdf, out_png=out_png)

    if not args.make_fig2a and not args.make_confusions:
        LOG.info("Nothing to do. Use --make-fig2a and/or --make-confusions.")


if __name__ == "__main__":
    main()
