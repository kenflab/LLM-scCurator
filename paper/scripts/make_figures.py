#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper/scripts/make_figures.py

Canonical figure generation script (manuscript-facing).

Inputs (versioned Source Data):
  - paper/source_data/benchmark_tables/*_SCORED.csv

Outputs (versioned Source Data + figures):
  - paper/source_data/figure_data/Fig2a.csv
  - paper/figures/Fig2a.pdf (+ png)
  - (optional) ED confusion matrices from *_SCORED.csv

This script is intentionally notebook-free and does NOT require raw .h5ad.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LOG = logging.getLogger("paper.make_figures")


# -------------------------
# Style
# -------------------------
def set_nature_rcparams() -> None:
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
# Discover benchmark tables
# -------------------------
def discover_scored_tables(bench_dir: Path) -> Dict[str, Path]:
    """
    Expect filenames like:
      cd8_benchmark_results_integrated_SCORED.csv
      cd4_benchmark_results_integrated_SCORED.csv
      msc_benchmark_results_integrated_SCORED.csv
      mouse_b_benchmark_results_integrated_SCORED.csv
    Returns:
      tag -> path
    """
    out: Dict[str, Path] = {}
    for p in sorted(bench_dir.glob("*_benchmark_results_integrated_SCORED.csv")):
        name = p.name
        tag = name.replace("_benchmark_results_integrated_SCORED.csv", "")
        out[tag] = p
    return out


# -------------------------
# Robust CI (bootstrap)
# -------------------------
def mean_ci_bootstrap(
    x: pd.Series,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    v = x.dropna().astype(float).values
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    mean = float(v.mean())
    if n == 1:
        return mean, mean, mean, 1

    boots = rng.choice(v, size=(n_boot, n), replace=True).mean(axis=1)
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    # clamp to [0,1]
    low = max(0.0, low)
    high = min(1.0, high)
    return mean, low, high, n


# -------------------------
# Fig2a (bars across datasets)
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


def build_fig2a_table(scored_paths: Dict[str, Path], seed: int = 42) -> pd.DataFrame:
    rows = []
    for tag, path in scored_paths.items():
        df = pd.read_csv(path)
        for pipeline in PIPELINE_ORDER_FIG2A:
            col = f"Score_{pipeline}"
            if col not in df.columns:
                continue
            mean, low, high, n = mean_ci_bootstrap(df[col], seed=seed)
            rows.append({
                "Dataset": DATASET_TITLES.get(tag, tag),
                "Dataset_Tag": tag,
                "Pipeline": pipeline,
                "N": n,
                "Mean": mean,
                "CI_Low": low,
                "CI_High": high,
                "Mean_pct": mean * 100.0,
                "CI_Low_pct": low * 100.0,
                "CI_High_pct": high * 100.0,
            })
    out = pd.DataFrame(rows)
    return out


def plot_fig2a_bars(summary_df: pd.DataFrame, out_pdf: Path, out_png: Path, dataset_order: Sequence[str]) -> None:
    # 1x4 layout (shared y)
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(16, 5), dpi=300, sharey=True)
    if len(dataset_order) == 1:
        axes = [axes]

    for ax, tag in zip(axes, dataset_order):
        sub = summary_df[summary_df["Dataset_Tag"] == tag].copy()
        if sub.empty:
            ax.axis("off")
            continue

        # keep pipeline order but only present ones
        present = [p for p in PIPELINE_ORDER_FIG2A if p in set(sub["Pipeline"])]
        sub = sub.set_index("Pipeline").loc[present].reset_index()

        x = np.arange(len(sub))
        means = sub["Mean_pct"].values
        yerr = np.vstack([means - sub["CI_Low_pct"].values, sub["CI_High_pct"].values - means])

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
# Optional: ED confusion matrices (from *_SCORED.csv)
# -------------------------
def _ensure_repo_imports(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root.resolve()))


def choose_prediction_column(df: pd.DataFrame, pipeline: str) -> str:
    # expected: Standard_Answer / Curated_Answer / CellTypist_Answer etc.
    cand = [f"{pipeline}_Answer", f"{pipeline}_Subtype_Clean"]
    for c in cand:
        if c in df.columns:
            return c
    raise ValueError(f"No prediction column found for pipeline={pipeline}. Tried: {cand}")


def compute_confusion_mean_score(
    df: pd.DataFrame,
    pipeline: str,
    cfg,
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
        df_used = df[df["GT_State"].astype(str) != str(cfg.default_state)].copy()

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
    }


def plot_confusion_panels(
    df: pd.DataFrame,
    cfg,
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

        ax.imshow(cm, vmin=0.0, vmax=1.0, cmap=cmaps[p])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="paper/source_data/benchmark_tables", type=str)
    ap.add_argument("--figure-data-dir", default="paper/source_data/figure_data", type=str)
    ap.add_argument("--figdir", default="paper/figures", type=str)
    ap.add_argument("--dataset-order", default="cd8,cd4,msc,mouse_b", type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--make-fig2a", action="store_true")
    ap.add_argument("--make-confusions", action="store_true")
    ap.add_argument("--repo-root", default=".", type=str, help="Repo root for importing benchmarks/* configs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    set_nature_rcparams()

    bench_dir = Path(args.bench_dir).resolve()
    fig_data_dir = Path(args.figure_data_dir).resolve()
    figdir = Path(args.figdir).resolve()
    safe_mkdir(fig_data_dir)
    safe_mkdir(figdir)

    scored = discover_scored_tables(bench_dir)
    if not scored:
        raise RuntimeError(f"No *_SCORED.csv found in: {bench_dir}")

    dataset_order = [x.strip() for x in args.dataset_order.split(",") if x.strip()]

    # Fig2a
    if args.make_fig2a:
        summary = build_fig2a_table(scored, seed=args.seed)

        out_csv = fig_data_dir / "Fig2a.csv"
        summary.to_csv(out_csv, index=False)
        LOG.info("Wrote: %s", out_csv)

        out_pdf = figdir / "Fig2a.pdf"
        out_png = figdir / "Fig2a.png"
        plot_fig2a_bars(summary, out_pdf=out_pdf, out_png=out_png, dataset_order=dataset_order)

    # Optional confusions
    if args.make_confusions:
        _ensure_repo_imports(Path(args.repo_root))

        # dataset-specific state orders and cfg objects
        from benchmarks.cd8_config import CD8_HIER_CFG
        from benchmarks.cd4_config import CD4_HIER_CFG
        from benchmarks.caf_config import CAF_HIER_CFG
        from benchmarks.mouse_b_config import MOUSE_B_CFG

        conf_cfg = {
            "cd8": ("CD8 T cell", CD8_HIER_CFG, ["Naive", "EffMem", "Exhausted", "Resident", "MAIT", "ISG", "Cycling", "Other"]),
            "cd4": ("CD4 T cell", CD4_HIER_CFG, ["Naive", "EffMem", "Exhausted", "Treg", "Tfh", "Th17", "ISG", "Cycling", "Other"]),
            "msc": ("MSC", CAF_HIER_CFG, ["iCAF", "myCAF", "PVL", "Cycling", "Endothelial", "Other"]),
            "mouse_b": ("Mouse B-lineage (TMS)", MOUSE_B_CFG, ["Mature_B", "Erythrocyte_like", "Mast_like", "pDC_Myeloid_like", "Other"]),
        }

        for tag in dataset_order:
            if tag not in scored:
                continue
            if tag not in conf_cfg:
                continue
            title, cfg, order = conf_cfg[tag]
            df = pd.read_csv(scored[tag])

            out_pdf = figdir / f"ED_{DATASET_TITLES.get(tag, tag)}_confusion.pdf"
            out_png = figdir / f"ED_{DATASET_TITLES.get(tag, tag)}_confusion.png"
            plot_confusion_panels(df, cfg=cfg, state_order=order, dataset_title=title, out_pdf=out_pdf, out_png=out_png)

    if not args.make_fig2a and not args.make_confusions:
        LOG.info("Nothing to do. Use --make-fig2a and/or --make-confusions.")


if __name__ == "__main__":
    main()
