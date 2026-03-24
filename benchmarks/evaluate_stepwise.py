#!/usr/bin/env python
# coding: utf-8

"""
Evaluate stepwise end-to-end marginal-value experiments using Sanno
(ontology-aware hierarchical scoring).

Expected input
--------------
Long-format CSV per dataset, with columns such as:
- Dataset
- Cluster_ID   (or meta.cluster / cluster)
- Variant      (standard, filter_only, regex_mask, full_core, full_pipeline)
- Pred_Text    (or pred_label / Pred_Label / prediction_text)
- Ground_Truth (optional; if missing, derived from cluster ID)

Outputs
-------
- <dataset>_stepwise_scored.csv
- <dataset>_stepwise_summary.csv
- <dataset>_confusion_<variant>.csv
- <dataset>_per_state_metrics_<variant>.csv
- Stepwise_Sanno_Summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

from .hierarchical_scoring import (
    score_hierarchical,
    _expected_major_state_generic,
    _parse_state_generic,
)
from .cd8_config import CD8_HIER_CFG
from .cd4_config import CD4_HIER_CFG
from .caf_config import CAF_HIER_CFG
from .mouse_b_config import MOUSE_B_CFG, score_mouse_b
from .scorer_caf import score_caf_hierarchical
from .gt_mappings import (
    get_cd8_ground_truth,
    get_cd4_ground_truth,
    get_msc_ground_truth,
    get_bcell_ground_truth,
)

# -----------------------------------------------------------------------------
# Dataset-specific scoring wrappers
# -----------------------------------------------------------------------------
def _score_cd8(row, pred_col, cfg):
    return score_hierarchical(row, pred_col, cfg)

def _score_cd4(row, pred_col, cfg):
    return score_hierarchical(row, pred_col, cfg)

def _score_msc(row, pred_col, cfg):
    return score_caf_hierarchical(row, pred_col)

def _score_mouse_b(row, pred_col, cfg):
    return score_mouse_b(row, pred_col)

TASKS = {
    "CD8": {
        "cfg": CD8_HIER_CFG,
        "score_func": _score_cd8,
        "gt_mapper": get_cd8_ground_truth,
    },
    "CD4": {
        "cfg": CD4_HIER_CFG,
        "score_func": _score_cd4,
        "gt_mapper": get_cd4_ground_truth,
    },
    "MSC": {
        "cfg": CAF_HIER_CFG,
        "score_func": _score_msc,
        "gt_mapper": get_msc_ground_truth,
    },
    "MOUSE_B": {
        "cfg": MOUSE_B_CFG,
        "score_func": _score_mouse_b,
        "gt_mapper": get_bcell_ground_truth,
    },
}

VARIANT_ORDER = [
    "standard",
    "filter_only",
    "regex_mask",
    "full_core",
    "full_pipeline",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _detect_cluster_col(df):
    for c in ["Cluster_ID", "meta.cluster", "cluster"]:
        if c in df.columns:
            return c
    return df.columns[0]

def _detect_variant_col(df):
    for c in ["Variant", "variant"]:
        if c in df.columns:
            return c
    raise ValueError("Variant column not found. Expected one of: Variant, variant")

def _detect_pred_col(df):
    for c in ["Pred_Text", "pred_label", "Pred_Label", "prediction_text", "Prediction"]:
        if c in df.columns:
            return c
    raise ValueError(
        "Prediction column not found. Expected one of: "
        "Pred_Text, pred_label, Pred_Label, prediction_text, Prediction"
    )

def bootstrap_mean_ci(values, n_boot=5000, seed=42, alpha=0.05):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return vals[0], vals[0]

    rng = np.random.default_rng(seed)
    boot = []
    n = len(vals)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boot.append(sample.mean())

    lo = np.quantile(boot, alpha / 2)
    hi = np.quantile(boot, 1 - alpha / 2)
    return float(lo), float(hi)

def _build_state_series(df, cfg, pred_col):
    gt_states = []
    pred_states = []

    for _, row in df.iterrows():
        gt_major, gt_state = _expected_major_state_generic(row["Ground_Truth"], cfg)
        if gt_state == cfg.default_state:
            continue
        pred_state = _parse_state_generic(str(row[pred_col]), cfg)
        gt_states.append(gt_state)
        pred_states.append(pred_state)

    if not gt_states:
        return None, None

    gt_series = pd.Series(gt_states, name="GT_state")
    pred_series = pd.Series(pred_states, name="Pred_state")
    return gt_series, pred_series

def _confusion_and_per_state_metrics(gt, pred):
    cm = pd.crosstab(gt, pred)
    labels = sorted(set(gt.unique()) | set(pred.unique()))
    rows = []

    for lab in labels:
        tp = cm.loc[lab, lab] if (lab in cm.index and lab in cm.columns) else 0
        support = int(cm.loc[lab].sum()) if lab in cm.index else 0
        fp = int(cm[lab].sum() - tp) if lab in cm.columns else 0
        fn = support - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "State": lab,
            "Support": support,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    metrics_df = pd.DataFrame(rows).sort_values("State")
    return cm, metrics_df

# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------
def evaluate_stepwise_dataset(dataset_name, csv_path, output_dir):
    print(f"\n[INFO] Processing {dataset_name} from {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return []

    task_spec = TASKS.get(dataset_name)
    if task_spec is None:
        print(f"[WARN] No task config found for {dataset_name}")
        return []

    cfg = task_spec["cfg"]
    gt_mapper = task_spec["gt_mapper"]
    score_func = task_spec["score_func"]

    df = pd.read_csv(csv_path)
    cluster_col = _detect_cluster_col(df)
    variant_col = _detect_variant_col(df)
    pred_col = _detect_pred_col(df)

    if "Ground_Truth" not in df.columns:
        print(f"  > Applying GT mapper ({gt_mapper.__name__}) using column '{cluster_col}'")
        df["Ground_Truth"] = df[cluster_col].apply(gt_mapper)

    # transparency columns
    gt_major = []
    gt_state = []
    used_in_cm = []
    for gt_label in df["Ground_Truth"]:
        major, state = _expected_major_state_generic(gt_label, cfg)
        gt_major.append(major)
        gt_state.append(state)
        used_in_cm.append(state != cfg.default_state)

    df["GT_Major"] = gt_major
    df["GT_State"] = gt_state
    df["UsedInConfusion"] = used_in_cm

    # score row-wise
    df["Sanno"] = df.apply(lambda row: score_func(row, pred_col, cfg), axis=1)

    summary_rows = []
    for variant in VARIANT_ORDER:
        sub = df[df[variant_col].astype(str) == variant].copy()
        if sub.empty:
            continue

        scores = sub["Sanno"].astype(float)
        mean_score = float(scores.mean())
        frac_eq1 = float((scores == 1.0).mean())
        frac_ge05 = float((scores >= 0.5).mean())
        ci_lo, ci_hi = bootstrap_mean_ci(scores.values, n_boot=5000, seed=42)

        print(
            f"  > {variant:15s}: "
            f"Mean={mean_score:.3f}, "
            f"Frac==1={frac_eq1:.3f}, "
            f"Frac>=0.5={frac_ge05:.3f}, "
            f"95%CI=({ci_lo:.3f}, {ci_hi:.3f})"
        )

        summary_rows.append({
            "Dataset": dataset_name,
            "Variant": variant,
            "N": int(len(scores)),
            "MeanScore": mean_score,
            "FracScoreEq1": frac_eq1,
            "FracScoreGe0_5": frac_ge05,
            "MeanScore_CI_Lo": ci_lo,
            "MeanScore_CI_Hi": ci_hi,
        })

        gt_series, pred_series = _build_state_series(sub, cfg, pred_col)
        if gt_series is not None:
            cm, metrics_df = _confusion_and_per_state_metrics(gt_series, pred_series)

            cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_{variant}.csv")
            metrics_path = os.path.join(output_dir, f"{dataset_name}_per_state_metrics_{variant}.csv")

            cm.to_csv(cm_path)
            metrics_df.to_csv(metrics_path, index=False)

            print(f"[INFO] {dataset_name}/{variant}: confusion → {cm_path}")
            print(f"[INFO] {dataset_name}/{variant}: per-state metrics → {metrics_path}")

    base = os.path.splitext(os.path.basename(csv_path))[0]
    scored_path = os.path.join(output_dir, f"{base}_SCORED.csv")
    summary_path = os.path.join(output_dir, f"{base}_SUMMARY.csv")

    df.to_csv(scored_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"[INFO] Saved scored CSV → {scored_path}")
    print(f"[INFO] Saved summary CSV → {summary_path}")

    return summary_rows

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stepwise end-to-end marginal value experiments using Sanno."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing long-format stepwise CSVs.")
    parser.add_argument("--out_dir", required=True, help="Directory for outputs.")
    args = parser.parse_args()

    file_map = {
        "CD8": "cd8_stepwise_long.csv",
        "CD4": "cd4_stepwise_long.csv",
        "MSC": "msc_stepwise_long.csv",
        "MOUSE_B": "mouse_b_stepwise_long.csv",
    }

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for ds_name, filename in file_map.items():
        csv_path = os.path.join(args.data_dir, filename)
        rows = evaluate_stepwise_dataset(ds_name, csv_path, args.out_dir)
        all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No datasets evaluated.")
        return

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.out_dir, "Stepwise_Sanno_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Global summary → {summary_path}")

if __name__ == "__main__":
    main()