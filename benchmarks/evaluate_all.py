"""
Ontology-aware evaluation for Figure 2 benchmarks.

This script evaluates multiple annotation pipelines using a hierarchy-aware scoring
scheme that tolerates near-misses within the same major lineage/state ontology.

Key design choices
------------------
- Ground-truth is derived deterministically from cluster identifiers using `gt_mappings.py`
  (single source of truth for label harmonization).
- Scoring is ontology-aware via per-task hierarchy configs (e.g., CD8/CD4/MSC/Mouse B),
  enabling partial credit for correct major lineage with imperfect subtype.
- Outputs include:
  (i) per-dataset scored CSVs (row-wise scores for each pipeline),
  (ii) summary table used for Fig. 2, and
  (iii) confusion matrices + per-state precision/recall/F1 for Standard and Curated.

Notes
-----
This script is intended for reproducible benchmarking. It does not call any LLMs and
does not modify upstream benchmark inputs; it only reads integrated CSVs and writes
derived evaluation artifacts.
"""


import os
import argparse
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

# ---- dataset-specific scoring wrappers --------------------------


def _score_cd8(row, col, cfg):
    return score_hierarchical(row, col, cfg)


def _score_cd4(row, col, cfg):
    return score_hierarchical(row, col, cfg)


def _score_msc(row, col, cfg):
    # `cfg` is unused for MSC, but kept to unify the wrapper signature across tasks.
    return score_caf_hierarchical(row, col)


def _score_mouse_b(row, col, cfg):
    # `cfg` is unused for Mouse B; retained for a unified wrapper signature.
    return score_mouse_b(row, col)


# ---- task configuration ------------------------------------------------------

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

METHODS_TO_EVALUATE = [
    "Standard_Answer",
    "Curated_Answer",
    "CellTypist_Answer",
    "SingleR_Answer",
    "Azimuth_Answer",
]

# ---- confusion matrix helpers -----------------------------------------------


def _build_state_series(df, cfg, pred_col):
    """
    Ontology-aware evaluation for Figure 2 benchmarks.

    This script evaluates multiple annotation pipelines using a hierarchy-aware scoring
    scheme that tolerates near-misses within the same major lineage/state ontology.

    Key design choices
    ------------------
    - Ground-truth is derived deterministically from cluster identifiers using `gt_mappings.py`
      (single source of truth for label harmonization).
    - Scoring is ontology-aware via per-task hierarchy configs (e.g., CD8/CD4/MSC/Mouse B),
      enabling partial credit for correct major lineage with imperfect subtype.
    - Outputs include:
      (i) per-dataset scored CSVs (row-wise scores for each pipeline),
      (ii) summary table used for Fig. 2, and
      (iii) confusion matrices + per-state precision/recall/F1 for Standard and Curated.

    Notes
    -----
    This script is intended for reproducible benchmarking. It does not call any LLMs and
    does not modify upstream benchmark inputs; it only reads integrated CSVs and writes
    derived evaluation artifacts.
    """

    gt_states = []
    pred_states = []

    for _, row in df.iterrows():
        gt_major, gt_state = _expected_major_state_generic(row["Ground_Truth"], cfg)
        if gt_state == cfg.default_state:
            continue  # skip unmapped states
        pred_state = _parse_state_generic(str(row[pred_col]), cfg)
        gt_states.append(gt_state)
        pred_states.append(pred_state)

    if not gt_states:
        return None, None

    gt_series = pd.Series(gt_states, name="GT_state")
    pred_series = pd.Series(pred_states, name="Pred_state")
    return gt_series, pred_series


def _confusion_and_per_state_metrics(gt, pred):
    """
    Compute a confusion matrix and per-state precision/recall/F1 metrics.

    Parameters
    ----------
    gt : pandas.Series
        Ground-truth state labels.
    pred : pandas.Series
        Predicted state labels.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        cm : Confusion matrix (rows = GT, cols = Pred).
        metrics_df : Per-state metrics with columns {State, Support, Precision, Recall, F1}.
    """

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
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        rows.append(
            {
                "State": lab,
                "Support": support,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("State")
    return cm, metrics_df


# ---- core evaluation ---------------------------------------------------------


def evaluate_dataset(dataset_name, csv_path, output_dir):
    """
    Evaluate one benchmark dataset with ontology-aware hierarchical scoring.

    Workflow
    --------
    1) Load the integrated per-cluster CSV for the given dataset.
    2) Derive `Ground_Truth` deterministically from the cluster identifier using a
       dataset-specific GT mapping function (single source of truth).
    3) For each available pipeline column, compute a per-row hierarchical score
       (0.0–1.0) using a task-specific hierarchy config.
    4) Write:
       - a scored CSV with appended score columns, and
       - confusion matrices + per-state metrics for Standard and Curated pipelines.

    Parameters
    ----------
    dataset_name : str
        Dataset key. Must match an entry in `TASKS` (e.g., "CD8", "CD4", "MSC", "MOUSE_B").
    csv_path : str
        Path to the integrated benchmark CSV for this dataset.
    output_dir : str
        Directory where evaluation artifacts will be written.

    Returns
    -------
    list[dict]
        Rows for the Figure 2 summary table. Each dict contains:
        {Dataset, Pipeline, N, MeanScore, FracScoreEq1, FracScoreGe0_5}.

    Notes
    -----
    - The script prefers cleaned subtype columns (`*_Subtype_Clean`) when available,
      falling back to raw answer columns (`*_Answer`) otherwise.
    - Confusion matrices are computed only for pipelines {Standard, Curated} to keep
      reviewer-facing outputs concise and comparable.
    """

    print(f"\n[INFO] Processing {dataset_name} from {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return []

    task_spec = TASKS.get(dataset_name)
    if not task_spec:
        print(f"[WARN] No config found for {dataset_name}")
        return []

    df = pd.read_csv(csv_path)

    # 1) choose cluster column
    if "Cluster_ID" in df.columns:
        cluster_col = "Cluster_ID"
    elif "meta.cluster" in df.columns:
        cluster_col = "meta.cluster"
    else:
        cluster_col = df.columns[0]  # fallback

    gt_mapper = task_spec["gt_mapper"]
    cfg = task_spec["cfg"]
    score_func = task_spec["score_func"]

    print(f"  > Applying GT mapper ({gt_mapper.__name__}) using column '{cluster_col}'")
    df["Ground_Truth"] = df[cluster_col].apply(gt_mapper)

    # 2) add GT major/state for transparency (and to mark which rows go into CM)
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

    # 3) scoring loop
    metrics_rows = []
    cm_inputs = {}  # pipeline name -> column name of textual prediction

    for method in METHODS_TO_EVALUATE:
        subtype_col = method.replace("Answer", "Subtype_Clean")
        target_col = None

        if subtype_col in df.columns:
            target_col = subtype_col
        elif method in df.columns:
            target_col = method

        if target_col is None:
            continue

        pipeline_name = method.replace("_Answer", "")
        score_col = f"Score_{pipeline_name}"

        df[score_col] = df.apply(lambda row: score_func(row, target_col, cfg), axis=1)

        scores = df[score_col].astype(float)
        mean_score = float(scores.mean())
        perfect = float((scores == 1.0).mean())
        frac_ge_0_5 = float((scores >= 0.5).mean())

        print(
            f"  > {pipeline_name:15s}: "
            f"Mean={mean_score:.3f}, "
            f"Frac==1={perfect:.3f}, "
            f"Frac>=0.5={frac_ge_0_5:.3f}"
        )

        metrics_rows.append(
            {
                "Dataset": dataset_name,
                "Pipeline": pipeline_name,
                "N": int(len(scores)),
                "MeanScore": mean_score,
                "FracScoreEq1": perfect,
                "FracScoreGe0_5": frac_ge_0_5,
            }
        )

        if pipeline_name in {"Standard", "Curated"}:
            cm_inputs[pipeline_name] = target_col

    # 4) save scored CSV
    base = os.path.splitext(os.path.basename(csv_path))[0]
    scored_path = os.path.join(output_dir, f"{base}_SCORED.csv")
    df.to_csv(scored_path, index=False)
    print(f"[INFO] Saved detailed scores → {scored_path}")

    # 5) confusion matrices (Standard / Curated)
    for pipeline, pred_col in cm_inputs.items():
        gt_series, pred_series = _build_state_series(df, cfg, pred_col)
        if gt_series is None:
            print(f"[WARN] {dataset_name}/{pipeline}: no valid GT states for confusion.")
            continue

        cm, metrics_df = _confusion_and_per_state_metrics(gt_series, pred_series)

        cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_{pipeline}.csv")
        metrics_path = os.path.join(
            output_dir, f"{dataset_name}_per_state_metrics_{pipeline}.csv"
        )
        cm.to_csv(cm_path)
        metrics_df.to_csv(metrics_path, index=False)

        print(f"[INFO] {dataset_name}/{pipeline}: confusion → {cm_path}")
        print(f"[INFO] {dataset_name}/{pipeline}: per-state metrics → {metrics_path}")

    return metrics_rows


# ---- CLI ---------------------------------------------------------------------


def main():
    """
    CLI entry point.

    Reads integrated benchmark CSVs from --data_dir, evaluates all datasets, and writes:
    - per-dataset scored CSVs,
    - confusion matrices (Standard/Curated),
    - Figure2_Benchmark_Summary.csv in --out_dir.
    """

    parser = argparse.ArgumentParser(
        description="Ontology-aware evaluation for LLM-scCurator benchmarks (Fig.2)."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the benchmark CSVs.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where outputs will be written.",
    )
    args = parser.parse_args()

    file_map = {
        "CD8": "cd8_benchmark_results_integrated.csv",
        "CD4": "cd4_benchmark_results_integrated.csv",
        "MSC": "msc_benchmark_results_integrated.csv",
        "MOUSE_B": "mouse_b_benchmark_results_integrated.csv",
    }

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for ds_name, filename in file_map.items():
        csv_path = os.path.join(args.data_dir, filename)
        rows = evaluate_dataset(ds_name, csv_path, args.out_dir)
        all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No datasets evaluated; summary not written.")
        return

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.out_dir, "Figure2_Benchmark_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Summary table → {summary_path}")


if __name__ == "__main__":
    main()
