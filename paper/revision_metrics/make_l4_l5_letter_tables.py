#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.cd8_config import CD8_HIER_CFG
from benchmarks.cd4_config import CD4_HIER_CFG
from benchmarks.caf_config import CAF_HIER_CFG
from benchmarks.mouse_b_config import MOUSE_B_CFG
from benchmarks.hierarchical_scoring import (
    _expected_major_state_generic,
    _parse_major_lineage_generic,
    _parse_state_generic,
    score_hierarchical,
)

CONFIGS = {"CD8 T": CD8_HIER_CFG, "CD4 T": CD4_HIER_CFG,
           "MSC": CAF_HIER_CFG, "Mouse B": MOUSE_B_CFG}
EXPECTED_CLUSTERS = {"CD8 T": 17, "CD4 T": 22, "MSC": 8, "Mouse B": 5}
PREDICTION_COLUMNS = {
    "Standard": ("Standard_CellType", "Standard_Answer"),
    "Curated": ("Full_Pipeline_CellType", "Full_Pipeline_Answer", "Curated_Answer"),
    "CellTypist": ("CellTypist_Answer",),
    "SingleR": ("SingleR_Answer",),
    "Azimuth": ("Azimuth_Answer",),
}

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


METHODS = {
    "Standard": "Score_Standard",
    "Curated": "Score_Curated",
    "CellTypist": "Score_CellTypist",
    "SingleR": "Score_SingleR",
    "Azimuth": "Score_Azimuth",
}


DEFAULT_WEIGHTS = {
    "CD8 T": (0.7, 0.3),
    "CD4 T": (0.7, 0.3),
    "MSC": (0.3, 0.7),
    "Mouse B": (0.5, 0.5),
}


WEIGHT_SCHEMES = [
    ("Default_task_specific", None, None),
    ("Equal_0.5_0.5", 0.5, 0.5),
    ("Lineage_heavy_0.8_0.2", 0.8, 0.2),
    ("State_heavy_0.3_0.7", 0.3, 0.7),
    ("Major_lineage_only", 1.0, 0.0),
    ("Exact_state_only", 0.0, 1.0),
]


def canonicalize_dataset(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    low = s.lower()

    if low.startswith("note."):
        return np.nan

    if s in DEFAULT_WEIGHTS:
        return s

    if "cd8" in low:
        return "CD8 T"
    if "cd4" in low:
        return "CD4 T"
    if "msc" in low:
        return "MSC"
    if "mouse" in low and "b" in low:
        return "Mouse B"

    return np.nan


def clean_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df["Dataset"] = df["Dataset"].map(canonicalize_dataset)
    df = df[df["Dataset"].notna()].copy()
    print(f"Clean dataset rows: {len(df)} / {before} rows retained")

    score_cols = [
        "Score_Standard",
        "Score_Curated",
        "Score_CellTypist",
        "Score_SingleR",
        "Score_Azimuth",
    ]

    existing_score_cols = [c for c in score_cols if c in df.columns]

    for c in existing_score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before_score = len(df)
    df = df[df[existing_score_cols].notna().any(axis=1)].copy()
    print(f"Rows with at least one numeric score: {len(df)} / {before_score}")

    return df

    
def parse_prediction_records(df: pd.DataFrame) -> pd.DataFrame:
    """Parse saved labels directly; Sanno cannot identify its components uniquely."""
    records = []
    for _, row in df.iterrows():
        cfg = CONFIGS[row["Dataset"]]
        if pd.isna(row["Ground_Truth"]) or not str(row["Ground_Truth"]).strip():
            raise ValueError(f"Missing reference label for {row['Cluster_ID']}")
        gt_major, gt_state = _expected_major_state_generic(row["Ground_Truth"], cfg)
        for method, score_col in METHODS.items():
            if score_col not in row or pd.isna(row[score_col]):
                continue  # An unavailable comparator is not a failed prediction.
            candidates = [
                str(row[c]) for c in PREDICTION_COLUMNS[method]
                if c in row and pd.notna(row[c]) and str(row[c]).strip()
            ]
            if not candidates:
                raise ValueError(f"Missing {method} prediction for {row['Cluster_ID']}")
            if len(set(candidates)) != 1:
                raise ValueError(f"Conflicting {method} prediction fields for {row['Cluster_ID']}")
            prediction = candidates[0]
            # Use the same normalization as score_hierarchical().
            normalized = prediction.lower().replace("*", "").replace("\n", " ").strip()
            major = _parse_major_lineage_generic(normalized, cfg)
            state = _parse_state_generic(normalized, cfg)
            scoring_row = {"Ground_Truth": row["Ground_Truth"], "Prediction": prediction}
            score = score_hierarchical(scoring_row, "Prediction", cfg)
            recorded = float(row[score_col])
            if not np.isclose(score, recorded, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Sanno mismatch: {row['Dataset']} {row['Cluster_ID']} {method}: "
                    f"recorded={recorded}, public={score}. Check the input and scorer version."
                )
            records.append({
                "Dataset": row["Dataset"], "Cluster_ID": row["Cluster_ID"],
                "Method": method, "Ground_Truth": row["Ground_Truth"],
                "Raw_Prediction": prediction,
                "GT_Major": gt_major, "GT_State": gt_state,
                "Pred_Major": major, "Pred_State": state,
                "Exact_State_Agreement": state == gt_state,
                "Major_Lineage_Accuracy": major == gt_major,
                "Exact_Match_Accuracy": state == gt_state and major == gt_major,
                "Recorded_Sanno": recorded, "Public_Sanno": score,
                "Ontology_Consistent_Accuracy_Sanno_ge_0.5": score >= 0.5,
                "Low_Consistency_Rate_Sanno_lt_0.5": score < 0.5,
            })
    return pd.DataFrame(records)


def validate_primary_design(df: pd.DataFrame) -> None:
    if df.duplicated(["Dataset", "Cluster_ID"]).any():
        raise ValueError("Duplicate benchmark cluster identifiers")
    observed = df.groupby("Dataset").size().to_dict()
    if observed != EXPECTED_CLUSTERS:
        raise ValueError(f"Expected all 52 predefined clusters: {EXPECTED_CLUSTERS}; got {observed}")
    if df[["Score_Standard", "Score_Curated"]].isna().any().any():
        raise ValueError("Standard and Full pipeline scores must be present for all 52 clusters")


def bootstrap_ci_paired_diff(
    diff_values: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    diff_values = np.asarray(diff_values, dtype=float)
    diff_values = diff_values[~np.isnan(diff_values)]

    if len(diff_values) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n = len(diff_values)
    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diff_values[idx].mean()

    low, high = np.percentile(boot_means, [2.5, 97.5])
    return (float(low), float(high))


def paired_p_value(diff_values: np.ndarray) -> float:
    diff_values = np.asarray(diff_values, dtype=float)
    diff_values = diff_values[~np.isnan(diff_values)]

    if len(diff_values) == 0:
        return np.nan

    if np.allclose(diff_values, 0):
        return 1.0

    if wilcoxon is None:
        return np.nan

    try:
        return float(wilcoxon(diff_values, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def same_direction(diff: float, default_diff: float, eps: float = 1e-12) -> bool:
    if pd.isna(diff) or pd.isna(default_diff):
        return False

    if abs(default_diff) < eps:
        return abs(diff) < eps

    return diff * default_diff >= -eps


def build_l4_complementary_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in records.groupby("Dataset", sort=True):
        for method in METHODS:
            tmp = sub[sub["Method"] == method]
            if tmp.empty:
                continue
            result = {"Dataset": dataset, "Method": method, "N": len(tmp),
                      "Mean_S_anno": float(tmp["Public_Sanno"].mean())}
            for metric in ["Exact_State_Agreement", "Exact_Match_Accuracy",
                           "Major_Lineage_Accuracy",
                           "Ontology_Consistent_Accuracy_Sanno_ge_0.5",
                           "Low_Consistency_Rate_Sanno_lt_0.5"]:
                result[metric] = float(tmp[metric].mean())
            rows.append(result)
    return pd.DataFrame(rows)


def build_l5_weight_sensitivity(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in records.groupby("Dataset", sort=True):
        cfg = CONFIGS[dataset]
        std = sub[sub["Method"] == "Standard"].set_index("Cluster_ID")
        cur = sub[sub["Method"] == "Curated"].set_index("Cluster_ID")
        if std.empty or set(std.index) != set(cur.index):
            raise ValueError(f"Incomplete Standard/Full pipeline pairs for {dataset}")
        cur = cur.loc[std.index]
        dataset_rows = []
        for scheme_name, w_lineage, w_state in WEIGHT_SCHEMES:
            if scheme_name == "Default_task_specific":
                w_lineage, w_state = cfg.w_lineage, cfg.w_state
            alternate_cfg = replace(cfg, w_lineage=w_lineage, w_state=w_state)
            # Re-score labels with unchanged aliases, near-lineage rules and penalties.
            standard_alt = std.apply(
                lambda r: score_hierarchical(r, "Raw_Prediction", alternate_cfg), axis=1
            ).to_numpy(dtype=float)
            curated_alt = cur.apply(
                lambda r: score_hierarchical(r, "Raw_Prediction", alternate_cfg), axis=1
            ).to_numpy(dtype=float)
            diff_values = curated_alt - standard_alt
            ci_low, ci_high = bootstrap_ci_paired_diff(diff_values)
            dataset_rows.append({
                "Dataset": dataset, "Weighting_Scheme": scheme_name,
                "w_lineage": float(w_lineage), "w_state": float(w_state),
                "N": len(std), "Mean_Standard": float(standard_alt.mean()),
                "Mean_Curated": float(curated_alt.mean()),
                "Mean_Difference_Curated_minus_Standard": float(diff_values.mean()),
                "Bootstrap_95CI_Lower": ci_low, "Bootstrap_95CI_Upper": ci_high,
                "Paired_P_Value": paired_p_value(diff_values),
            })
        default_diff = dataset_rows[0]["Mean_Difference_Curated_minus_Standard"]
        for row in dataset_rows:
            row["Direction_Consistent_With_Default"] = same_direction(
                row["Mean_Difference_Curated_minus_Standard"], default_diff
            )
        rows.extend(dataset_rows)
    return pd.DataFrame(rows)


def write_tables_to_excel(
    in_xlsx: Path,
    out_xlsx: Path,
    l4: pd.DataFrame,
    l5: pd.DataFrame,
) -> None:
    if not in_xlsx.exists():
        raise FileNotFoundError(f"Input Excel file not found: {in_xlsx}")

    if in_xlsx.resolve() != out_xlsx.resolve():
        shutil.copyfile(in_xlsx, out_xlsx)

    with pd.ExcelWriter(
        out_xlsx,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        l4.to_excel(writer, sheet_name="L4_complementary_metrics", index=False)
        l5.to_excel(writer, sheet_name="L5_weight_sensitivity", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute primary benchmark metrics from saved labels.")
    parser.add_argument("--in-xlsx", required=True)
    parser.add_argument("--audit-sheet", default="L2_per_cluster_audit")
    parser.add_argument("--out-xlsx", help="Optional new workbook containing updated L4/L5 sheets")
    parser.add_argument("--csv-outdir", help="Directory for full-precision CSV tables and evaluation records")
    parser.add_argument("--round", type=int, default=4, help="Workbook precision; CSV files retain full precision")
    args = parser.parse_args()
    if not args.out_xlsx and not args.csv_outdir:
        parser.error("Supply --out-xlsx and/or --csv-outdir")
    in_xlsx = Path(args.in_xlsx)
    if args.out_xlsx and Path(args.out_xlsx).resolve() == in_xlsx.resolve():
        parser.error("--out-xlsx must differ from --in-xlsx")
    df = clean_audit_table(pd.read_excel(in_xlsx, sheet_name=args.audit_sheet))
    validate_primary_design(df)
    print("Using all 52 predefined clusters; UsedInConfusion does not filter quantitative results.")
    records = parse_prediction_records(df)
    l4 = build_l4_complementary_metrics(records)
    l5 = build_l5_weight_sensitivity(records)
    if args.out_xlsx:
        out_xlsx = Path(args.out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        write_tables_to_excel(in_xlsx, out_xlsx, l4.round(args.round), l5.round(args.round))
        print(f"Wrote {out_xlsx}")
    if args.csv_outdir:
        outdir = Path(args.csv_outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        l4.to_csv(outdir / "L4_complementary_metrics.csv", index=False)
        l5.to_csv(outdir / "L5_weight_sensitivity.csv", index=False)
        records.to_csv(outdir / "L4_prediction_audit.csv", index=False)
        sources = [Path(__file__), *[REPO_ROOT / "benchmarks" / f for f in
                   ["cd8_config.py", "cd4_config.py", "caf_config.py",
                    "mouse_b_config.py", "hierarchical_scoring.py"]]]
        metadata = {
            "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
            "input_xlsx": str(in_xlsx.resolve()),
            "input_sha256": hashlib.sha256(in_xlsx.read_bytes()).hexdigest(),
            "audit_sheet": args.audit_sheet, "n_clusters": len(df),
            "n_predictions_checked": len(records), "new_inference_calls": 0,
            "recorded_sanno_reproduced": True,
            "bootstrap_resamples": 10000, "bootstrap_seed": 42,
            "source_sha256": {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
        }
        (outdir / "L4_L5_evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote full-precision L4/L5 tables and {len(records)} prediction records to {outdir}")
    for method in ["Standard", "Curated"]:
        sub = l4[l4.Method == method]
        mean = np.average(sub.Exact_State_Agreement, weights=sub.N)
        print(f"{method}: exact state agreement = {mean:.6f}")


if __name__ == "__main__":
    main()
