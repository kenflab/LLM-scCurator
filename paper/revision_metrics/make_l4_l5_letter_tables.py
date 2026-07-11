#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

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

    
def parse_bool(x) -> bool:
    if pd.isna(x):
        return False

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) != 0.0

    return str(x).strip().lower() in {
        "true",
        "t",
        "1",
        "1.0",
        "yes",
        "y",
    }


def get_default_weights(dataset: str) -> tuple[float, float]:
    dataset = str(dataset).strip()

    if dataset in DEFAULT_WEIGHTS:
        return DEFAULT_WEIGHTS[dataset]

    lower = dataset.lower()

    if "cd8" in lower:
        return (0.7, 0.3)
    if "cd4" in lower:
        return (0.7, 0.3)
    if "msc" in lower or "caf" in lower:
        return (0.3, 0.7)
    if "mouse" in lower and "b" in lower:
        return (0.5, 0.5)

    raise ValueError(f"Unknown dataset for default S_anno weights: {dataset}")


def filter_quantitative_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preferred filter:
      Included_In_Quantitative_Benchmark == TRUE

    Fallback:
      UsedInConfusion == TRUE

    Important:
      If the preferred column exists but contains no TRUE rows, fall back instead of
      returning an empty table.
    """
    original_n = len(df)

    if "Included_In_Quantitative_Benchmark" in df.columns:
        tmp = df[df["Included_In_Quantitative_Benchmark"].map(parse_bool)].copy()
        print(
            f"Filter Included_In_Quantitative_Benchmark: "
            f"{len(tmp)} / {original_n} rows retained"
        )
        if len(tmp) > 0:
            return tmp

    if "UsedInConfusion" in df.columns:
        tmp = df[df["UsedInConfusion"].map(parse_bool)].copy()
        print(f"Filter UsedInConfusion: {len(tmp)} / {original_n} rows retained")
        if len(tmp) > 0:
            return tmp

    print(
        "WARNING: No benchmark filter retained rows. "
        "Proceeding with all rows instead."
    )
    return df.copy()


def infer_components_from_score(
    score: float,
    w_lineage: float,
    w_state: float,
) -> tuple[float, float]:
    """
    Infer discrete S_anno components from an existing S_anno score.

    Candidate components:
      s_lineage in {0, 0.5, 1}
      s_state   in {0, 1}

    Tie-break rule:
      Prefer major-lineage agreement over isolated state agreement.
      This matters mainly when w_lineage == w_state and score == 0.5.
    """
    if pd.isna(score):
        return (np.nan, np.nan)

    score = float(score)

    candidates = []
    for s_lineage in (0.0, 0.5, 1.0):
        for s_state in (0.0, 1.0):
            expected = w_lineage * s_lineage + w_state * s_state
            error = abs(score - expected)

            # Tie-break:
            # 1. smallest error
            # 2. prefer higher lineage agreement
            # 3. prefer lower state agreement when ambiguous
            candidates.append((error, -s_lineage, s_state, s_lineage, s_state))

    candidates.sort()
    _, _, _, best_lineage, best_state = candidates[0]

    return (best_lineage, best_state)


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


def build_l4_complementary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, sub in df.groupby("Dataset", dropna=False):
        w_lineage, w_state = get_default_weights(dataset)

        for method, score_col in METHODS.items():
            if score_col not in sub.columns:
                continue

            tmp = sub.copy()
            tmp["_score"] = pd.to_numeric(tmp[score_col], errors="coerce")
            tmp = tmp.dropna(subset=["_score"])

            if tmp.empty:
                continue

            components = tmp["_score"].apply(
                lambda x: infer_components_from_score(x, w_lineage, w_state)
            )

            tmp["_s_lineage"] = [x[0] for x in components]
            tmp["_s_state"] = [x[1] for x in components]

            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "N": int(len(tmp)),
                    "Mean_S_anno": float(tmp["_score"].mean()),
                    "Exact_State_Agreement": float((tmp["_s_state"] == 1.0).mean()),
                    "Major_Lineage_Accuracy": float((tmp["_s_lineage"] == 1.0).mean()),
                    "Ontology_Consistent_Accuracy_Sanno_ge_0.5": float(
                        (tmp["_score"] >= 0.5).mean()
                    ),
                    "Low_Consistency_Rate_Sanno_lt_0.5": float(
                        (tmp["_score"] < 0.5).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_l5_weight_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Score_Standard", "Score_Curated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for L5: {sorted(missing)}")

    rows = []

    for dataset, sub in df.groupby("Dataset", dropna=False):
        default_w_lineage, default_w_state = get_default_weights(dataset)

        tmp = sub.copy()
        tmp["_score_standard"] = pd.to_numeric(tmp["Score_Standard"], errors="coerce")
        tmp["_score_curated"] = pd.to_numeric(tmp["Score_Curated"], errors="coerce")
        tmp = tmp.dropna(subset=["_score_standard", "_score_curated"])

        if tmp.empty:
            continue

        std_components = tmp["_score_standard"].apply(
            lambda x: infer_components_from_score(
                x, default_w_lineage, default_w_state
            )
        )
        cur_components = tmp["_score_curated"].apply(
            lambda x: infer_components_from_score(
                x, default_w_lineage, default_w_state
            )
        )

        tmp["_std_s_lineage"] = [x[0] for x in std_components]
        tmp["_std_s_state"] = [x[1] for x in std_components]
        tmp["_cur_s_lineage"] = [x[0] for x in cur_components]
        tmp["_cur_s_state"] = [x[1] for x in cur_components]

        dataset_rows = []

        for scheme_name, w_lineage, w_state in WEIGHT_SCHEMES:
            if scheme_name == "Default_task_specific":
                w_lineage, w_state = default_w_lineage, default_w_state

            standard_alt = (
                w_lineage * tmp["_std_s_lineage"] + w_state * tmp["_std_s_state"]
            )
            curated_alt = (
                w_lineage * tmp["_cur_s_lineage"] + w_state * tmp["_cur_s_state"]
            )

            diff_values = curated_alt.to_numpy(dtype=float) - standard_alt.to_numpy(
                dtype=float
            )

            mean_standard = float(np.nanmean(standard_alt))
            mean_curated = float(np.nanmean(curated_alt))
            mean_diff = float(np.nanmean(diff_values))
            ci_low, ci_high = bootstrap_ci_paired_diff(diff_values)
            p_value = paired_p_value(diff_values)

            dataset_rows.append(
                {
                    "Dataset": dataset,
                    "Weighting_Scheme": scheme_name,
                    "w_lineage": float(w_lineage),
                    "w_state": float(w_state),
                    "N": int(len(tmp)),
                    "Mean_Standard": mean_standard,
                    "Mean_Curated": mean_curated,
                    "Mean_Difference_Curated_minus_Standard": mean_diff,
                    "Bootstrap_95CI_Lower": ci_low,
                    "Bootstrap_95CI_Upper": ci_high,
                    "Paired_P_Value": p_value,
                }
            )

        default_diff = dataset_rows[0]["Mean_Difference_Curated_minus_Standard"]

        for row in dataset_rows:
            row["Direction_Consistent_With_Default"] = same_direction(
                row["Mean_Difference_Curated_minus_Standard"],
                default_diff,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-xlsx", required=True)
    parser.add_argument("--audit-sheet", default="L2_per_cluster_audit")
    parser.add_argument("--out-xlsx", required=True)
    parser.add_argument("--csv-outdir", default=None)
    parser.add_argument("--round", type=int, default=4)

    args = parser.parse_args()

    in_xlsx = Path(args.in_xlsx)
    out_xlsx = Path(args.out_xlsx)

    df = pd.read_excel(in_xlsx, sheet_name=args.audit_sheet)
    print(f"Loaded audit sheet: {args.audit_sheet}, shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    df = clean_audit_table(df)
    print(f"After audit-table cleaning: shape={df.shape}")

    # df = filter_quantitative_benchmark(df)
    print(
        "Benchmark filter: using all cleaned evaluable clusters. "
        "UsedInConfusion is not used for L4/L5 because it is a confusion-matrix flag."
    )
    print(f"After benchmark filtering: shape={df.shape}")

    l4 = build_l4_complementary_metrics(df)
    l5 = build_l5_weight_sensitivity(df)

    float_cols_l4 = l4.select_dtypes(include=["float"]).columns
    float_cols_l5 = l5.select_dtypes(include=["float"]).columns

    l4[float_cols_l4] = l4[float_cols_l4].round(args.round)
    l5[float_cols_l5] = l5[float_cols_l5].round(args.round)

    write_tables_to_excel(in_xlsx, out_xlsx, l4, l5)

    if args.csv_outdir:
        csv_outdir = Path(args.csv_outdir)
        csv_outdir.mkdir(parents=True, exist_ok=True)
        l4.to_csv(csv_outdir / "L4_complementary_metrics.csv", index=False)
        l5.to_csv(csv_outdir / "L5_weight_sensitivity.csv", index=False)

    print("Wrote:")
    print(f"  {out_xlsx}")
    print(f"  sheet: L4_complementary_metrics  shape={l4.shape}")
    print(f"  sheet: L5_weight_sensitivity     shape={l5.shape}")


if __name__ == "__main__":
    main()
