#!/usr/bin/env python3
"""Offline evaluation of recorded CD8 calls with the repository's public scorer.

Only pandas and the Python standard library are required. No LLM backend is
imported, and no network or inference calls are made. Original CSVs are copied
byte-for-byte to the output's source_records directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pandas as pd

PARSER_ID = "public_cd8_config_and_hierarchical_scoring"
INPUT_TYPES = ["Standard", "LLM-scCurator"]
EVALUATION_COLUMNS = [
    "GT_Major", "GT_State", "Pred_Major", "Pred_State", "Parser_Used",
    "Score_Sanno", "Exact_State_Agreement", "Major_Lineage_Match",
    "Ontology_Consistent_Sanno_ge_0.5", "Low_Consistency_Sanno_lt_0.5",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicCD8Evaluator:
    def __init__(self, repo_root: Path):
        self.root = Path(repo_root).resolve()
        self.paths = {
            name: self.root / "benchmarks" / f"{name}.py"
            for name in ("hierarchical_scoring", "cd8_config")
        }
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Required public evaluation file: {path}")
        # Use a private namespace to avoid an unrelated installed 'benchmarks'
        # package. Relative imports within cd8_config still use the local files.
        token = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        package_name = f"_llmsc_cd8_{token}"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(self.root / "benchmarks")]
            sys.modules[package_name] = package
        self.scorer = importlib.import_module(f"{package_name}.hierarchical_scoring")
        self.config_module = importlib.import_module(f"{package_name}.cd8_config")
        self.cfg = self.config_module.CD8_HIER_CFG
        for name, module in (("hierarchical_scoring", self.scorer),
                             ("cd8_config", self.config_module)):
            if Path(module.__file__).resolve() != self.paths[name]:
                raise RuntimeError("Public scoring module was loaded from an unexpected path.")

    def expected(self, ground_truth: str) -> tuple[str, str]:
        gt = str(ground_truth)
        if not any(pattern.lower() in gt.lower() for pattern, _ in self.cfg.gt_rules):
            raise ValueError(f"Ground_Truth has no public CD8 mapping: {gt!r}")
        return self.scorer._expected_major_state_generic(gt, self.cfg)

    def evaluate(self, label: str, ground_truth: str) -> dict:
        # Exactly the normalization used inside score_hierarchical. Do not add
        # lineage from the task context, cluster ID, input type, or reasoning.
        normalized = str(label).lower().replace("*", "").replace("\n", " ").strip()
        gt_major, gt_state = self.expected(ground_truth)
        pred_major = self.scorer._parse_major_lineage_generic(normalized, self.cfg)
        pred_state = self.scorer._parse_state_generic(normalized, self.cfg)
        score = float(self.scorer.score_hierarchical(
            pd.Series({"Ground_Truth": ground_truth, "Parsed_CellType": label}),
            "Parsed_CellType", self.cfg,
        ))
        return {
            "GT_Major": gt_major, "GT_State": gt_state,
            "Pred_Major": pred_major, "Pred_State": pred_state,
            "Parser_Used": PARSER_ID, "Score_Sanno": score,
            "Exact_State_Agreement": pred_state == gt_state,
            "Major_Lineage_Match": pred_major == gt_major,
            # Retain the existing machine-readable column name. In table
            # captions, use 'hierarchy-consistent', not ontology graph distance.
            "Ontology_Consistent_Sanno_ge_0.5": score >= 0.5,
            "Low_Consistency_Sanno_lt_0.5": score < 0.5,
        }

    def provenance(self) -> dict:
        def git(*args):
            try:
                result = subprocess.run(
                    ["git", "-C", str(self.root), *args], capture_output=True,
                    text=True, check=True,
                )
                return result.stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return "unavailable"
        return {
            "Evaluation_Parser_Source": PARSER_ID,
            "Evaluation_Git_Commit": git("rev-parse", "HEAD"),
            "Evaluation_Config_SHA256": sha256(self.paths["cd8_config"]),
            "Evaluation_Scorer_SHA256": sha256(self.paths["hierarchical_scoring"]),
            "Evaluation_Files_Git_Status": git(
                "status", "--porcelain", "--", "benchmarks/cd8_config.py",
                "benchmarks/hierarchical_scoring.py",
            ),
            "Evaluation_Lineage_Weight": self.cfg.w_lineage,
            "Evaluation_State_Weight": self.cfg.w_state,
            "Evaluation_Prediction_Field": "Parsed_CellType (recorded text, unchanged)",
            "Evaluation_Context_Lineage_Inference": "None",
        }


def validate_calls(df: pd.DataFrame, evaluator: PublicCD8Evaluator) -> None:
    required = set(EVALUATION_COLUMNS) | {
        "Dataset", "Cluster_ID", "Ground_Truth", "Input_Type", "Repeat",
        "Genes", "Prompt", "Raw_Output", "Parsed_CellType", "Parse_Status",
        "Backend", "Model_ID", "Parsed_Confidence", "Parsed_Reasoning",
        "Call_Started_At", "Call_Finished_At", "Failure_Type",
    }
    if missing := required - set(df.columns):
        raise ValueError(f"Missing call-record columns: {sorted(missing)}")
    if any(c.startswith("Legacy_") for c in df.columns):
        raise ValueError("Input already contains Legacy_ fields. Use the original call CSV.")
    if len(df) != 102 or df["Cluster_ID"].nunique() != 17:
        raise ValueError("This migration expects the complete 102 calls from 17 CD8 clusters.")
    if set(df["Input_Type"]) != set(INPUT_TYPES):
        raise ValueError("Expected exactly Standard and LLM-scCurator input types.")
    if not df["Cluster_ID"].str.startswith("CD8.").all():
        raise ValueError("Expected CD8 cluster identifiers.")
    repeats = pd.to_numeric(df["Repeat"], errors="raise")
    if set(repeats) != {1, 2, 3}:
        raise ValueError("Expected repeat identifiers 1, 2, and 3.")
    keys = df.assign(_repeat=repeats)
    if keys.duplicated(["Cluster_ID", "Input_Type", "_repeat"]).any():
        raise ValueError("Duplicate cluster/input/repeat records.")
    groups = df.groupby(["Cluster_ID", "Input_Type"], sort=False)
    if len(groups) != 34 or not groups.size().eq(3).all():
        raise ValueError("Expected 34 cluster/input pairs, each with exactly three calls.")
    for column in ["Genes", "Prompt", "Backend", "Model_ID", "Ground_Truth"]:
        if not groups[column].nunique(dropna=False).eq(1).all():
            raise ValueError(f"{column} changes within a repeated-call pair.")
    if not df.groupby("Cluster_ID")["Ground_Truth"].nunique().eq(1).all():
        raise ValueError("Ground_Truth differs between inputs for the same cluster.")
    # The supplied dataset has no failures. Stop if a different dataset would
    # require an explicit decision about failure handling and denominators.
    if not df["Parse_Status"].eq("parsed").all() or df["Failure_Type"].ne("").any():
        raise ValueError("Unexpected parse/backend failures; review before this migration.")
    if df["Parsed_CellType"].str.strip().eq("").any():
        raise ValueError("Empty recorded prediction label.")
    for column in ["Score_Sanno"]:
        numeric = pd.to_numeric(df[column], errors="raise")
        if not numeric.between(0, 1).all():
            raise ValueError(f"Invalid values in {column}.")
    for row in df.itertuples(index=False):
        expected = evaluator.expected(row.Ground_Truth)
        if expected != (row.GT_Major, row.GT_State):
            raise ValueError(f"Reference mapping differs for {row.Cluster_ID}: {expected}")


def rescore(df: pd.DataFrame, evaluator: PublicCD8Evaluator) -> pd.DataFrame:
    validate_calls(df, evaluator)
    out = df.copy()
    for column in EVALUATION_COLUMNS:
        out[f"Legacy_{column}"] = out[column]
    evaluations = pd.DataFrame([
        evaluator.evaluate(row.Parsed_CellType, row.Ground_Truth)
        for row in df.itertuples(index=False)
    ], index=df.index)
    for column in EVALUATION_COLUMNS:
        out[column] = evaluations[column]
    out["Score_Changed"] = (
        out["Score_Sanno"] - pd.to_numeric(out["Legacy_Score_Sanno"])
    ).abs() > 1e-12
    # Verify every non-evaluation field, including the full raw responses.
    retained = [c for c in df.columns if c not in EVALUATION_COLUMNS]
    if not out[retained].equals(df[retained]):
        raise AssertionError("A non-evaluation field was modified.")
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = []
    for (cluster, input_type), sub in df.groupby(["Cluster_ID", "Input_Type"], sort=False):
        labels = sub["Parsed_CellType"].astype(str).tolist()
        count = len(labels)
        same_pairs = sum(labels[i] == labels[j] for i in range(count) for j in range(i + 1, count))
        pairs.append({
            "Cluster_ID": cluster, "Input_Type": input_type, "N_Calls": count,
            "N_Unique_Labels": len(set(labels)),
            "Pairwise_Label_Agreement": same_pairs / (count * (count - 1) / 2),
            "Mean_Sanno": sub["Score_Sanno"].mean(),
            "Within_Pair_SD_Sanno": sub["Score_Sanno"].std(ddof=1),
            "Legacy_Mean_Sanno": pd.to_numeric(sub["Legacy_Score_Sanno"]).mean(),
        })
    pairs = pd.DataFrame(pairs)
    rows = []
    for input_type in INPUT_TYPES:
        sub = df[df["Input_Type"] == input_type]
        pair = pairs[pairs["Input_Type"] == input_type]
        rows.append({
            "Summary_Row": input_type,
            "Backend": sub["Backend"].iloc[0], "Model_ID": sub["Model_ID"].iloc[0],
            "N_Clusters": len(pair), "N_Calls": len(sub),
            "N_Repeats_Per_Cluster_Input": int(pair["N_Calls"].iloc[0]),
            "Parse_Failure_Rate": sub["Parse_Status"].ne("parsed").mean(),
            "Mean_Sanno_Across_Calls": sub["Score_Sanno"].mean(),
            "Mean_Sanno_By_Cluster": pair["Mean_Sanno"].mean(),
            "SD_Sanno_Across_Calls": sub["Score_Sanno"].std(ddof=1),
            "Mean_Within_Cluster_Pairwise_Label_Agreement": pair["Pairwise_Label_Agreement"].mean(),
            "All_Repeats_Same_Label_Rate": pair["N_Unique_Labels"].eq(1).mean(),
            "Exact_State_Agreement_Across_Calls": sub["Exact_State_Agreement"].mean(),
            "Major_Lineage_Accuracy_Across_Calls": sub["Major_Lineage_Match"].mean(),
            "Ontology_Consistent_Accuracy_Sanno_ge_0.5": sub["Ontology_Consistent_Sanno_ge_0.5"].mean(),
            "Low_Consistency_Rate_Sanno_lt_0.5": sub["Low_Consistency_Sanno_lt_0.5"].mean(),
            "Mean_Within_Cluster_SD_Sanno": pair["Within_Pair_SD_Sanno"].mean(),
            "N_Calls_Score_Changed": int(sub["Score_Changed"].sum()),
            "Legacy_Mean_Sanno_Across_Calls": pd.to_numeric(sub["Legacy_Score_Sanno"]).mean(),
        })
    paired_scores = pairs.pivot(index="Cluster_ID", columns="Input_Type", values="Mean_Sanno")
    delta = paired_scores["LLM-scCurator"] - paired_scores["Standard"]
    rows.append({
        "Summary_Row": "LLM-scCurator_minus_Standard",
        "Backend": df["Backend"].iloc[0], "Model_ID": df["Model_ID"].iloc[0],
        "N_Clusters": len(delta), "N_Repeats_Per_Cluster_Input": 3,
        "Mean_Delta_Sanno_By_Cluster": delta.mean(),
        "Median_Delta_Sanno_By_Cluster": delta.median(),
        "N_Clusters_Delta_Positive": int((delta > 1e-12).sum()),
        "N_Clusters_Delta_Zero": int((delta.abs() <= 1e-12).sum()),
        "N_Clusters_Delta_Negative": int((delta < -1e-12).sum()),
    })
    return pd.DataFrame(rows), pairs


def read_csv(path: Path) -> pd.DataFrame:
    # Do not turn literal model output such as 'NA' into a missing value.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--calls-csv", required=True, type=Path)
    parser.add_argument("--legacy-summary-csv", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    source_paths = [args.metadata_csv, args.calls_csv]
    if args.legacy_summary_csv:
        source_paths.append(args.legacy_summary_csv)
    for path in source_paths:
        if not path.is_file():
            parser.error(f"Source file not found: {path}")
    if args.outdir.exists():
        parser.error("Output directory already exists. Choose a new directory to preserve previous results.")

    evaluator = PublicCD8Evaluator(args.repo_root)
    original = read_csv(args.calls_csv)
    updated = rescore(original, evaluator)
    summary, pairs = summarize(updated)
    metadata = read_csv(args.metadata_csv)
    if len(metadata) != 1 or "Parser_Source" not in metadata.columns:
        parser.error("Expected the original one-row local runtime metadata CSV, not the formatted L6 worksheet.")
    if "Legacy_Parser_Source" in metadata.columns:
        parser.error("Metadata already rescored; select the original CSV.")
    metadata["Legacy_Parser_Source"] = metadata["Parser_Source"]
    metadata["Parser_Source"] = PARSER_ID
    provenance = evaluator.provenance()
    provenance.update({
        "Rescoring_Date_UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Rescoring_Script_SHA256": sha256(Path(__file__)),
        "Rescoring_Python": sys.version.split()[0],
        "Rescoring_Pandas": pd.__version__,
        "Rescoring_Mode": "Offline; recorded calls only; no new inference",
        "Rescoring_Source_Calls_SHA256": sha256(args.calls_csv),
        "Rescoring_Source_Metadata_SHA256": sha256(args.metadata_csv),
        "Rescoring_N_Calls": len(updated),
        "Rescoring_N_Changed_Scores": int(updated["Score_Changed"].sum()),
        "Rescoring_Unit_For_Paired_Comparison": "17 clusters; repeats averaged within cluster and input",
        "Rescoring_Stability_Definition": "Exact equality of recorded Parsed_CellType within each cluster/input pair",
        "Rescoring_Interpretation": "Label-based scoring; state-only labels may lack a recognized lineage alias",
    })
    for key, value in provenance.items():
        metadata[key] = value

    out = args.outdir
    out.mkdir(parents=True, exist_ok=False)
    sources = out / "source_records"
    sources.mkdir()
    # Copy originals, including their original names, without editing them.
    for path in source_paths:
        shutil.copy2(path, sources / path.name)
    metadata.to_csv(out / "L6_llm_inference_metadata.csv", index=False)
    updated.to_csv(out / "L7_repeated_run_robustness.csv", index=False)
    summary.to_csv(out / "L8_local_open_weight_backend_summary.csv", index=False)
    summary.set_index("Summary_Row").T.rename_axis("Summary_Row").to_csv(out / "L8_for_letter.csv")
    pairs.to_csv(out / "cluster_input_audit.csv", index=False)
    pd.DataFrame({"Metadata field": list(provenance), "Local repeated-run analysis": list(provenance.values())}).to_csv(
        out / "L6_rescoring_addendum.csv", index=False,
    )
    # CSV round-trip check of all preserved non-evaluation fields.
    saved = read_csv(out / "L7_repeated_run_robustness.csv")
    retained = [c for c in original.columns if c not in EVALUATION_COLUMNS]
    if not saved[retained].equals(original[retained]):
        raise AssertionError("CSV round-trip changed an original non-evaluation field.")
    report = {
        "validation": "passed", "calls": len(updated),
        "clusters": updated["Cluster_ID"].nunique(), "cluster_input_pairs": len(pairs),
        "scores_changed": int(updated["Score_Changed"].sum()),
        "all_pairs_have_three_calls": bool(pairs["N_Calls"].eq(3).all()),
        "all_pairs_have_one_unique_label": bool(pairs["N_Unique_Labels"].eq(1).all()),
        "original_non_evaluation_fields_preserved": True,
        "source_sha256": {p.name: sha256(p) for p in source_paths},
        "evaluation_provenance": provenance,
        "output_sha256": {p.name: sha256(p) for p in sorted(out.glob("*.csv"))},
    }
    (out / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(summary.loc[:1, ["Summary_Row", "N_Clusters", "N_Calls", "Mean_Sanno_Across_Calls",
                           "Legacy_Mean_Sanno_Across_Calls", "All_Repeats_Same_Label_Rate"]].to_string(index=False))
    print(f"Changed scores: {report['scores_changed']}/{len(updated)}")
    print(f"Original non-evaluation fields preserved: {report['original_non_evaluation_fields_preserved']}")
    print(f"Results written to: {out.resolve()}")


if __name__ == "__main__":
    main()
