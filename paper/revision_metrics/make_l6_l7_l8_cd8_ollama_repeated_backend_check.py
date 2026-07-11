#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROMPT_TEMPLATE_ID = "cd8_ollama_local_json_v1"


PROMPT_TEMPLATE = """You are assisting with cell-type annotation for a tumor-infiltrating CD8+ T-cell single-cell RNA-seq benchmark.

You will receive a ranked marker-gene list for one cluster. Use the marker genes to infer the most likely concise cell-state label.

Important constraints:
- Return only one JSON object.
- Do not use markdown.
- Do not include text outside JSON.
- Use exactly these keys: cell_type, confidence, reasoning.
- confidence must be one of: High, Medium, Low.
- reasoning should be brief and mention the main marker evidence.
- If the evidence is ambiguous, choose the most plausible CD8+ T-cell state and use Medium or Low confidence.

Dataset context:
- Tumor-infiltrating CD8+ T cells.
- Relevant states may include naive/central-memory-like, effector, effector-memory, exhausted, interferon-stimulated/ISG-high, MAIT-like, cycling/proliferating, and NK-like cytotoxic states.

Cluster ID: {cluster_id}
Input type: {input_type}
Ranked marker genes:
{genes}

Return only JSON in this schema:
{{"cell_type": "...", "confidence": "High|Medium|Low", "reasoning": "..."}}
"""


STATE_ORDER = [
    "Naive",
    "Effector",
    "EffectorMemory",
    "Exhausted",
    "ISG",
    "MAIT",
    "NK_killer",
    "Cycling",
    "Other",
    "Unknown",
]


def run_cmd(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        txt = (out.stdout or "") + ("\n" + out.stderr if out.stderr else "")
        return txt.strip()
    except Exception as e:
        return f"ERROR: {repr(e)}"


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def split_genes(x: Any) -> list[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = re.split(r"[;,]\s*|\s+", s)
    out = []
    seen = set()
    for p in parts:
        p = str(p).strip()
        if not p:
            continue
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def clean_json_text(raw: str) -> str:
    s = str(raw).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return s.strip()


def parse_llm_json(raw: str) -> tuple[dict[str, str], str, str]:
    try:
        obj = json.loads(clean_json_text(raw))
        if not isinstance(obj, dict):
            raise ValueError("Parsed JSON is not an object.")

        cell_type = str(obj.get("cell_type", "Unknown")).strip() or "Unknown"
        confidence = str(obj.get("confidence", "Low")).strip() or "Low"
        reasoning = str(obj.get("reasoning", "")).strip()

        if confidence not in {"High", "Medium", "Low"}:
            confidence = "Low"

        return (
            {
                "cell_type": cell_type,
                "confidence": confidence,
                "reasoning": reasoning,
            },
            "parsed",
            "",
        )
    except Exception as e:
        return (
            {
                "cell_type": "ParseError",
                "confidence": "Low",
                "reasoning": str(raw)[:1000],
            },
            "parse_error",
            repr(e),
        )


def normalize_text(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def fallback_parse_cd8_major_state(prediction_text: str) -> tuple[str, str]:
    """
    Conservative fallback parser for the CD8 local-backend check.
    The script first tries to use the existing parser from make_l4_l5_letter_tables.py.
    This fallback is used only if that parser cannot be loaded.
    """
    s = normalize_text(prediction_text)

    non_t_patterns = {
        "B": [" b cell", "plasma", "immunoglobulin", "cd79a", "ms4a1"],
        "Myeloid": ["monocyte", "macrophage", "myeloid", "dc", "dendritic", "neutrophil"],
        "Stromal": ["fibroblast", "caf", "endothelial", "pericyte", "smooth muscle"],
        "Epithelial": ["epithelial", "tumor", "carcinoma", "malignant"],
        "Erythroid": ["erythroid", "red blood", "rbc"],
    }

    for major, pats in non_t_patterns.items():
        if any(p.strip() in s for p in pats):
            return major, "Other"

    if "mait" in s or "mucosal associated invariant" in s:
        return "T", "MAIT"

    if (
        "isg" in s
        or "interferon" in s
        or "ifn" in s
        or "ifit" in s
        or "isg15" in s
        or "mx1" in s
    ):
        return "T", "ISG"

    if (
        "cycling" in s
        or "proliferating" in s
        or "cell cycle" in s
        or "mki67" in s
        or "top2a" in s
    ):
        return "T", "Cycling"

    if (
        "exhaust" in s
        or "tex" in s
        or "pd 1" in s
        or "pdcd1" in s
        or "checkpoint" in s
        or "havcr2" in s
        or "lag3" in s
        or "tox" in s
    ):
        return "T", "Exhausted"

    if "nk" in s or "natural killer" in s or "nkg7" in s or "gnly" in s:
        if "t cell" not in s and "cd8" not in s:
            return "NK", "NK_killer"
        return "T", "NK_killer"

    if (
        "naive" in s
        or "central memory" in s
        or "tcm" in s
        or "ccr7" in s
        or "sell" in s
        or "tcf7" in s
        or "lef1" in s
        or "il7r" in s
    ):
        return "T", "Naive"

    if (
        "effector memory" in s
        or "tem" in s
        or "trm" in s
        or "resident memory" in s
        or "memory" in s
        or "gzmk" in s
        or "ltb" in s
        or "aqp3" in s
    ):
        return "T", "EffectorMemory"

    if (
        "effector" in s
        or "cytotoxic" in s
        or "killer" in s
        or "gzmb" in s
        or "prf1" in s
        or "cx3cr1" in s
        or "klrg1" in s
    ):
        return "T", "Effector"

    if "cd8" in s or "t cell" in s or "t lymphocyte" in s:
        return "T", "Unknown"

    return "Unknown", "Unknown"


def load_existing_parser(parser_script: Path | None):
    if parser_script is None or not parser_script.exists():
        return None

    spec = importlib.util.spec_from_file_location("existing_l4_l5_parser", parser_script)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "parse_prediction_major_state", None)
    if callable(fn):
        return fn

    return None


def parse_major_state(
    dataset: str,
    prediction_text: str,
    existing_parser=None,
) -> tuple[str, str, str]:
    if existing_parser is not None:
        try:
            maj, st = existing_parser(dataset, prediction_text)
            return str(maj), str(st), "existing_parser"
        except Exception:
            pass

    maj, st = fallback_parse_cd8_major_state(prediction_text)
    return maj, st, "fallback_cd8_parser"


def score_cd8_sanno(
    pred_major: str,
    pred_state: str,
    gt_major: str,
    gt_state: str,
    w_lineage: float = 0.7,
    w_state: float = 0.3,
) -> float:
    pred_major = str(pred_major)
    pred_state = str(pred_state)
    gt_major = str(gt_major)
    gt_state = str(gt_state)

    major_match = pred_major == gt_major

    if not major_match:
        return 0.0

    state_match = pred_state == gt_state
    return float(w_lineage * 1.0 + w_state * float(state_match))


def load_cd8_inputs(in_xlsx: Path, audit_sheet: str) -> pd.DataFrame:
    df = pd.read_excel(in_xlsx, sheet_name=audit_sheet)
    df = df.copy()

    if "Dataset" in df.columns:
        df = df[df["Dataset"].astype(str).str.strip().isin(["CD8 T", "CD8", "CD8+ T"])].copy()
    else:
        df = df[df["Cluster_ID"].astype(str).str.contains("CD8", case=False, na=False)].copy()
        df["Dataset"] = "CD8 T"

    if "Cluster_ID" not in df.columns:
        raise ValueError("Cluster_ID column is required.")

    required = ["Standard_Genes", "Curated_Genes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required gene-list columns: {missing}")

    if "GT_Major" not in df.columns:
        df["GT_Major"] = "T"
    if "GT_State" not in df.columns:
        raise ValueError("GT_State column is required for scoring. Use L2_per_cluster_audit as input.")

    if "Ground_Truth" not in df.columns:
        df["Ground_Truth"] = df.get("Ground_Truth_Label", "")

    df = df.dropna(subset=["Cluster_ID"]).copy()
    df["Cluster_ID"] = df["Cluster_ID"].astype(str)
    df = df[~df["Cluster_ID"].str.lower().str.contains("note", na=False)].copy()

    df = df.sort_values("Cluster_ID").reset_index(drop=True)
    return df


def completed_key_set(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    old = pd.read_csv(path)
    if old.empty:
        return set()
    out = set()
    for _, r in old.iterrows():
        out.add((str(r["Cluster_ID"]), str(r["Input_Type"]), int(r["Repeat"])))
    return out


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def pairwise_agreement(labels: list[str]) -> float:
    labels = [str(x) for x in labels if pd.notna(x)]
    n = len(labels)
    if n < 2:
        return np.nan
    total = 0
    same = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            same += int(labels[i] == labels[j])
    return same / total if total else np.nan


def build_l8_summary(l7: pd.DataFrame) -> pd.DataFrame:
    rows = []

    valid = l7.copy()
    valid["Is_Parse_Failure"] = valid["Parse_Status"].astype(str) != "parsed"
    valid["Ontology_Consistent"] = valid["Score_Sanno"].astype(float) >= 0.5
    valid["Exact_State"] = valid["Pred_State"].astype(str) == valid["GT_State"].astype(str)
    valid["Major_Lineage_Match"] = valid["Pred_Major"].astype(str) == valid["GT_Major"].astype(str)

    for input_type, sub in valid.groupby("Input_Type", sort=False):
        per_cluster = (
            sub.groupby("Cluster_ID")
            .agg(
                Mean_Sanno=("Score_Sanno", "mean"),
                SD_Sanno=("Score_Sanno", "std"),
                Modal_Label=("Parsed_CellType", lambda x: x.astype(str).mode().iloc[0] if len(x.astype(str).mode()) else ""),
                Pairwise_Label_Agreement=("Parsed_CellType", lambda x: pairwise_agreement(list(x))),
                All_Repeats_Same_Label=("Parsed_CellType", lambda x: int(len(set(map(str, x))) == 1)),
            )
            .reset_index()
        )

        rows.append(
            {
                "Summary_Row": input_type,
                "Backend": sub["Backend"].iloc[0],
                "Model_ID": sub["Model_ID"].iloc[0],
                "N_Clusters": sub["Cluster_ID"].nunique(),
                "N_Calls": len(sub),
                "N_Repeats_Per_Cluster_Input": int(sub["Repeat"].max()),
                "Parse_Failure_Rate": float(valid.loc[sub.index, "Is_Parse_Failure"].mean()),
                "Mean_Sanno_Across_Calls": float(sub["Score_Sanno"].astype(float).mean()),
                "Mean_Sanno_By_Cluster": float(per_cluster["Mean_Sanno"].mean()),
                "SD_Sanno_Across_Calls": float(sub["Score_Sanno"].astype(float).std(ddof=1)),
                "Mean_Within_Cluster_Pairwise_Label_Agreement": float(per_cluster["Pairwise_Label_Agreement"].mean()),
                "All_Repeats_Same_Label_Rate": float(per_cluster["All_Repeats_Same_Label"].mean()),
                "Exact_State_Agreement_Across_Calls": float(valid.loc[sub.index, "Exact_State"].mean()),
                "Major_Lineage_Accuracy_Across_Calls": float(valid.loc[sub.index, "Major_Lineage_Match"].mean()),
                "Ontology_Consistent_Accuracy_Sanno_ge_0.5": float(valid.loc[sub.index, "Ontology_Consistent"].mean()),
                "Low_Consistency_Rate_Sanno_lt_0.5": float((sub["Score_Sanno"].astype(float) < 0.5).mean()),
            }
        )

    per_cluster_input = (
        valid.groupby(["Cluster_ID", "Input_Type"])["Score_Sanno"]
        .mean()
        .unstack()
    )

    if {"Standard", "LLM-scCurator"}.issubset(per_cluster_input.columns):
        delta = per_cluster_input["LLM-scCurator"] - per_cluster_input["Standard"]
        rows.append(
            {
                "Summary_Row": "LLM-scCurator_minus_Standard",
                "Backend": valid["Backend"].iloc[0],
                "Model_ID": valid["Model_ID"].iloc[0],
                "N_Clusters": int(delta.dropna().shape[0]),
                "N_Calls": np.nan,
                "N_Repeats_Per_Cluster_Input": int(valid["Repeat"].max()),
                "Parse_Failure_Rate": np.nan,
                "Mean_Sanno_Across_Calls": np.nan,
                "Mean_Sanno_By_Cluster": np.nan,
                "SD_Sanno_Across_Calls": np.nan,
                "Mean_Within_Cluster_Pairwise_Label_Agreement": np.nan,
                "All_Repeats_Same_Label_Rate": np.nan,
                "Exact_State_Agreement_Across_Calls": np.nan,
                "Major_Lineage_Accuracy_Across_Calls": np.nan,
                "Ontology_Consistent_Accuracy_Sanno_ge_0.5": np.nan,
                "Low_Consistency_Rate_Sanno_lt_0.5": np.nan,
                "Mean_Delta_Sanno_By_Cluster": float(delta.mean()),
                "Median_Delta_Sanno_By_Cluster": float(delta.median()),
                "N_Clusters_Delta_Positive": int((delta > 0).sum()),
                "N_Clusters_Delta_Zero": int((delta == 0).sum()),
                "N_Clusters_Delta_Negative": int((delta < 0).sum()),
            }
        )

    return pd.DataFrame(rows)


def write_tables_to_excel(
    in_xlsx: Path,
    out_xlsx: Path,
    l6: pd.DataFrame,
    l7: pd.DataFrame,
    l8: pd.DataFrame,
) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    if not out_xlsx.exists():
        if in_xlsx.exists() and in_xlsx.resolve() != out_xlsx.resolve():
            shutil.copy2(in_xlsx, out_xlsx)
        else:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                pd.DataFrame({"created": [now_iso()]}).to_excel(writer, sheet_name="README", index=False)

    # Excel sheet names must be <=31 characters.
    # Full table names are preserved in CSV filenames and in the response/table captions.
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        l6.to_excel(writer, sheet_name="l6_llm_inference_metadata", index=False)
        l7.to_excel(writer, sheet_name="l7_repeated_run_robustness", index=False)
        l8.to_excel(writer, sheet_name="l8_local_backend_summary", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-xlsx", default="paper/revision/LetterTables.xlsx")
    parser.add_argument("--audit-sheet", default="L2_per_cluster_audit")
    parser.add_argument("--out-xlsx", default="paper/revision/LetterTables.xlsx")
    parser.add_argument("--outdir", default="paper/revision_tables")
    parser.add_argument("--parser-script", default="paper/revision_metrics/make_l4_l5_letter_tables.py")
    parser.add_argument("--host", default=os.getenv("LLMSC_OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.getenv("LLMSC_OLLAMA_MODEL", "llama3.1:8b"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLMSC_OLLAMA_TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LLMSC_OLLAMA_TIMEOUT", "600")))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--primary-version-note", default="Primary manuscript analyses fixed to PyPI llm-sc-curator==0.1.1")
    parser.add_argument("--revision-version-note", default="Revision-only local backend check using LLM-scCurator v0.1.2 / OllamaBackend")
    args = parser.parse_args()

    in_xlsx = Path(args.in_xlsx)
    out_xlsx = Path(args.out_xlsx)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_csv = outdir / "l7_repeated_run_robustness.csv"
    l6_csv = outdir / "l6_llm_inference_metadata.csv"
    l8_csv = outdir / "l8_local_open_weight_backend_summary.csv"

    if args.reset:
        for p in [raw_csv, l6_csv, l8_csv]:
            if p.exists():
                p.unlink()

    try:
        from llm_sc_curator.backends import OllamaBackend
    except Exception as e:
        raise ImportError(
            "Could not import OllamaBackend. Install the revision-only release first, e.g. "
            "python -m pip install 'git+https://github.com/kenflab/LLM-scCurator.git@v0.1.2'"
        ) from e

    try:
        import llm_sc_curator
        package_path = str(Path(llm_sc_curator.__file__).resolve())
    except Exception:
        package_path = ""

    try:
        llmsc_version = md.version("llm-sc-curator")
    except Exception:
        llmsc_version = "unknown"

    parser_path = Path(args.parser_script) if args.parser_script else None
    existing_parser = load_existing_parser(parser_path)

    df = load_cd8_inputs(in_xlsx, args.audit_sheet)
    print(f"Loaded CD8 input rows: {df.shape[0]}")
    if df.shape[0] != 17:
        print(f"WARNING: Expected 17 CD8 clusters, found {df.shape[0]}.")

    backend = OllamaBackend(
        host=args.host,
        model_name=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    started_at = now_iso()
    expected_calls = int(df.shape[0] * 2 * args.repeats)

    metadata = {
        "Analysis_ID": "CD8_local_ollama_repeated_backend_check",
        "Purpose": "Revision-only local open-weight backend reproducibility and portability check",
        "Dataset": "CD8 T",
        "N_Clusters": int(df.shape[0]),
        "Input_Types": "Standard;LLM-scCurator",
        "Repeats_Per_Input": int(args.repeats),
        "Expected_Calls": expected_calls,
        "Primary_Manuscript_Software_Note": args.primary_version_note,
        "Revision_Local_Backend_Software_Note": args.revision_version_note,
        "LLMscCurator_Version_Installed": llmsc_version,
        "LLMscCurator_Package_Path": package_path,
        "Backend_Class": "OllamaBackend",
        "Ollama_Host": args.host,
        "Model_ID": args.model,
        "Temperature": float(args.temperature),
        "Timeout_sec": float(args.timeout),
        "Prompt_Template_ID": PROMPT_TEMPLATE_ID,
        "Prompt_Template_Text": PROMPT_TEMPLATE,
        "Hosted_API_Used": "No",
        "Ollama_Version": run_cmd(["ollama", "--version"]),
        "Ollama_List": run_cmd(["ollama", "list"]),
        "Python": sys.version.replace("\n", " "),
        "Python_Executable": sys.executable,
        "Platform": platform.platform(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Hostname": socket.gethostname(),
        "Started_At": started_at,
        "Finished_At": "",
        "Parser_Source": str(parser_path) if existing_parser is not None else "fallback_cd8_parser",
        "Notes": "Marker lists were frozen from L2_per_cluster_audit; no marker recomputation was performed.",
    }

    completed = completed_key_set(raw_csv)
    new_calls = 0

    for _, row in df.iterrows():
        cluster_id = str(row["Cluster_ID"])
        gt_major = str(row.get("GT_Major", "T"))
        gt_state = str(row.get("GT_State", "Unknown"))
        ground_truth = str(row.get("Ground_Truth", row.get("Ground_Truth_Label", "")))

        input_specs = [
            ("Standard", row.get("Standard_Genes", "")),
            ("LLM-scCurator", row.get("Curated_Genes", "")),
        ]

        for input_type, gene_text in input_specs:
            genes = split_genes(gene_text)
            prompt = PROMPT_TEMPLATE.format(
                cluster_id=cluster_id,
                input_type=input_type,
                genes=", ".join(genes),
            )

            for repeat in range(1, args.repeats + 1):
                key = (cluster_id, input_type, repeat)
                if key in completed:
                    continue

                if args.max_calls is not None and new_calls >= args.max_calls:
                    break

                print(f"[{new_calls + 1}] {cluster_id} | {input_type} | repeat {repeat}")

                call_start = now_iso()
                t0 = time.time()

                try:
                    raw = backend.generate(prompt, json_mode=True)
                    failure_type = ""
                except Exception as e:
                    raw = json.dumps(
                        {
                            "cell_type": "Error",
                            "confidence": "Low",
                            "reasoning": repr(e),
                        }
                    )
                    failure_type = repr(e)

                duration = time.time() - t0
                parsed, parse_status, parse_error = parse_llm_json(raw)

                pred_major, pred_state, parser_used = parse_major_state(
                    "CD8 T",
                    parsed["cell_type"],
                    existing_parser=existing_parser,
                )

                score = score_cd8_sanno(
                    pred_major=pred_major,
                    pred_state=pred_state,
                    gt_major=gt_major,
                    gt_state=gt_state,
                )

                out_row = {
                    "Dataset": "CD8 T",
                    "Cluster_ID": cluster_id,
                    "Ground_Truth": ground_truth,
                    "GT_Major": gt_major,
                    "GT_State": gt_state,
                    "Input_Type": input_type,
                    "Repeat": repeat,
                    "N_Genes": len(genes),
                    "Genes": ";".join(genes),
                    "Prompt_Template_ID": PROMPT_TEMPLATE_ID,
                    "Prompt": prompt,
                    "Backend": "OllamaBackend",
                    "Model_ID": args.model,
                    "Host": args.host,
                    "Temperature": args.temperature,
                    "Timeout_sec": args.timeout,
                    "Raw_Output": raw,
                    "Parsed_CellType": parsed["cell_type"],
                    "Parsed_Confidence": parsed["confidence"],
                    "Parsed_Reasoning": parsed["reasoning"],
                    "Parse_Status": parse_status,
                    "Parse_Error": parse_error,
                    "Failure_Type": failure_type,
                    "Pred_Major": pred_major,
                    "Pred_State": pred_state,
                    "Parser_Used": parser_used,
                    "Score_Sanno": score,
                    "Exact_State_Agreement": pred_state == gt_state,
                    "Major_Lineage_Match": pred_major == gt_major,
                    "Ontology_Consistent_Sanno_ge_0.5": score >= 0.5,
                    "Low_Consistency_Sanno_lt_0.5": score < 0.5,
                    "Call_Started_At": call_start,
                    "Call_Finished_At": now_iso(),
                    "Duration_sec": round(duration, 3),
                }

                append_csv_row(raw_csv, out_row)
                new_calls += 1
                time.sleep(args.sleep_sec)

            if args.max_calls is not None and new_calls >= args.max_calls:
                break

        if args.max_calls is not None and new_calls >= args.max_calls:
            break

    if raw_csv.exists():
        l7 = pd.read_csv(raw_csv)
    else:
        l7 = pd.DataFrame()

    metadata["Finished_At"] = now_iso()
    metadata["Completed_Calls"] = int(l7.shape[0])
    metadata["Completed_Expected_Call_Fraction"] = float(l7.shape[0] / expected_calls) if expected_calls else np.nan
    l6 = pd.DataFrame([metadata])

    if l7.empty:
        l8 = pd.DataFrame()
    else:
        l8 = build_l8_summary(l7)

    l6.to_csv(l6_csv, index=False)
    l8.to_csv(l8_csv, index=False)

    write_tables_to_excel(
        in_xlsx=in_xlsx,
        out_xlsx=out_xlsx,
        l6=l6,
        l7=l7,
        l8=l8,
    )

    print("\nWrote:")
    print(f"  {l6_csv}")
    print(f"  {raw_csv}")
    print(f"  {l8_csv}")
    print(f"  {out_xlsx}")
    print("\nSheets:")
    print("  L6_llm_inference_metadata")
    print("  L7_repeated_run_robustness")
    print("  L8_local_backend_summary")

    if not l8.empty:
        print("\nL6 preview:")
        print(l8.to_string(index=False))


if __name__ == "__main__":
    main()
