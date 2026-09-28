#!/usr/bin/env python3
"""Build the low-cost Reviewer Figure L5 analysis for Reviewer 1, comment 8.

This reviewer-only analysis deliberately separates two questions.

Workflow-level descriptive reference
------------------------------------
1. Existing Gemini Standard workflow (reused; no new inference)
2. Existing integrated LLM-scCurator Full pipeline (reused; no new inference)
3. Native GPTCelltype with ranked Standard DEG top 10 (one API call by default)

These three conditions are *not* a matched-backend superiority experiment.
They differ in backend, prompt, context, and input construction. The script
therefore reports their observed CD8 differences descriptively, without a
cross-workflow P value, and writes explicit interpretation flags to every
relevant output.

Same-backend prompt/heuristic controls
--------------------------------------
4. Manual o3 + GPTCelltype-style basic prompt + Standard DEG top 10
   (reused from Reviewer Figure L4)
5. Manual o3 + the GPTCelltype-paper CD3/chain-of-thought sentence +
   Standard DEG top 10
6. Manual o3 + the same CD3 sentence + regex_mask top 10

Conditions 5 and 6 require manual copy/paste only; they never call an API.
Five independent fresh chats are prepared by default. Comparing condition 6
with condition 5 isolates the contribution of the prespecified regex masking
under the same model, prompt, gene budget, and repeats. The paper's CD3
sentence is lineage-oriented and is therefore described as a paper-defined
CoT-inspired prompt, rather than assumed to be stronger for CD8 substates.

Cost control
------------
The script never calls an API unless ``--run-api`` is supplied. Native
GPTCelltype receives one condition batched across all 17 clusters, so the
default requires exactly one API request. Successful requests are checkpointed
and reused. ``--force-api`` is required to discard a complete checkpoint.

Typical first run (preparation only; zero API calls):

  python paper/revision_metrics/make_gptcelltype_shared_annotator_check.py \
    --in-xlsx LetterTables.xlsx \
    --l4-manual-predictions \
      paper/revision_tables/l4_manual_o3_top10_final/L4_manual_o3_predictions.csv \
    --outdir paper/revision_tables/l5_reviewer1_low_cost

After inspecting the native input, make the single GPTCelltype request:

  python paper/revision_metrics/make_gptcelltype_shared_annotator_check.py \
    --in-xlsx LetterTables.xlsx \
    --l4-manual-predictions \
      paper/revision_tables/l4_manual_o3_top10_final/L4_manual_o3_predictions.csv \
    --outdir paper/revision_tables/l5_reviewer1_low_cost \
    --run-api

Then complete the two manual conditions in ``L5_manual_predictions.csv`` and
rerun the same command without ``--run-api``. No new API call is made because
the native GPTCelltype checkpoint is reused.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import itertools
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


ANALYSIS_ID = "ReviewerFig_L5_low_cost_GPTCelltype_prompt_controls_v1"

SCRIPT_BUILD = "l5_low_cost_20260724_public_parser_20260928"

API_CONDITION = "GPTCelltype_native_top10"

MANUAL_CONDITIONS = [
    "Manual_o3_Standard_CoT_top10",
    "Manual_o3_RegexMask_CoT_top10",
]
REUSED_L4_CONDITION = "Manual_o3_Standard_basic_top10"

# All workflow-level conditions retained for scoring, tables, metadata,
# and reproduction checks.
WORKFLOW_CONDITIONS = [
    "Gemini_Standard_existing",
    "LLM_scCurator_Full_pipeline_existing",
    API_CONDITION,
]

# Only these two conditions are displayed in figure panels A and B.
# Gemini Standard remains in the analysis outputs but is omitted from the plot.
WORKFLOW_PLOT_CONDITIONS = [
    "LLM_scCurator_Full_pipeline_existing",
    API_CONDITION,
]

PROMPT_CONTROL_CONDITIONS = [
    REUSED_L4_CONDITION,
    *MANUAL_CONDITIONS,
]

CONDITION_ORDER = WORKFLOW_CONDITIONS + PROMPT_CONTROL_CONDITIONS

CONDITION_LABELS = {
    "Gemini_Standard_existing": "Gemini Standard\n(existing)",
    "LLM_scCurator_Full_pipeline_existing": "LLM-scCurator Full\n(existing)",
    API_CONDITION: "Native GPTCelltype\nStandard top 10",
    REUSED_L4_CONDITION: "o3 basic\nStandard top 10",
    "Manual_o3_Standard_CoT_top10": "o3 paper CoT\nStandard top 10",
    "Manual_o3_RegexMask_CoT_top10": "o3 paper CoT\nregex_mask top 10",
}

CONDITION_COLORS = {
    # Gemini is retained in tables/checks but not shown in panels A and B.
    "Gemini_Standard_existing": "#7A8CA5",

    # Panels A and B
    "LLM_scCurator_Full_pipeline_existing": "#D5524A",
    API_CONDITION: "#3D8ED0",

    # Panels C and D
    # Same blue as Native GPTCelltype because this is the corresponding
    # basic-prompt Standard top-10 condition.
    REUSED_L4_CONDITION: "#3D8ED0",

    # Green for the paper CoT prompt with unmodified Standard top-10 markers.
    "Manual_o3_Standard_CoT_top10": "#59A14F",

    # Yellow/gold for paper CoT plus regex masking.
    "Manual_o3_RegexMask_CoT_top10": "#E3B341",
}

PAPER_COT_SENTENCE = (
    "Because CD3 gene is a marker gene of T cells, if CD3 gene is included "
    "in the marker gene list of an unknown cell type, the cell type is likely "
    "to be T cells, a subtype of T cells, or a mixed cell type containing T cells"
)

# The helper intentionally calls GPTCelltype::gptcelltype. It does not
# reimplement the package through a different OpenAI client.
R_RUNNER = r'''args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 10) {
  stop(paste(
    "Expected 10 arguments: input_csv output_csv metadata_csv prompt_csv",
    "model tissue repeats inter_call_sleep_sec max_api_retries retry_base_wait_sec"
  ))
}

input_csv <- args[[1]]
output_csv <- args[[2]]
metadata_csv <- args[[3]]
prompt_csv <- args[[4]]
model_id <- args[[5]]
tissue_name <- args[[6]]
repeats <- as.integer(args[[7]])
inter_call_sleep_sec <- as.numeric(args[[8]])
max_api_retries <- as.integer(args[[9]])
retry_base_wait_sec <- as.numeric(args[[10]])

if (Sys.getenv("OPENAI_API_KEY") == "") {
  stop("OPENAI_API_KEY is not set.")
}
if (!requireNamespace("GPTCelltype", quietly = TRUE)) {
  stop("R package GPTCelltype is not installed.")
}
if (!requireNamespace("openai", quietly = TRUE)) {
  stop("R package openai is not installed.")
}

d <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("Condition", "Cluster_ID", "Gene", "Rank")
if (!all(required %in% colnames(d))) {
  stop(paste("Missing input columns:", paste(setdiff(required, colnames(d)), collapse = ", ")))
}

conditions <- unique(d$Condition)
if (length(conditions) != 1L) {
  stop("The low-cost L5 native GPTCelltype input must contain exactly one condition.")
}
cluster_order <- unique(d$Cluster_ID)

empty_results <- data.frame(
  Condition = character(),
  Repeat = integer(),
  Cluster_ID = character(),
  Raw_Prediction = character(),
  Model_ID = character(),
  Tissue_Name = character(),
  stringsAsFactors = FALSE
)

if (file.exists(output_csv)) {
  results <- read.csv(output_csv, stringsAsFactors = FALSE, check.names = FALSE)
  missing_result_cols <- setdiff(colnames(empty_results), colnames(results))
  if (length(missing_result_cols) > 0) {
    stop(paste("Existing prediction file is missing columns:", paste(missing_result_cols, collapse = ", ")))
  }
  valid_model <- !is.na(results$Model_ID) & nzchar(as.character(results$Model_ID))
  existing_models <- unique(as.character(results$Model_ID[valid_model]))
  if (length(existing_models) > 0 && !all(existing_models == model_id)) {
    stop(paste(
      "Existing prediction file uses a different model:",
      paste(existing_models, collapse = ";")
    ))
  }
} else {
  results <- empty_results
}

write_checkpoint <- function(x, path) {
  tmp <- paste0(path, ".tmp")
  write.csv(x, tmp, row.names = FALSE, na = "")
  if (!file.copy(tmp, path, overwrite = TRUE)) {
    stop(paste("Could not write checkpoint:", path))
  }
  unlink(tmp)
}

is_complete <- function(x, condition, repeat_id, present_clusters) {
  sub <- x[x$Condition == condition & x$Repeat == repeat_id, , drop = FALSE]
  nrow(sub) == length(present_clusters) &&
    setequal(as.character(sub$Cluster_ID), as.character(present_clusters)) &&
    all(!is.na(sub$Raw_Prediction) & nzchar(as.character(sub$Raw_Prediction)))
}

rate_limit_wait <- function(error_message, attempt) {
  lower <- tolower(error_message)
  matched <- regexec("try again in ([0-9.]+)s", lower, perl = TRUE)
  pieces <- regmatches(lower, matched)[[1]]
  server_wait <- if (length(pieces) >= 2) suppressWarnings(as.numeric(pieces[[2]])) else NA_real_
  exponential_wait <- retry_base_wait_sec * (2 ^ (attempt - 1L))
  wait_sec <- exponential_wait
  if (is.finite(server_wait)) {
    wait_sec <- max(wait_sec, server_wait + 2)
  }
  min(wait_sec, 120)
}

call_gptcelltype_with_retry <- function(gene_lists, condition, repeat_id) {
  for (attempt in seq_len(max_api_retries + 1L)) {
    attempt_result <- tryCatch(
      list(
        ok = TRUE,
        value = GPTCelltype::gptcelltype(
          input = gene_lists,
          tissuename = tissue_name,
          model = model_id
        )
      ),
      error = function(e) list(ok = FALSE, message = conditionMessage(e))
    )
    if (isTRUE(attempt_result$ok)) {
      return(attempt_result$value)
    }
    error_message <- as.character(attempt_result$message)
    is_rate_limit <- grepl("429|rate limit|tokens per min|tpm", error_message, ignore.case = TRUE)
    if (!is_rate_limit || attempt > max_api_retries) {
      stop(error_message)
    }
    wait_sec <- rate_limit_wait(error_message, attempt)
    message(sprintf(
      "Rate limit for %s repeat %d; waiting %.1f s before retry %d/%d.",
      condition, repeat_id, wait_sec, attempt, max_api_retries
    ))
    Sys.sleep(wait_sec)
  }
  stop("Retry loop ended unexpectedly.")
}

write_r_metadata <- function(status, calls_made_this_run) {
  completed_keys <- unique(paste(results$Condition, results$Repeat, sep = "::"))
  completed_keys <- completed_keys[nzchar(completed_keys)]
  metadata <- data.frame(
    Key = c(
      "R_version", "GPTCelltype_version", "openai_R_version", "Model_ID",
      "Tissue_Name", "Repeats", "Conditions", "API_calls_expected",
      "API_calls_completed", "API_calls_made_this_run", "Inter_call_sleep_sec",
      "Max_API_retries", "Retry_base_wait_sec", "Run_status"
    ),
    Value = c(
      R.version.string,
      as.character(utils::packageVersion("GPTCelltype")),
      as.character(utils::packageVersion("openai")),
      model_id,
      tissue_name,
      as.character(repeats),
      paste(conditions, collapse = ";"),
      as.character(length(conditions) * repeats),
      as.character(length(completed_keys)),
      as.character(calls_made_this_run),
      as.character(inter_call_sleep_sec),
      as.character(max_api_retries),
      as.character(retry_base_wait_sec),
      status
    ),
    stringsAsFactors = FALSE
  )
  write.csv(metadata, metadata_csv, row.names = FALSE, na = "")
}

prompt_rows <- list()
pidx <- 1L
calls_made_this_run <- 0L
write_r_metadata("started_or_resumed", calls_made_this_run)

for (condition in conditions) {
  sub <- d[d$Condition == condition, , drop = FALSE]
  sub <- sub[order(match(sub$Cluster_ID, cluster_order), sub$Rank), , drop = FALSE]
  present_clusters <- unique(sub$Cluster_ID)
  gene_lists <- lapply(present_clusters, function(cid) sub$Gene[sub$Cluster_ID == cid])
  names(gene_lists) <- present_clusters

  collapsed <- vapply(gene_lists, paste, collapse = ",", FUN.VALUE = character(1))
  native_prompt <- paste0(
    "Identify cell types of ", tissue_name,
    " cells using the following markers separately for each\n row. ",
    "Only provide the cell type name. Do not show numbers before the name.\n ",
    "Some can be a mixture of multiple cell types.\n",
    paste(unname(collapsed), collapse = "\n")
  )
  prompt_rows[[pidx]] <- data.frame(
    Condition = condition,
    Model_ID = model_id,
    Tissue_Name = tissue_name,
    Prompt = native_prompt,
    stringsAsFactors = FALSE
  )
  pidx <- pidx + 1L

  for (repeat_id in seq_len(repeats)) {
    if (is_complete(results, condition, repeat_id, present_clusters)) {
      message(sprintf("Checkpoint found; skipping condition=%s repeat=%d", condition, repeat_id))
      next
    }
    if (calls_made_this_run > 0L && inter_call_sleep_sec > 0) {
      message(sprintf("Waiting %.1f s between API calls.", inter_call_sleep_sec))
      Sys.sleep(inter_call_sleep_sec)
    }
    message(sprintf("GPTCelltype call: condition=%s repeat=%d", condition, repeat_id))
    pred <- as.character(call_gptcelltype_with_retry(gene_lists, condition, repeat_id))
    if (length(pred) != length(present_clusters)) {
      stop(sprintf(
        "GPTCelltype returned %d labels for %d clusters in %s repeat %d.",
        length(pred), length(present_clusters), condition, repeat_id
      ))
    }
    new_rows <- data.frame(
      Condition = condition,
      Repeat = repeat_id,
      Cluster_ID = present_clusters,
      Raw_Prediction = unname(pred),
      Model_ID = model_id,
      Tissue_Name = tissue_name,
      stringsAsFactors = FALSE
    )
    results <- results[!(results$Condition == condition & results$Repeat == repeat_id), , drop = FALSE]
    results <- rbind(results, new_rows)
    write_checkpoint(results, output_csv)
    calls_made_this_run <- calls_made_this_run + 1L
    write_r_metadata("checkpoint_written", calls_made_this_run)
  }
}

write.csv(do.call(rbind, prompt_rows), prompt_csv, row.names = FALSE, na = "")
write_r_metadata("completed", calls_made_this_run)
'''


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv_with_encoding_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "cp932",
        "shift_jis",
        "mac_roman",
        "latin-1",
    ):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except (
            UnicodeDecodeError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(
        f"Could not decode CSV {path}. Attempts: {' | '.join(errors)}"
    )


def load_cd8_audit(path: Path, sheet: str) -> pd.DataFrame:
    audit = pd.read_excel(path, sheet_name=sheet)
    if "Cluster_ID" not in audit.columns:
        raise ValueError(f"{sheet} does not contain Cluster_ID.")
    if "Dataset" in audit.columns:
        keep = (
            audit["Dataset"]
            .astype(str)
            .str.strip()
            .isin(["CD8 T", "CD8", "CD8+ T"])
        )
    else:
        keep = audit["Cluster_ID"].astype(str).str.contains(
            "CD8", case=False, na=False
        )
    audit = audit.loc[keep].copy()
    required = ["Cluster_ID", "Ground_Truth"]
    missing = [column for column in required if column not in audit.columns]
    if missing:
        raise ValueError(f"Missing audit columns: {missing}")
    audit = audit.dropna(subset=["Cluster_ID"]).copy()
    audit["Cluster_ID"] = audit["Cluster_ID"].astype(str).str.strip()
    audit = audit[
        ~audit["Cluster_ID"].str.lower().str.contains("note", na=False)
    ].copy()
    audit = audit.sort_values("Cluster_ID").reset_index(drop=True)
    if audit["Cluster_ID"].duplicated().any():
        duplicates = audit.loc[
            audit["Cluster_ID"].duplicated(), "Cluster_ID"
        ].tolist()
        raise ValueError(f"Duplicate CD8 cluster IDs: {duplicates}")
    if len(audit) != 17:
        raise ValueError(f"Expected 17 CD8 clusters, found {len(audit)}.")
    return audit


def _normalized_dataset(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def load_ranked_marker_lists(
    path: Path,
    sheet: str,
    *,
    dataset: str,
    variants: Sequence[str],
    cluster_order: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    markers = pd.read_excel(path, sheet_name=sheet)
    required = ["dataset", "cluster", "variant", "rank", "gene"]
    missing = [column for column in required if column not in markers.columns]
    if missing:
        raise ValueError(f"{sheet} is missing ranked-marker columns: {missing}")

    wanted_dataset = _normalized_dataset(dataset)
    markers = markers[
        markers["dataset"].map(_normalized_dataset).eq(wanted_dataset)
    ].copy()
    markers["variant"] = markers["variant"].astype(str).str.strip().str.lower()
    markers["cluster"] = markers["cluster"].astype(str).str.strip()
    markers["rank"] = pd.to_numeric(markers["rank"], errors="raise")
    markers["gene"] = markers["gene"].fillna("").astype(str).str.strip()
    markers = markers[markers["gene"] != ""].copy()

    output: dict[str, dict[str, list[str]]] = {}
    for requested_variant in variants:
        variant = requested_variant.strip().lower()
        selected = markers[markers["variant"].eq(variant)].sort_values(
            ["cluster", "rank"]
        )
        by_cluster: dict[str, list[str]] = {}
        for cluster, subset in selected.groupby("cluster", sort=False):
            genes: list[str] = []
            seen: set[str] = set()
            for gene in subset["gene"].astype(str):
                key = gene.upper()
                if key not in seen:
                    genes.append(gene)
                    seen.add(key)
            by_cluster[str(cluster)] = genes
        missing_clusters = sorted(set(cluster_order) - set(by_cluster))
        if missing_clusters:
            raise ValueError(
                f"Variant {requested_variant!r} is missing clusters: "
                f"{missing_clusters}"
            )
        output[requested_variant] = {
            cluster: by_cluster[cluster] for cluster in cluster_order
        }
    return output


def build_marker_input(
    audit: pd.DataFrame,
    lists_by_condition: dict[str, dict[str, list[str]]],
    max_genes: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, marker_lists in lists_by_condition.items():
        for _, audit_row in audit.iterrows():
            cluster = str(audit_row["Cluster_ID"])
            genes = marker_lists[cluster][:max_genes]
            if len(genes) != max_genes:
                raise ValueError(
                    f"{condition} has {len(genes)} genes for {cluster}; "
                    f"{max_genes} are required for the prespecified top-N design."
                )
            for rank, gene in enumerate(genes, start=1):
                rows.append(
                    {
                        "Condition": condition,
                        "Cluster_ID": cluster,
                        "Ground_Truth": audit_row["Ground_Truth"],
                        "Gene": gene,
                        "Rank": rank,
                        "N_Genes": len(genes),
                    }
                )
    return pd.DataFrame(rows)


def _prompt_gene_rows(
    marker_input: pd.DataFrame,
    condition: str,
) -> list[str]:
    subset = marker_input[marker_input["Condition"].eq(condition)]
    rows: list[str] = []
    for _, cluster_rows in subset.groupby("Cluster_ID", sort=False):
        rows.append(
            ",".join(
                cluster_rows.sort_values("Rank")["Gene"].astype(str)
            )
        )
    if not rows:
        raise ValueError(f"No marker rows found for condition {condition}.")
    return rows


def native_gptcelltype_prompt(
    marker_input: pd.DataFrame,
    condition: str,
    tissue: str,
) -> str:
    rows = _prompt_gene_rows(marker_input, condition)
    return (
        f"Identify cell types of {tissue} cells using the following markers "
        "separately for each\n row. Only provide the cell type name. Do not "
        "show numbers before the name.\n Some can be a mixture of multiple "
        "cell types.\n"
        + "\n".join(rows)
    )


def manual_basic_prompt(
    marker_input: pd.DataFrame,
    condition: str,
    tissue: str,
) -> str:
    """Match the formatting-guarded basic prompt used for Reviewer Figure L4."""
    rows = _prompt_gene_rows(marker_input, condition)
    return (
        f"Identify cell types of {tissue} cells using the following markers "
        "separately for each row. Only provide the cell type name. Do not show "
        "numbers before the name. Some can be a mixture of multiple cell types.\n"
        f"Return exactly {len(rows)} non-empty lines, one cell-type annotation "
        "per input row, in the same order, with no header, bullets, numbering, "
        "or explanation.\n"
        + "\n".join(rows)
    )


def manual_cot_prompt(
    marker_input: pd.DataFrame,
    condition: str,
    tissue: str,
) -> str:
    return PAPER_COT_SENTENCE + ".\n" + manual_basic_prompt(
        marker_input, condition, tissue
    )


def write_prompt_manifests(
    api_path: Path,
    manual_path: Path,
    api_input: pd.DataFrame,
    manual_input: pd.DataFrame,
    *,
    api_model: str,
    manual_model: str,
    tissue: str,
    manual_repeats: int,
) -> None:
    api_prompt = native_gptcelltype_prompt(
        api_input, API_CONDITION, tissue
    )
    pd.DataFrame(
        [
            {
                "Condition": API_CONDITION,
                "Model_ID": api_model,
                "Tissue_Name": tissue,
                "Prompt_Strategy": "Native GPTCelltype basic prompt",
                "Prompt_SHA256": sha256_text(api_prompt),
                "Prompt": api_prompt,
            }
        ]
    ).to_csv(api_path, index=False)

    manual_rows: list[dict[str, Any]] = []
    for condition in MANUAL_CONDITIONS:
        prompt = manual_cot_prompt(manual_input, condition, tissue)
        for repeat in range(1, manual_repeats + 1):
            manual_rows.append(
                {
                    "Run_ID": f"{condition}__r{repeat:02d}",
                    "Condition": condition,
                    "Repeat": repeat,
                    "Model_Label": manual_model,
                    "Tissue_Name": tissue,
                    "Fresh_Chat_Required": True,
                    "Prompt_Strategy": (
                        "GPTCelltype-paper CD3/CoT sentence plus basic prompt "
                        "and deterministic line-count formatting guard"
                    ),
                    "Prompt_SHA256": sha256_text(prompt),
                    "Prompt": prompt,
                }
            )
    pd.DataFrame(manual_rows).to_csv(
        manual_path, index=False, encoding="utf-8-sig"
    )


def build_manual_prediction_template(
    audit: pd.DataFrame,
    *,
    model: str,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clusters = audit["Cluster_ID"].astype(str).tolist()
    for condition in MANUAL_CONDITIONS:
        for repeat in range(1, repeats + 1):
            run_id = f"{condition}__r{repeat:02d}"
            for row_index, cluster in enumerate(clusters, start=1):
                rows.append(
                    {
                        "Run_ID": run_id,
                        "Condition": condition,
                        "Repeat": repeat,
                        "Row_Index": row_index,
                        "Cluster_ID": cluster,
                        "Raw_Prediction": "",
                        "Model_ID": model,
                        "Run_Date": "",
                        "Fresh_Chat": "",
                        "Notes": "",
                    }
                )
    return pd.DataFrame(rows)


def _available_backup_path(path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(
        f"{path.stem}.pre_update_{timestamp}{path.suffix}"
    )
    counter = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.pre_update_{timestamp}_{counter}{path.suffix}"
        )
        counter += 1
    return candidate


def update_manual_prediction_template(
    path: Path,
    audit: pd.DataFrame,
    *,
    model: str,
    repeats: int,
) -> dict[str, Any]:
    expected = build_manual_prediction_template(
        audit, model=model, repeats=repeats
    )
    if not path.exists():
        expected.to_csv(path, index=False, encoding="utf-8-sig")
        return {
            "created": True,
            "encoding": "new_utf-8-sig",
            "preserved_nonempty": 0,
            "added_rows": len(expected),
            "backup": "",
        }

    existing, encoding = read_csv_with_encoding_fallback(path)
    key_columns = ["Condition", "Repeat", "Cluster_ID"]
    missing = [column for column in key_columns if column not in existing.columns]
    if missing:
        raise ValueError(
            f"Existing manual template is missing key columns: {missing}"
        )
    existing = existing.copy()
    existing["Condition"] = existing["Condition"].astype(str).str.strip()
    existing["Cluster_ID"] = existing["Cluster_ID"].astype(str).str.strip()
    existing["Repeat"] = pd.to_numeric(
        existing["Repeat"], errors="raise"
    ).astype(int)
    if existing.duplicated(key_columns).any():
        bad = existing.loc[
            existing.duplicated(key_columns, keep=False), key_columns
        ].head(10)
        raise ValueError(
            "Duplicate rows in existing manual template:\n"
            f"{bad.to_string(index=False)}"
        )

    expected_keys = set(map(tuple, expected[key_columns].to_numpy()))
    existing_keys = set(map(tuple, existing[key_columns].to_numpy()))
    unexpected = sorted(existing_keys - expected_keys)
    if unexpected:
        raise ValueError(
            "Existing manual template contains rows outside this two-condition "
            f"L5 design. Examples: {unexpected[:10]}"
        )

    expected_indexed = expected.set_index(key_columns)
    existing_indexed = existing.set_index(key_columns)
    structural = {"Run_ID", "Row_Index"}
    for column in existing_indexed.columns:
        if column in structural:
            continue
        if column not in expected_indexed.columns:
            expected_indexed[column] = pd.NA
        expected_indexed.loc[existing_indexed.index, column] = (
            existing_indexed[column].to_numpy()
        )
    merged = expected_indexed.reset_index()
    base_columns = list(expected.columns)
    extra_columns = [
        column for column in merged.columns if column not in base_columns
    ]
    merged = merged[base_columns + extra_columns]
    preserved_nonempty = int(
        existing.get("Raw_Prediction", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    added_rows = len(expected_keys - existing_keys)
    needs_rewrite = added_rows > 0 or encoding not in {"utf-8", "utf-8-sig"}
    backup = ""
    if needs_rewrite:
        backup_path = _available_backup_path(path)
        shutil.copy2(path, backup_path)
        merged.to_csv(path, index=False, encoding="utf-8-sig")
        backup = str(backup_path)
    return {
        "created": False,
        "encoding": encoding,
        "preserved_nonempty": preserved_nonempty,
        "added_rows": added_rows,
        "backup": backup,
    }


def write_manual_protocol(
    path: Path,
    *,
    prompt_path: Path,
    prediction_path: Path,
    l4_path: Path,
) -> None:
    text = f"""Manual o3 protocol for Reviewer Figure L5

Purpose
-------
Test the Reviewer 1 comment-8 control "simple recurrent-noise removal plus a
stronger prompt" without new API cost. The prompt content uses the exact CD3
sentence reported for GPTCelltype's chain-of-thought strategy. Because that
sentence primarily reinforces T-cell lineage and all 17 clusters are CD8
T-cell clusters, the analysis calls it a paper-defined CoT-inspired prompt; it
does not assume that it is stronger for CD8 state resolution.

Procedure
---------
1. Open {prompt_path.name}.
2. Complete every Run_ID in a separate new/temporary ChatGPT conversation.
3. Paste only Prompt. Do not add the condition name, cluster IDs, ground truth,
   earlier answers, or other biological context.
4. Keep browsing, tools, and memory/context carry-over off when possible. Do
   not revise a response because it looks biologically unexpected.
5. Copy exactly one returned label per row into Raw_Prediction for the matching
   Run_ID and Row_Index in {prediction_path.name}.
6. Record the displayed model in Model_ID, date/time in Run_Date, and TRUE in
   Fresh_Chat. Record technical retries in Notes and repeat the whole Run_ID in
   a new chat.
7. Save the CSV and rerun the Python script. No API flag is needed for scoring.

Reused basic-prompt comparator
------------------------------
The Standard top-10 basic-prompt predictions are imported from:
{l4_path}
They must contain 17 clusters x the requested repeats for Condition=Standard_DEG.
The script also checks the neighboring L4 prompt manifest against the newly
reconstructed Standard top-10 basic prompt unless
--allow-l4-prompt-mismatch is explicitly supplied.

Interpretation
--------------
Manual_o3_RegexMask_CoT_top10 minus Manual_o3_Standard_CoT_top10 is the
prespecified simple-filter effect under matched model, prompt, gene budget, and
repeats. Manual_o3_Standard_CoT_top10 minus the reused L4 basic condition is a
prompt sensitivity contrast. These are separate from the workflow-level
existing Gemini versus native GPTCelltype descriptive panel.
"""
    path.write_text(text, encoding="utf-8")


def _truthy(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "t", "yes", "y", "1"})
    )


def validate_prediction_design(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    conditions: Sequence[str],
    repeats: int,
    model: str | None,
    require_fresh_chat: bool,
    label: str,
) -> pd.DataFrame:
    required = ["Condition", "Repeat", "Cluster_ID", "Raw_Prediction"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    predictions = predictions.copy()
    predictions["Condition"] = predictions["Condition"].astype(str).str.strip()
    predictions["Cluster_ID"] = predictions["Cluster_ID"].astype(str).str.strip()
    predictions["Repeat"] = pd.to_numeric(
        predictions["Repeat"], errors="raise"
    ).astype(int)
    predictions["Raw_Prediction"] = (
        predictions["Raw_Prediction"].fillna("").astype(str).str.strip()
    )
    keys = ["Condition", "Repeat", "Cluster_ID"]
    if predictions.duplicated(keys).any():
        bad = predictions.loc[
            predictions.duplicated(keys, keep=False), keys
        ].head(10)
        raise ValueError(
            f"{label} has duplicate keys:\n{bad.to_string(index=False)}"
        )
    expected = set(
        itertools.product(
            conditions,
            range(1, repeats + 1),
            audit["Cluster_ID"].astype(str).tolist(),
        )
    )
    observed = set(
        zip(
            predictions["Condition"],
            predictions["Repeat"],
            predictions["Cluster_ID"],
        )
    )
    if expected != observed or len(predictions) != len(expected):
        raise ValueError(
            f"{label} does not match the expected design. "
            f"Missing examples={sorted(expected - observed)[:10]}; "
            f"extra examples={sorted(observed - expected)[:10]}"
        )
    if predictions["Raw_Prediction"].eq("").any():
        bad = predictions.loc[
            predictions["Raw_Prediction"].eq(""), keys
        ].head(10)
        raise ValueError(
            f"{label} has empty predictions (first examples):\n"
            f"{bad.to_string(index=False)}"
        )
    if model is not None and "Model_ID" in predictions.columns:
        models = set(
            predictions["Model_ID"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
        )
        if models and models != {model}:
            raise ValueError(
                f"{label} Model_ID values {models} do not match {model!r}."
            )
    if require_fresh_chat:
        if "Fresh_Chat" not in predictions.columns:
            raise ValueError(f"{label} is missing Fresh_Chat.")
        fresh = _truthy(predictions["Fresh_Chat"])
        if not fresh.all():
            bad = predictions.loc[~fresh, keys].head(10)
            raise ValueError(
                f"{label} must have Fresh_Chat=TRUE for every row. Examples:\n"
                f"{bad.to_string(index=False)}"
            )
    return predictions


def load_reused_l4_basic_predictions(
    path: Path,
    audit: pd.DataFrame,
    *,
    marker_input: pd.DataFrame,
    tissue: str,
    model: str,
    repeats: int,
    allow_prompt_mismatch: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    l4, encoding = read_csv_with_encoding_fallback(path)
    l4["Condition"] = l4["Condition"].astype(str).str.strip()
    l4 = l4[l4["Condition"].eq("Standard_DEG")].copy()
    l4 = validate_prediction_design(
        l4,
        audit,
        conditions=["Standard_DEG"],
        repeats=repeats,
        model=model,
        require_fresh_chat=True,
        label="Reused L4 Standard basic predictions",
    )
    l4["Condition"] = REUSED_L4_CONDITION
    l4["Source_File"] = str(path.resolve())
    l4["Reuse_Status"] = "Reused without new inference"

    prompt_manifest = path.parent / "L4_shared_annotator_prompt_manifest.csv"
    prompt_check = "manifest_not_found"
    expected_prompt = manual_basic_prompt(
        marker_input, "Manual_o3_Standard_CoT_top10", tissue
    )
    expected_sha = sha256_text(expected_prompt)
    observed_shas: set[str] = set()
    if prompt_manifest.exists():
        manifest, _ = read_csv_with_encoding_fallback(prompt_manifest)
        if "Condition" not in manifest.columns:
            raise ValueError(
                f"L4 prompt manifest lacks Condition: {prompt_manifest}"
            )
        standard = manifest[
            manifest["Condition"].astype(str).str.strip().eq("Standard_DEG")
        ].copy()
        if "Prompt_SHA256" in standard.columns:
            observed_shas = set(
                standard["Prompt_SHA256"]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[lambda values: values.ne("")]
            )
        elif "Prompt" in standard.columns:
            observed_shas = {
                sha256_text(str(value)) for value in standard["Prompt"].dropna()
            }
        if observed_shas == {expected_sha}:
            prompt_check = "matched"
        else:
            prompt_check = "mismatch"
            if not allow_prompt_mismatch:
                raise ValueError(
                    "The reused L4 Standard basic prompt does not match the "
                    "newly reconstructed Standard top-10 prompt. This prevents "
                    "silent reuse of non-comparable predictions. Inspect "
                    f"{prompt_manifest} or add --allow-l4-prompt-mismatch only "
                    "after documenting the difference."
                )
    return l4, {
        "L4_CSV_Encoding": encoding,
        "L4_Prompt_Manifest": str(prompt_manifest),
        "L4_Prompt_Check": prompt_check,
        "Expected_L4_Basic_Prompt_SHA256": expected_sha,
        "Observed_L4_Basic_Prompt_SHA256": ";".join(sorted(observed_shas)),
    }


def _choose_text(row: pd.Series, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column in row.index:
            value = "" if pd.isna(row[column]) else str(row[column]).strip()
            if value:
                return value
    return ""


def build_existing_gemini_predictions(
    audit: pd.DataFrame,
    model_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in audit.iterrows():
        standard = _choose_text(
            row, ["Standard_Answer", "Standard_CellType"]
        )
        full = _choose_text(
            row,
            [
                "Full_Pipeline_Answer",
                "Full_Pipeline_CellType",
                "Curated_Answer",
                "Curated_CellType",
            ],
        )
        if not standard or not full:
            raise ValueError(
                "Existing Gemini Standard/Full predictions are incomplete for "
                f"{row['Cluster_ID']}."
            )
        for condition, prediction in [
            ("Gemini_Standard_existing", standard),
            ("LLM_scCurator_Full_pipeline_existing", full),
        ]:
            rows.append(
                {
                    "Condition": condition,
                    "Repeat": 1,
                    "Cluster_ID": str(row["Cluster_ID"]),
                    "Raw_Prediction": prediction,
                    "Model_ID": model_label,
                    "Inference_Mode": "Existing primary analysis; no new inference",
                    "Formal_Head_to_Head": False,
                    "Backends_Matched": False,
                    "Input_Budgets_Matched": False,
                    "Descriptive_Only": True,
                }
            )
    return pd.DataFrame(rows)


def load_project_scorer() -> tuple[
    Any,
    Callable[..., float],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    try:
        cd8_module = importlib.import_module("benchmarks.cd8_config")
        scoring_module = importlib.import_module(
            "benchmarks.hierarchical_scoring"
        )
        return (
            cd8_module.CD8_HIER_CFG,
            scoring_module.score_hierarchical,
            scoring_module._parse_major_lineage_generic,
            scoring_module._parse_state_generic,
            scoring_module._expected_major_state_generic,
        )
    except Exception as exc:
        raise ImportError(
            "Could not import the frozen manuscript scorer. Run from the "
            "repository root with benchmarks/ on PYTHONPATH. This script does "
            "not approximate S_anno."
        ) from exc


def score_predictions(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    cfg, score_fn, parse_major, parse_state, expected_state = load_project_scorer()
    scored = predictions.merge(
        audit[["Cluster_ID", "Ground_Truth"]],
        on="Cluster_ID",
        how="left",
        validate="many_to_one",
    )
    scores: list[float] = []
    pred_states: list[str] = []
    gt_states: list[str] = []
    pred_majors: list[str] = []
    gt_majors: list[str] = []
    for _, row in scored.iterrows():
        score_row = pd.Series(
            {
                "Ground_Truth": str(row["Ground_Truth"]),
                "Pred_Text": str(row["Raw_Prediction"]),
            }
        )
        scores.append(float(score_fn(score_row, "Pred_Text", cfg=cfg)))
        # Use the same normalization and lineage/state parsers as S_anno.
        normalized = str(row["Raw_Prediction"]).lower().replace("*", "").replace("\n", " ").strip()
        pred_state = str(parse_state(normalized, cfg))
        pred_major = str(parse_major(normalized, cfg))
        gt_major_raw, gt_state_raw = expected_state(str(row["Ground_Truth"]), cfg)
        gt_state = str(gt_state_raw)
        pred_states.append(pred_state)
        gt_states.append(gt_state)
        pred_majors.append(pred_major)
        gt_majors.append(str(gt_major_raw))
    scored["Pred_State"] = pred_states
    scored["GT_State"] = gt_states
    scored["Pred_Major"] = pred_majors
    scored["GT_Major"] = gt_majors
    scored["Score_Sanno"] = scores
    scored["Exact_State"] = scored["Pred_State"] == scored["GT_State"]
    scored["Major_Lineage_Match"] = (
        scored["Pred_Major"] == scored["GT_Major"]
    )
    scored["Hierarchy_Consistent"] = scored["Score_Sanno"] >= 0.5
    return scored


def check_existing_gemini_scores(
    scored: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Confirm that rescoring reused Gemini labels reproduces LetterTables."""
    mappings = [
        ("Gemini_Standard_existing", "Score_Standard"),
        ("LLM_scCurator_Full_pipeline_existing", "Score_Curated"),
    ]
    rows: list[pd.DataFrame] = []
    for condition, audit_column in mappings:
        if audit_column not in audit.columns:
            raise ValueError(
                f"Cannot audit reused Gemini scores: {audit_column} is absent."
            )
        observed = scored[
            scored["Condition"].eq(condition)
        ][["Cluster_ID", "Score_Sanno"]].copy()
        expected = audit[["Cluster_ID", audit_column]].copy()
        checked = observed.merge(
            expected,
            on="Cluster_ID",
            how="outer",
            validate="one_to_one",
        )
        checked.insert(0, "Condition", condition)
        checked = checked.rename(columns={audit_column: "LetterTables_Sanno"})
        checked["Absolute_Difference"] = (
            checked["Score_Sanno"] - checked["LetterTables_Sanno"]
        ).abs()
        checked["Within_Tolerance"] = (
            checked["Absolute_Difference"] <= tolerance
        )
        rows.append(checked)
    result = pd.concat(rows, ignore_index=True)
    if not result["Within_Tolerance"].all():
        bad = result.loc[
            ~result["Within_Tolerance"],
            [
                "Condition",
                "Cluster_ID",
                "Score_Sanno",
                "LetterTables_Sanno",
                "Absolute_Difference",
            ],
        ].head(10)
        raise ValueError(
            "Rescored existing Gemini predictions do not reproduce the frozen "
            "LetterTables scores. Do not mix scorer versions. Examples:\n"
            f"{bad.to_string(index=False)}"
        )
    return result


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    bootstrap = array[indices].mean(axis=1)
    return (
        mean,
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def sign_flip_pvalue(
    differences: Sequence[float],
    *,
    seed: int,
    max_exact_n: int = 20,
) -> float:
    array = np.asarray(differences, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan
    observed = abs(float(array.mean()))
    if len(array) <= max_exact_n:
        signs = np.asarray(
            list(itertools.product([-1.0, 1.0], repeat=len(array)))
        )
        statistics = np.abs((signs * array).mean(axis=1))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(100_000, len(array)))
        statistics = np.abs((signs * array).mean(axis=1))
    return float(
        (np.count_nonzero(statistics >= observed - 1e-15) + 1)
        / (len(statistics) + 1)
    )


def summarize_predictions(
    scored: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_cluster = (
        scored.groupby(
            ["Condition", "Cluster_ID", "Ground_Truth"],
            as_index=False,
        )
        .agg(
            Mean_Sanno=("Score_Sanno", "mean"),
            Exact_State_Rate=("Exact_State", "mean"),
            Major_Lineage_Accuracy=("Major_Lineage_Match", "mean"),
            Hierarchy_Consistent_Rate=("Hierarchy_Consistent", "mean"),
            Modal_Prediction=(
                "Raw_Prediction",
                lambda values: (
                    values.astype(str).mode().iloc[0]
                    if not values.astype(str).mode().empty
                    else ""
                ),
            ),
            Modal_Pred_State=(
                "Pred_State",
                lambda values: (
                    values.astype(str).mode().iloc[0]
                    if not values.astype(str).mode().empty
                    else ""
                ),
            ),
            N_Repeats=("Repeat", "nunique"),
        )
    )
    rows: list[dict[str, Any]] = []
    for index, condition in enumerate(CONDITION_ORDER):
        subset = per_cluster[per_cluster["Condition"].eq(condition)]
        if subset.empty:
            continue
        mean, low, high = bootstrap_mean_ci(
            subset["Mean_Sanno"], n_bootstrap, seed + index
        )
        rows.append(
            {
                "Panel": (
                    "Workflow-level descriptive reference"
                    if condition in WORKFLOW_CONDITIONS
                    else "Same-backend manual prompt/heuristic controls"
                ),
                "Condition": condition,
                "N_Clusters": len(subset),
                "N_Repeats": int(subset["N_Repeats"].max()),
                "Mean_Sanno": mean,
                "Mean_Sanno_Bootstrap_CI_Low": low,
                "Mean_Sanno_Bootstrap_CI_High": high,
                "Exact_State_Agreement": float(
                    subset["Exact_State_Rate"].mean()
                ),
                "Major_Lineage_Accuracy": float(
                    subset["Major_Lineage_Accuracy"].mean()
                ),
                "Hierarchy_Consistent_Accuracy": float(
                    subset["Hierarchy_Consistent_Rate"].mean()
                ),
                "Formal_Head_to_Head": False,
                "Descriptive_Only": condition in WORKFLOW_CONDITIONS,
            }
        )
    return per_cluster, pd.DataFrame(rows)


def paired_contrast_row(
    per_cluster: pd.DataFrame,
    *,
    left: str,
    right: str,
    n_bootstrap: int,
    seed: int,
    allow_inference: bool,
    interpretation: str,
) -> dict[str, Any]:
    wide = per_cluster.pivot(
        index="Cluster_ID", columns="Condition", values="Mean_Sanno"
    )
    paired = wide[[left, right]].dropna()
    differences = (paired[left] - paired[right]).to_numpy(dtype=float)
    mean, low, high = bootstrap_mean_ci(
        differences, n_bootstrap, seed
    )
    wilcoxon_p = np.nan
    sign_flip_p = np.nan
    if allow_inference:
        try:
            from scipy.stats import wilcoxon

            wilcoxon_p = (
                1.0
                if np.allclose(differences, 0)
                else float(
                    wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            )
        except Exception:
            wilcoxon_p = np.nan
        sign_flip_p = sign_flip_pvalue(differences, seed=seed)
    return {
        "Contrast": f"{left}_minus_{right}",
        "N_Paired_Clusters": len(differences),
        "Mean_Paired_Difference_Sanno": mean,
        "Bootstrap_95CI_Low": low,
        "Bootstrap_95CI_High": high,
        "Paired_Wilcoxon_P": wilcoxon_p,
        "Two_Sided_Sign_Flip_P": sign_flip_p,
        "N_Improved": int((differences > 0).sum()),
        "N_Unchanged": int(np.isclose(differences, 0).sum()),
        "N_Worsened": int((differences < 0).sum()),
        "Formal_Inference": allow_inference,
        "Interpretation": interpretation,
    }


def build_contrasts(
    per_cluster: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual_pairs = [
        (
            "Manual_o3_Standard_CoT_top10",
            REUSED_L4_CONDITION,
            "Prompt sensitivity under manual o3; reused L4 basic runs.",
        ),
        (
            "Manual_o3_RegexMask_CoT_top10",
            "Manual_o3_Standard_CoT_top10",
            "Prespecified regex-mask effect with model, prompt, top-10 budget, and repeats matched.",
        ),
    ]
    manual_rows = [
        paired_contrast_row(
            per_cluster,
            left=left,
            right=right,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
            allow_inference=True,
            interpretation=interpretation,
        )
        for index, (left, right, interpretation) in enumerate(
            manual_pairs, start=10
        )
    ]
    workflow = paired_contrast_row(
        per_cluster,
        left="LLM_scCurator_Full_pipeline_existing",
        right=API_CONDITION,
        n_bootstrap=n_bootstrap,
        seed=seed + 50,
        allow_inference=False,
        interpretation=(
            "Descriptive CD8 workflow-level difference only. Backend, prompt, "
            "context augmentation, and input budget are not matched; this is "
            "not a superiority test."
        ),
    )
    workflow.update(
        {
            "Formal_Head_to_Head": False,
            "Backends_Matched": False,
            "Input_Budgets_Matched": False,
            "Descriptive_Only": True,
        }
    )
    return pd.DataFrame(manual_rows), pd.DataFrame([workflow])


def _draw_cluster_panel(
    axis: Any,
    per_cluster: pd.DataFrame,
    summary: pd.DataFrame,
    conditions: Sequence[str],
    title: str,
) -> None:
    wide = per_cluster.pivot(
        index="Cluster_ID", columns="Condition", values="Mean_Sanno"
    )[list(conditions)].dropna()
    x = np.arange(len(conditions), dtype=float)
    for _, row in wide.iterrows():
        y = row[list(conditions)].to_numpy(dtype=float)
        axis.plot(x, y, color="#C4C4C4", lw=0.65, alpha=0.55, zorder=1)
        axis.scatter(
            x,
            y,
            c=[CONDITION_COLORS[c] for c in conditions],
            s=17,
            alpha=0.8,
            zorder=2,
        )
    for index, condition in enumerate(conditions):
        row = summary[summary["Condition"].eq(condition)].iloc[0]
        mean = float(row["Mean_Sanno"])
        low = float(row["Mean_Sanno_Bootstrap_CI_Low"])
        high = float(row["Mean_Sanno_Bootstrap_CI_High"])
        axis.errorbar(
            index,
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            fmt="D",
            ms=5,
            color="black",
            mfc="white",
            capsize=2.5,
            lw=1.1,
            zorder=4,
        )
    axis.set_xticks(
        x, [CONDITION_LABELS[condition] for condition in conditions]
    )
    axis.set_ylabel(r"$S_{anno}$")
    axis.set_ylim(-0.04, 1.08)
    axis.set_title(title, loc="left", fontweight="bold")


def _draw_metric_panel(
    axis: Any,
    summary: pd.DataFrame,
    conditions: Sequence[str],
    title: str,
    show_legend: bool = True,
) -> None:
    metrics = [
        ("Mean_Sanno", r"Mean $S_{anno}$"),
        ("Exact_State_Agreement", "Exact\nstate"),
        ("Major_Lineage_Accuracy", "Major\nlineage"),
        ("Hierarchy_Consistent_Accuracy", "Hierarchy-\nconsistent"),
    ]
    positions = np.arange(len(metrics), dtype=float)
    width = 0.8 / len(conditions)
    for index, condition in enumerate(conditions):
        row = summary[summary["Condition"].eq(condition)].iloc[0]
        values = [100 * float(row[column]) for column, _ in metrics]
        offset = (index - (len(conditions) - 1) / 2) * width
        axis.bar(
            positions + offset,
            values,
            width=width,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition].replace("\n", " "),
        )
    axis.set_xticks(positions, [label for _, label in metrics])
    axis.set_ylabel("Score or accuracy (%)")
    axis.set_ylim(0, 108)
    axis.set_title(title, loc="left", fontweight="bold")

    if show_legend:
        axis.legend(
            frameon=False,
            fontsize=7.5,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            handlelength=1.4,
            handletextpad=0.6,
            labelspacing=0.7,
        )


def _individual_panel_paths(
    out_pdf: Path,
    out_png: Path,
    panel_letter: str,
) -> tuple[Path, Path]:
    """
    Construct output paths for an individual figure panel.

    Example
    -------
    ReviewerFig_L5_GPTCelltype_prompt_controls_A.pdf
    ReviewerFig_L5_GPTCelltype_prompt_controls_A.png
    """
    panel_pdf = out_pdf.with_name(
        f"{out_pdf.stem}_{panel_letter}{out_pdf.suffix}"
    )
    panel_png = out_png.with_name(
        f"{out_png.stem}_{panel_letter}{out_png.suffix}"
    )
    return panel_pdf, panel_png


def _save_individual_panel(
    draw_panel: Callable[[Any], None],
    *,
    panel_letter: str,
    out_pdf: Path,
    out_png: Path,
    figsize: tuple[float, float] = (5.2, 4.1),
    left: float = 0.16,
    right: float = 0.97,
    bottom: float = 0.20,
    top: float = 0.90,
) -> None:
    """Draw one panel and save it as an independent PDF and PNG."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize)
    draw_panel(axis)

    figure.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )

    panel_pdf, panel_png = _individual_panel_paths(
        out_pdf,
        out_png,
        panel_letter,
    )

    figure.savefig(
        panel_pdf,
        bbox_inches="tight",
    )
    figure.savefig(
        panel_png,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)

    
def make_figure(
    per_cluster: pd.DataFrame,
    summary: pd.DataFrame,
    workflow_contrast: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    workflow = workflow_contrast.iloc[0]

    def draw_panel_a(axis: Any) -> None:
        _draw_cluster_panel(
            axis,
            per_cluster,
            summary,
            WORKFLOW_PLOT_CONDITIONS,
            "A  Workflow-level CD8 reference",
        )
        axis.text(
            0.5,
            0.965,
            (
                "Full minus native GPTCelltype: "
                f"{100 * workflow['Mean_Paired_Difference_Sanno']:+.1f} pp "
                "(descriptive)"
            ),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
        )

    def draw_panel_b(axis: Any) -> None:
        _draw_metric_panel(
            axis,
            summary,
            WORKFLOW_PLOT_CONDITIONS,
            "B  Workflow metrics (colors as A)",
        )

    def draw_panel_c(axis: Any) -> None:
        _draw_cluster_panel(
            axis,
            per_cluster,
            summary,
            PROMPT_CONTROL_CONDITIONS,
            "C  Same-backend manual o3 controls",
        )

    def draw_panel_d(axis: Any) -> None:
        _draw_metric_panel(
            axis,
            summary,
            PROMPT_CONTROL_CONDITIONS,
            "D  Prompt/filter metrics (colors as C)",
        )

    # ------------------------------------------------------------
    # Combined 2 x 2 figure
    # ------------------------------------------------------------
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.1),
    )

    draw_panel_a(axes[0, 0])
    draw_panel_b(axes[0, 1])
    draw_panel_c(axes[1, 0])
    draw_panel_d(axes[1, 1])

    figure.suptitle(
        (
            "Reviewer Figure L5: GPTCelltype reference and "
            "low-cost prompt/filter controls"
        ),
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        (
            "Panels A-B are descriptive: backend, prompt, context, and input "
            "budget are not matched. Panels C-D use manual o3; basic runs are "
            "reused from L4 and CoT runs use fresh chats."
        ),
        ha="center",
        va="bottom",
        fontsize=7,
    )

    figure.tight_layout(
        rect=(0, 0.03, 1, 0.965),
    )

    figure.savefig(
        out_pdf,
        bbox_inches="tight",
    )
    figure.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)

    # ------------------------------------------------------------
    # Individual panel A
    # ------------------------------------------------------------
    _save_individual_panel(
        draw_panel_a,
        panel_letter="A",
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(4.2, 4.1),
        left=0.14,
        right=0.97,
        bottom=0.20,
        top=0.90,
    )

    # ------------------------------------------------------------
    # Individual panel B
    # ------------------------------------------------------------
    _save_individual_panel(
        draw_panel_b,
        panel_letter="B",
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(6.5, 4.1),
        left=0.14,
        right=0.68,
        bottom=0.20,
        top=0.90,
    )

    # ------------------------------------------------------------
    # Individual panel C
    # ------------------------------------------------------------
    _save_individual_panel(
        draw_panel_c,
        panel_letter="C",
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(5.5, 4.1),
        left=0.15,
        right=0.98,
        bottom=0.22,
        top=0.90,
    )

    # ------------------------------------------------------------
    # Individual panel D
    # ------------------------------------------------------------
    _save_individual_panel(
        draw_panel_d,
        panel_letter="D",
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(6.8, 4.1),
        left=0.14,
        right=0.65,
        bottom=0.22,
        top=0.90,
    )


def run_r_helper(
    *,
    rscript: str,
    helper_path: Path,
    marker_path: Path,
    prediction_path: Path,
    metadata_path: Path,
    prompt_path: Path,
    model: str,
    tissue: str,
    repeats: int,
    inter_call_sleep_sec: float,
    max_api_retries: int,
    retry_base_wait_sec: float,
    timeout_sec: int,
) -> None:
    if shutil.which(rscript) is None:
        raise FileNotFoundError(f"Rscript executable not found: {rscript}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. It is read only from the environment."
        )
    subprocess.run(
        [
            rscript,
            str(helper_path),
            str(marker_path),
            str(prediction_path),
            str(metadata_path),
            str(prompt_path),
            model,
            tissue,
            str(repeats),
            str(inter_call_sleep_sec),
            str(max_api_retries),
            str(retry_base_wait_sec),
        ],
        check=True,
        timeout=timeout_sec,
    )


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    input_xlsx: Path,
    status: str,
    l4_metadata: dict[str, Any],
    manual_template_status: dict[str, Any],
    r_metadata_path: Path,
) -> None:
    metadata: dict[str, Any] = {
        "Analysis_ID": ANALYSIS_ID,
        "Script_Build": SCRIPT_BUILD,
        "Status": status,
        "Timestamp": now_iso(),
        "Input_XLSX": str(input_xlsx.resolve()),
        "Input_XLSX_SHA256": sha256_file(input_xlsx),
        "Audit_Sheet": args.audit_sheet,
        "Marker_Sheet": args.marker_sheet,
        "Dataset": args.dataset,
        "Standard_Variant": args.standard_variant,
        "Regex_Mask_Variant": args.regex_variant,
        "Top_Genes": args.top_genes,
        "Native_GPTCelltype_Model": args.api_model,
        "Native_GPTCelltype_Repeats": args.api_repeats,
        "Expected_New_API_Calls": args.api_repeats,
        "Manual_Model_Label": args.manual_model,
        "Manual_Repeats": args.manual_repeats,
        "Expected_New_Manual_Chats": (
            len(MANUAL_CONDITIONS) * args.manual_repeats
        ),
        "Tissue_Name": args.tissue,
        "Paper_CoT_Sentence": PAPER_COT_SENTENCE,
        "CoT_Qualification": (
            "Paper-defined CD3/CoT-inspired prompt; not assumed to be stronger "
            "for CD8 substate resolution."
        ),
        "L4_Basic_Predictions_Reused": str(args.l4_manual_predictions),
        "Existing_Gemini_Inference_Reused": True,
        "Formal_Head_to_Head": False,
        "Cross_Workflow_Backends_Matched": False,
        "Cross_Workflow_Prompts_Matched": False,
        "Cross_Workflow_Input_Budgets_Matched": False,
        "Cross_Workflow_Descriptive_Only": True,
        "Cross_Workflow_P_Values_Computed": False,
        "Manual_Regex_Control_Matched": (
            "model;prompt;top10_budget;repeats;cluster_order"
        ),
        "Sanno_Scorer": (
            "benchmarks.hierarchical_scoring.score_hierarchical with "
            "CD8_HIER_CFG"
        ),
        "Python": sys.version.replace("\n", " "),
        "Python_Executable": sys.executable,
        "Platform": platform.platform(),
        "Hostname": socket.gethostname(),
        "API_Key_Recorded": False,
    }
    metadata.update(l4_metadata)
    metadata.update(
        {
            f"Manual_Template_{key}": value
            for key, value in manual_template_status.items()
        }
    )
    if r_metadata_path.exists():
        r_metadata = pd.read_csv(r_metadata_path)
        for _, row in r_metadata.iterrows():
            metadata[f"R_{row['Key']}"] = row["Value"]
    pd.DataFrame(
        {"Key": list(metadata), "Value": list(metadata.values())}
    ).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the low-cost Reviewer Figure L5 native GPTCelltype "
            "reference and manual o3 prompt/filter controls."
        )
    )
    parser.add_argument("--in-xlsx", default="LetterTables.xlsx")
    parser.add_argument("--audit-sheet", default="L2_per_cluster_audit")
    parser.add_argument("--marker-sheet", default="L3_per_gene_marker_list")
    parser.add_argument("--dataset", default="CD8")
    parser.add_argument("--standard-variant", default="standard")
    parser.add_argument("--regex-variant", default="regex_mask")
    parser.add_argument("--top-genes", type=int, default=10)
    parser.add_argument(
        "--l4-manual-predictions",
        default=(
            "paper/revision_tables/l4_manual_o3_top10_final/"
            "L4_manual_o3_predictions.csv"
        ),
    )
    parser.add_argument(
        "--allow-l4-prompt-mismatch",
        action="store_true",
        help=(
            "Allow reuse when the L4 prompt hash differs. Use only after "
            "inspecting and documenting the difference."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="paper/revision_tables/l5_reviewer1_low_cost",
    )
    parser.add_argument("--figure-dir", default="paper/revision_figures")
    parser.add_argument(
        "--figure-stem",
        default="ReviewerFig_L5_GPTCelltype_prompt_controls",
    )
    parser.add_argument("--api-model", default="gpt-4")
    parser.add_argument("--api-repeats", type=int, default=1)
    parser.add_argument("--manual-model", default="o3")
    parser.add_argument("--manual-repeats", type=int, default=5)
    parser.add_argument(
        "--gemini-model-label",
        default="Gemini 2.5 Pro (existing primary analysis)",
    )
    parser.add_argument(
        "--tissue", default="human tumor-infiltrating CD8 T"
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--api-timeout-sec", type=int, default=900)
    parser.add_argument("--inter-call-sleep-sec", type=float, default=12.0)
    parser.add_argument("--max-api-retries", type=int, default=6)
    parser.add_argument("--retry-base-wait-sec", type=float, default=10.0)
    args = parser.parse_args()
    if args.top_genes < 1:
        parser.error("--top-genes must be at least 1.")
    if args.api_repeats < 1:
        parser.error("--api-repeats must be at least 1.")
    if args.manual_repeats < 1:
        parser.error("--manual-repeats must be at least 1.")
    if args.inter_call_sleep_sec < 0:
        parser.error("--inter-call-sleep-sec cannot be negative.")
    if args.max_api_retries < 0:
        parser.error("--max-api-retries cannot be negative.")
    if args.retry_base_wait_sec <= 0:
        parser.error("--retry-base-wait-sec must be positive.")
    if args.force_api and not args.run_api:
        parser.error("--force-api requires --run-api.")
    return args


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.in_xlsx)
    l4_path = Path(args.l4_manual_predictions)
    outdir = Path(args.outdir)
    figure_dir = Path(args.figure_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    api_input_path = outdir / "L5_GPTCelltype_native_top10_input.csv"
    api_prompt_path = outdir / "L5_GPTCelltype_native_prompt.csv"
    api_prediction_path = outdir / "L5_GPTCelltype_native_predictions.csv"
    api_r_metadata_path = outdir / "L5_GPTCelltype_R_metadata.csv"
    r_helper_path = outdir / "L5_gptcelltype_native_runner.R"
    manual_input_path = outdir / "L5_manual_marker_inputs.csv"
    manual_prompt_path = outdir / "L5_manual_prompt_manifest.csv"
    manual_prediction_path = outdir / "L5_manual_predictions.csv"
    manual_protocol_path = outdir / "L5_manual_protocol.txt"
    reused_l4_path = outdir / "L5_reused_L4_standard_basic_predictions.csv"
    existing_gemini_path = outdir / "L5_existing_Gemini_predictions.csv"
    existing_score_check_path = (
        outdir / "L5_existing_Gemini_score_reproduction_check.csv"
    )
    combined_path = outdir / "L5_all_predictions.csv"
    scored_path = outdir / "L5_all_predictions_scored.csv"
    per_cluster_path = outdir / "L5_per_cluster_metrics.csv"
    summary_path = outdir / "L5_summary.csv"
    manual_contrast_path = outdir / "L5_manual_matched_contrasts.csv"
    workflow_contrast_path = outdir / "L5_workflow_descriptive_contrast.csv"
    metadata_path = outdir / "L5_run_metadata.csv"
    out_pdf = figure_dir / f"{args.figure_stem}.pdf"
    out_png = figure_dir / f"{args.figure_stem}.png"

    if not input_xlsx.exists():
        raise FileNotFoundError(input_xlsx)
    audit = load_cd8_audit(input_xlsx, args.audit_sheet)
    cluster_order = audit["Cluster_ID"].astype(str).tolist()
    marker_lists = load_ranked_marker_lists(
        input_xlsx,
        args.marker_sheet,
        dataset=args.dataset,
        variants=[args.standard_variant, args.regex_variant],
        cluster_order=cluster_order,
    )
    standard = marker_lists[args.standard_variant]
    regex_mask = marker_lists[args.regex_variant]

    api_input = build_marker_input(
        audit,
        {API_CONDITION: standard},
        args.top_genes,
    )
    manual_input = build_marker_input(
        audit,
        {
            "Manual_o3_Standard_CoT_top10": standard,
            "Manual_o3_RegexMask_CoT_top10": regex_mask,
        },
        args.top_genes,
    )
    api_input.to_csv(api_input_path, index=False)
    manual_input.to_csv(manual_input_path, index=False)
    write_prompt_manifests(
        api_prompt_path,
        manual_prompt_path,
        api_input,
        manual_input,
        api_model=args.api_model,
        manual_model=args.manual_model,
        tissue=args.tissue,
        manual_repeats=args.manual_repeats,
    )
    r_helper_path.write_text(R_RUNNER, encoding="utf-8")
    manual_status = update_manual_prediction_template(
        manual_prediction_path,
        audit,
        model=args.manual_model,
        repeats=args.manual_repeats,
    )
    write_manual_protocol(
        manual_protocol_path,
        prompt_path=manual_prompt_path,
        prediction_path=manual_prediction_path,
        l4_path=l4_path,
    )

    print(f"Script build: {SCRIPT_BUILD}")
    print("Prepared low-cost Reviewer Figure L5 inputs.")
    print(f"  Clusters: {len(audit)}")
    print(f"  Native GPTCelltype: Standard DEG top {args.top_genes}")
    print(
        f"  Expected new API calls at api-repeats={args.api_repeats}: "
        f"{args.api_repeats}"
    )
    print(
        "  Manual CoT conditions: Standard top 10 and regex_mask top 10 "
        f"({2 * args.manual_repeats} fresh chats; no API)"
    )
    print(f"  Native input: {api_input_path}")
    print(f"  Manual prompt manifest: {manual_prompt_path}")
    print(f"  Manual predictions: {manual_prediction_path}")
    if manual_status["created"]:
        print("  A blank manual prediction template was created.")
    else:
        print(
            "  Existing manual rows preserved: "
            f"{manual_status['preserved_nonempty']} non-empty"
        )

    if args.run_api:
        run_needed = True
        if api_prediction_path.exists() and not args.force_api:
            try:
                api_existing = pd.read_csv(api_prediction_path)
                validate_prediction_design(
                    api_existing,
                    audit,
                    conditions=[API_CONDITION],
                    repeats=args.api_repeats,
                    model=args.api_model,
                    require_fresh_chat=False,
                    label="Native GPTCelltype predictions",
                )
                run_needed = False
                print(
                    "Complete native GPTCelltype checkpoint found; reusing: "
                    f"{api_prediction_path}"
                )
            except (
                ValueError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                print(
                    "Partial native GPTCelltype checkpoint found; resuming: "
                    f"{api_prediction_path}"
                )
                print(f"  Checkpoint status: {exc}")
        if api_prediction_path.exists() and args.force_api:
            backup_path = _available_backup_path(api_prediction_path)
            shutil.copy2(api_prediction_path, backup_path)
            api_prediction_path.unlink()
            print(f"  Previous API checkpoint backed up: {backup_path}")
        if run_needed:
            print(
                f"Running native GPTCelltype with model={args.api_model!r}; "
                f"maximum new calls={args.api_repeats}."
            )
            run_r_helper(
                rscript=args.rscript,
                helper_path=r_helper_path,
                marker_path=api_input_path,
                prediction_path=api_prediction_path,
                metadata_path=api_r_metadata_path,
                prompt_path=api_prompt_path,
                model=args.api_model,
                tissue=args.tissue,
                repeats=args.api_repeats,
                inter_call_sleep_sec=args.inter_call_sleep_sec,
                max_api_retries=args.max_api_retries,
                retry_base_wait_sec=args.retry_base_wait_sec,
                timeout_sec=args.api_timeout_sec,
            )

    l4_metadata: dict[str, Any] = {
        "L4_Prompt_Check": "not_checked",
        "L4_Predictions_Status": "not_loaded",
    }
    blockers: list[str] = []
    reused_l4: pd.DataFrame | None = None
    try:
        reused_l4, l4_metadata = load_reused_l4_basic_predictions(
            l4_path,
            audit,
            marker_input=manual_input,
            tissue=args.tissue,
            model=args.manual_model,
            repeats=args.manual_repeats,
            allow_prompt_mismatch=args.allow_l4_prompt_mismatch,
        )
        l4_metadata["L4_Predictions_Status"] = "complete_and_reused"
        reused_l4.to_csv(reused_l4_path, index=False, encoding="utf-8-sig")
    except (FileNotFoundError, ValueError) as exc:
        blockers.append(f"Reused L4 basic predictions: {exc}")
        l4_metadata["L4_Predictions_Status"] = f"blocked: {exc}"

    api_predictions: pd.DataFrame | None = None
    if api_prediction_path.exists():
        try:
            api_predictions = validate_prediction_design(
                pd.read_csv(api_prediction_path),
                audit,
                conditions=[API_CONDITION],
                repeats=args.api_repeats,
                model=args.api_model,
                require_fresh_chat=False,
                label="Native GPTCelltype predictions",
            )
        except (
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            blockers.append(f"Native GPTCelltype predictions: {exc}")
    else:
        blockers.append(
            "Native GPTCelltype predictions are absent. Inspect inputs and add "
            "--run-api for the single default API request."
        )

    manual_predictions: pd.DataFrame | None = None
    try:
        manual_raw, _ = read_csv_with_encoding_fallback(
            manual_prediction_path
        )
        manual_predictions = validate_prediction_design(
            manual_raw,
            audit,
            conditions=MANUAL_CONDITIONS,
            repeats=args.manual_repeats,
            model=args.manual_model,
            require_fresh_chat=True,
            label="L5 manual CoT predictions",
        )
    except ValueError as exc:
        blockers.append(f"Manual CoT predictions: {exc}")

    existing_gemini = build_existing_gemini_predictions(
        audit, args.gemini_model_label
    )
    existing_gemini.to_csv(existing_gemini_path, index=False)

    if blockers:
        write_metadata(
            metadata_path,
            args=args,
            input_xlsx=input_xlsx,
            status="awaiting_predictions",
            l4_metadata=l4_metadata,
            manual_template_status=manual_status,
            r_metadata_path=api_r_metadata_path,
        )
        print("Reviewer Figure L5 is prepared but not yet scored.")
        for blocker in blockers:
            print(f"  - {blocker}")
        print(
            "Complete only the missing steps, then rerun. Existing API and "
            "manual checkpoints will be reused."
        )
        return

    assert reused_l4 is not None
    assert api_predictions is not None
    assert manual_predictions is not None
    api_predictions = api_predictions.copy()
    api_predictions["Inference_Mode"] = "Native GPTCelltype API"
    api_predictions["Formal_Head_to_Head"] = False
    api_predictions["Backends_Matched"] = False
    api_predictions["Input_Budgets_Matched"] = False
    api_predictions["Descriptive_Only"] = True
    reused_l4 = reused_l4.copy()
    reused_l4["Inference_Mode"] = "Manual o3; reused from Reviewer Figure L4"
    reused_l4["Formal_Head_to_Head"] = False
    reused_l4["Backends_Matched"] = True
    reused_l4["Input_Budgets_Matched"] = True
    reused_l4["Descriptive_Only"] = False
    manual_predictions = manual_predictions.copy()
    manual_predictions["Inference_Mode"] = "Manual o3 fresh-chat CoT control"
    manual_predictions["Formal_Head_to_Head"] = False
    manual_predictions["Backends_Matched"] = True
    manual_predictions["Input_Budgets_Matched"] = True
    manual_predictions["Descriptive_Only"] = False

    combined = pd.concat(
        [
            existing_gemini,
            api_predictions,
            reused_l4,
            manual_predictions,
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    scored = score_predictions(combined, audit)
    existing_score_check = check_existing_gemini_scores(scored, audit)
    per_cluster, summary = summarize_predictions(
        scored,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    manual_contrasts, workflow_contrast = build_contrasts(
        per_cluster,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    existing_score_check.to_csv(existing_score_check_path, index=False)
    per_cluster.to_csv(per_cluster_path, index=False)
    summary.to_csv(summary_path, index=False)
    manual_contrasts.to_csv(manual_contrast_path, index=False)
    workflow_contrast.to_csv(workflow_contrast_path, index=False)
    make_figure(
        per_cluster,
        summary,
        workflow_contrast,
        out_pdf,
        out_png,
    )
    write_metadata(
        metadata_path,
        args=args,
        input_xlsx=input_xlsx,
        status="completed",
        l4_metadata=l4_metadata,
        manual_template_status=manual_status,
        r_metadata_path=api_r_metadata_path,
    )

    print("Reviewer Figure L5 completed.")
    for _, row in summary.iterrows():
        print(
            f"  {row['Condition']}: mean S_anno="
            f"{100 * row['Mean_Sanno']:.1f}%, exact state="
            f"{100 * row['Exact_State_Agreement']:.1f}%, major lineage="
            f"{100 * row['Major_Lineage_Accuracy']:.1f}%"
        )
    workflow = workflow_contrast.iloc[0]
    difference_pp = 100 * float(
        workflow["Mean_Paired_Difference_Sanno"]
    )
    if difference_pp > 1e-12:
        result_clause = (
            "had a higher mean S_anno than the native GPTCelltype top-10 "
            f"reference by {difference_pp:.1f} percentage points"
        )
    elif difference_pp < -1e-12:
        result_clause = (
            "had a lower mean S_anno than the native GPTCelltype top-10 "
            f"reference by {abs(difference_pp):.1f} percentage points"
        )
    else:
        result_clause = (
            "had the same mean S_anno as the native GPTCelltype top-10 "
            "reference"
        )
    print(
        "  Safe workflow-level wording: In this reviewer-only CD8 "
        "descriptive comparison, the existing integrated LLM-scCurator Full "
        f"pipeline {result_clause}. This was not a matched superiority test."
    )
    print(f"  Figure: {out_pdf}")
    print(f"  Figure: {out_png}")


if __name__ == "__main__":
    main()
