#!/usr/bin/env python3
"""Generate reviewer-only Figure L4 with COSG, Festem, and MarkerMap.

This script compares five marker-list inputs with a shared annotator:

  1. Standard DEG (top 10 by default, following the GPTCelltype paper)
  2. COSG (native top 50 by default)
  3. Festem
  4. MarkerMap
  5. LLM-scCurator ``full_core`` applied to the Standard DEG candidates

COSG, Festem, MarkerMap, and full_core are evaluated at their native list
lengths; Standard DEG is capped at its ranked top 10. The lists are not
truncated to a common within-cluster budget. The script reports actual input
length in panel A and labels the analysis as a descriptive, reviewer-requested
comparison rather than a matched-budget performance benchmark.

External marker input
---------------------
Festem and MarkerMap are supplied as precomputed CSV files. COSG is normally
reproduced from the same CD8 AnnData object with ``--input-h5ad``. A
precomputed COSG file can instead be supplied with ``--cosg-csv``. Preferred
CSV format (one row per cluster-specific marker):

    cluster,rank,gene

Accepted column aliases include Cluster_ID/cluster_id, Rank/gene_rank, and
Gene/marker. If rank is absent, file order is used.

Generate both files from the CD8 AnnData object with::

    python paper/revision_metrics/prepare_l4_festem_markermap.py \
      --input-h5ad cd8_benchmark_data.h5ad \
      --outdir paper/revision_tables

Festem and supervised MarkerMap may instead produce one global selected panel.
A global file may contain:

    rank,gene

A global panel cannot be passed unchanged as cluster-specific evidence: doing
so would give GPTCelltype the same marker row for every cluster. By default the
script therefore rejects global panels. To make the required conversion
explicit, use:

    --global-panel-policy deg_intersection

This retains, for each cluster, the genes selected by the external method that
also occur in that cluster's Standard DEG list. External-method rank is
preserved. The selected gene identities are not supplemented, and the resulting
cluster-specific list lengths are not matched across methods.

Manual o3 workflow (default)
----------------------------
The default mode makes no API calls. It writes:

* the exact GPTCelltype-style prompt for every condition/repeat;
* a run manifest requiring a fresh ChatGPT conversation for every run; and
* a row-level prediction template.

Paste each prompt into a new o3 chat, then paste the 17 returned labels into the
matching rows of ``L4_manual_o3_predictions.csv``. Keep the condition hidden
from the model, preserve row order, and record the displayed model label and
run date. Re-running this script then validates, scores, and plots the completed
responses. The manual workflow is described as "manual o3 annotation using the
GPTCelltype-style basic prompt", not as execution of the GPTCelltype package.

The previous native GPTCelltype API route remains available with
``--run-api --model gpt-4`` when a strict package-level comparison is desired.

Examples
--------
Prepare top-10 Standard inputs, native external/curated inputs, and five manual
o3 repetitions::

    python paper/revision_metrics/make_l4_external_marker_noise_manual_o3_top10_cosg.py \
      --marker-table LetterTables.xlsx \
      --input-h5ad cd8_benchmark_data.h5ad \
      --festem-csv paper/revision_tables/festem_cd8_markers.csv \
      --markermap-csv paper/revision_tables/markermap_cd8_markers.csv \
      --outdir paper/revision_tables/l4_manual_o3_top10_final

After completing ``L4_manual_o3_predictions.csv``, rerun the same command to
score and draw the figure.

Optional native GPTCelltype API execution::

    python paper/revision_metrics/make_l4_external_marker_noise_manual_o3_top10_cosg.py \
      --marker-table LetterTables.xlsx \
      --input-h5ad cd8_benchmark_data.h5ad \
      --festem-csv paper/revision_tables/festem_cd8_markers.csv \
      --markermap-csv paper/revision_tables/markermap_cd8_markers.csv \
      --run-api --model gpt-4 --repeats 1

For global selected panels, add ``--global-panel-policy deg_intersection``.

Requirements
------------
Python: pandas, numpy, scipy, matplotlib, openpyxl
Project: llm_sc_curator, benchmarks.cd8_config,
         benchmarks.hierarchical_scoring
Optional R API mode: GPTCelltype and openai
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Keep matplotlib from trying to write to a read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmsccurator-mpl-cache")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_ID = "ReviewerFig_L4_COSG_Festem_MarkerMap_manual_o3_native_lengths_v3"
SCRIPT_BUILD = "manual_o3_top10_cosg_20260724_public_parser_20260928"
DEFAULT_MARKER_TABLE = Path("paper/revision/LetterTables.xlsx")
DEFAULT_MARKER_SHEET = "L3_per_gene_marker_list"
DEFAULT_AUDIT_SHEET = "L2_per_cluster_audit"
DEFAULT_OUTDIR = Path("paper/revision_tables/l4_external_marker_annotation")
DEFAULT_FIGDIR = Path("paper/revision_figures")
DEFAULT_FIGURE_STEM = "ReviewerFig_L4_COSG_Festem_MarkerMap_manual_o3"

CONDITION_ORDER = [
    "Standard_DEG",
    "COSG",
    "Festem",
    "MarkerMap",
    "LLM_scCurator_full_core",
]
CONDITION_LABELS = {
    "Standard_DEG": "Standard\nDEG",
    "COSG": "COSG",
    "Festem": "Festem",
    "MarkerMap": "MarkerMap",
    "LLM_scCurator_full_core": "LLM-scCurator\nfull_core",
}
CONDITION_COLORS = {
    "Standard_DEG": "#4E79A7",
    "COSG": "#59A14F",
    "Festem": "#76B7B2",
    "MarkerMap": "#F28E2B",
    "LLM_scCurator_full_core": "#D62728",
}

COLUMN_ALIASES = {
    "dataset": ("dataset", "Dataset"),
    "cluster": ("cluster", "Cluster", "Cluster_ID", "cluster_id"),
    "variant": ("variant", "Variant", "method_variant"),
    "method": ("method", "Method", "condition", "Condition"),
    "rank": ("rank", "Rank", "gene_rank"),
    "gene": ("gene", "Gene", "names", "marker", "Marker"),
}

DATE_LIKE_GENE = re.compile(
    r"^(?:\d{1,2}[-/][A-Za-z]{3}(?:[-/]\d{2,4})?|"
    r"[A-Za-z]{3}[-/]\d{1,2}(?:[-/]\d{2,4})?|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"\d{4}-\d{1,2}-\d{1,2}(?:[ T]00:00:00)?)$"
)
EXCEL_SERIAL_LIKE_GENE = re.compile(r"^4[0-9]{4}(?:\.0+)?$")

# This helper intentionally calls GPTCelltype::gptcelltype rather than
# recreating the package prompt through a different SDK.
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
cluster_order <- unique(d$Cluster_ID)
prompt_rows <- list()
pidx <- 1L

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
    stop(paste(
      "Existing prediction file is missing columns:",
      paste(missing_result_cols, collapse = ", ")
    ))
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
    is_rate_limit <- grepl(
      "429|rate limit|tokens per min|tpm",
      error_message,
      ignore.case = TRUE
    )
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

calls_made_this_run <- 0L
write_r_metadata("started_or_resumed", calls_made_this_run)

for (condition in conditions) {
  sub <- d[d$Condition == condition, , drop = FALSE]
  sub <- sub[order(match(sub$Cluster_ID, cluster_order), sub$Rank), , drop = FALSE]
  present_clusters <- unique(sub$Cluster_ID)
  gene_lists <- lapply(
    present_clusters,
    function(cid) sub$Gene[sub$Cluster_ID == cid]
  )
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
      message(sprintf(
        "Checkpoint found; skipping condition=%s repeat=%d",
        condition, repeat_id
      ))
      next
    }
    if (calls_made_this_run > 0L && inter_call_sleep_sec > 0) {
      message(sprintf("Waiting %.1f s between API calls.", inter_call_sleep_sec))
      Sys.sleep(inter_call_sleep_sec)
    }
    message(sprintf(
      "GPTCelltype call: condition=%s repeat=%d",
      condition, repeat_id
    ))
    pred <- call_gptcelltype_with_retry(gene_lists, condition, repeat_id)
    pred <- as.character(pred)
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
    results <- results[
      !(results$Condition == condition & results$Repeat == repeat_id),
      ,
      drop = FALSE
    ]
    results <- rbind(results, new_rows)
    write_checkpoint(results, output_csv)
    calls_made_this_run <- calls_made_this_run + 1L
    write_r_metadata("checkpoint_written", calls_made_this_run)
    message(sprintf(
      "Checkpoint written: condition=%s repeat=%d",
      condition, repeat_id
    ))
  }
}

write.csv(do.call(rbind, prompt_rows), prompt_csv, row.names = FALSE, na = "")
write_r_metadata("completed", calls_made_this_run)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native-length Standard DEG, COSG, Festem, MarkerMap, and "
            "LLM-scCurator marker lists with a shared manual o3 annotator "
            "(or, optionally, the native GPTCelltype API workflow)."
        )
    )
    parser.add_argument("--marker-table", default=str(DEFAULT_MARKER_TABLE))
    parser.add_argument("--marker-sheet", default=DEFAULT_MARKER_SHEET)
    parser.add_argument("--audit-sheet", default=DEFAULT_AUDIT_SHEET)
    parser.add_argument("--dataset", default="CD8")
    parser.add_argument("--standard-variant", default="standard")
    parser.add_argument("--llmsc-variant", default="full_core")
    parser.add_argument("--festem-csv", required=True)
    parser.add_argument("--markermap-csv", required=True)
    cosg_source = parser.add_mutually_exclusive_group(required=True)
    cosg_source.add_argument(
        "--cosg-csv",
        help=(
            "Precomputed cluster-specific COSG marker CSV. A prior "
            "L4_external_marker_noise_gene_level.csv is accepted and filtered "
            "to Method=COSG, although a native top-50 COSG export is preferred."
        ),
    )
    cosg_source.add_argument(
        "--input-h5ad",
        help=(
            "AnnData file from which to reproduce native COSG top-50 lists "
            "using the original L4 settings."
        ),
    )
    parser.add_argument("--group-col", default="meta.cluster")
    parser.add_argument("--cosg-n-genes", type=int, default=50)
    parser.add_argument("--cosg-key", default="cosg_l4_manual_o3")
    parser.add_argument("--cosg-mu", type=float, default=100.0)
    parser.add_argument("--cosg-expressed-pct", type=float, default=0.1)
    parser.add_argument(
        "--cosg-remove-lowly-expressed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cosg-use-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cosg-layer",
        default=None,
        help="Optional AnnData layer for COSG; omit to use adata.X.",
    )
    parser.add_argument(
        "--global-panel-policy",
        choices=("error", "deg_intersection"),
        default="error",
        help=(
            "How to convert a global external panel to cluster-specific evidence. "
            "Default 'error' requires cluster-specific input. "
            "'deg_intersection' explicitly intersects the global panel with each "
            "cluster's Standard DEG list while preserving external-panel rank."
        ),
    )
    parser.add_argument(
        "--standard-max-genes",
        type=int,
        default=10,
        help=(
            "Maximum Standard DEG list length (default 10, following the "
            "GPTCelltype paper); 0 keeps every available gene."
        ),
    )
    parser.add_argument(
        "--llmsc-max-genes",
        type=int,
        default=0,
        help="Maximum LLM-scCurator list length; 0 keeps every available gene.",
    )
    parser.add_argument(
        "--external-max-genes",
        type=int,
        default=0,
        help=(
            "Maximum COSG/Festem/MarkerMap list length; 0 keeps every "
            "available gene."
        ),
    )
    parser.add_argument(
        "--minimum-genes",
        type=int,
        default=1,
        help="Fail if any cluster/method list contains fewer genes.",
    )
    parser.add_argument(
        "--skip-noise-diagnostic",
        action="store_true",
        help=(
            "Skip NOISE_PATTERNS/NOISE_LISTS scoring. Useful only when preparing "
            "inputs outside the project environment."
        ),
    )

    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGDIR))
    parser.add_argument("--figure-stem", default=DEFAULT_FIGURE_STEM)
    parser.add_argument(
        "--model",
        default="o3",
        help=(
            "Displayed model label for manual runs (default: o3). With "
            "--run-api, pass the intended GPTCelltype API model explicitly, "
            "for example --model gpt-4."
        ),
    )
    parser.add_argument(
        "--tissue",
        default="human tumor-infiltrating CD8 T",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent repeats; use a fresh chat for every manual run.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--api-timeout-sec", type=int, default=1800)
    parser.add_argument("--inter-call-sleep-sec", type=float, default=12.0)
    parser.add_argument("--max-api-retries", type=int, default=6)
    parser.add_argument("--retry-base-wait-sec", type=float, default=10.0)

    args = parser.parse_args()
    for name in (
        "standard_max_genes",
        "llmsc_max_genes",
        "external_max_genes",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative.")
    if args.minimum_genes < 1:
        parser.error("--minimum-genes must be at least 1.")
    if args.cosg_n_genes < 1:
        parser.error("--cosg-n-genes must be at least 1.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be at least 1.")
    if args.inter_call_sleep_sec < 0:
        parser.error("--inter-call-sleep-sec cannot be negative.")
    if args.max_api_retries < 0:
        parser.error("--max-api-retries cannot be negative.")
    if args.retry_base_wait_sec <= 0:
        parser.error("--retry-base-wait-sec must be positive.")
    if args.force_api and not args.run_api:
        parser.error("--force-api requires --run-api.")
    if args.run_api and args.model.strip().lower() == "o3":
        parser.error(
            "--run-api requires an explicit GPTCelltype API model, for example "
            "--model gpt-4. The default o3 label is for manual web-interface runs."
        )
    return args


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _find_column(
    frame: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = True,
) -> str | None:
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(
            f"Missing {logical_name!r} column. Accepted names: "
            f"{COLUMN_ALIASES[logical_name]}; observed: {list(frame.columns)}"
        )
    return None


def _normalise_dataset(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _normalise_variant(value: object) -> str:
    return str(value).strip().lower()


def _clean_gene(value: object) -> str:
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        raise ValueError(f"Date-valued gene identifier detected: {value!r}")
    if pd.isna(value):
        return ""
    gene = str(value).strip()
    if DATE_LIKE_GENE.fullmatch(gene) or EXCEL_SERIAL_LIKE_GENE.fullmatch(gene):
        raise ValueError(
            f"Possible Excel gene-to-date conversion detected: {gene!r}. "
            "Restore the original gene symbol before running Figure L4."
        )
    return gene


def _deduplicate_genes(genes: Sequence[str], label: str) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        cleaned = _clean_gene(gene)
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    if len(unique) != len([g for g in genes if str(g).strip()]):
        warnings.warn(f"Duplicate genes were removed from {label}.", stacklevel=2)
    return unique


def _limit_genes(genes: Sequence[str], maximum: int) -> list[str]:
    values = list(genes)
    return values if maximum == 0 else values[:maximum]


def _read_tabular(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def marker_lists_from_frame(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    rank_col: str,
    gene_col: str,
    label: str,
) -> dict[str, list[str]]:
    work = frame.copy()
    work["_cluster"] = work[cluster_col].astype(str).str.strip()
    work["_rank"] = pd.to_numeric(work[rank_col], errors="coerce")
    work["_gene"] = work[gene_col].map(_clean_gene)
    work = work[(work["_cluster"] != "") & (work["_gene"] != "")].copy()
    if work["_rank"].isna().any():
        bad = work.loc[work["_rank"].isna(), ["_cluster", "_gene"]].head(5)
        raise ValueError(
            f"Non-numeric marker ranks detected for {label}:\n"
            f"{bad.to_string(index=False)}"
        )
    duplicate_rank = work.duplicated(["_cluster", "_rank"], keep=False)
    if duplicate_rank.any():
        bad = work.loc[duplicate_rank, ["_cluster", "_rank", "_gene"]].head(10)
        raise ValueError(
            f"Duplicate cluster/rank rows for {label}:\n"
            f"{bad.to_string(index=False)}"
        )

    result: dict[str, list[str]] = {}
    for cluster, sub in work.sort_values(["_cluster", "_rank"]).groupby(
        "_cluster",
        sort=False,
    ):
        result[str(cluster)] = _deduplicate_genes(
            sub["_gene"].tolist(),
            f"{label}, cluster {cluster}",
        )
    return result


def read_existing_marker_lists(
    path: Path,
    sheet: str,
    dataset: str,
    standard_variant: str,
    llmsc_variant: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet)
    else:
        raw = pd.read_csv(path)

    dataset_col = _find_column(raw, "dataset")
    variant_col = _find_column(raw, "variant")
    cluster_col = _find_column(raw, "cluster")
    rank_col = _find_column(raw, "rank")
    gene_col = _find_column(raw, "gene")
    wanted_dataset = _normalise_dataset(dataset)
    wanted_variants = {
        _normalise_variant(standard_variant),
        _normalise_variant(llmsc_variant),
    }
    selected = raw[
        raw[dataset_col].map(_normalise_dataset).eq(wanted_dataset)
        & raw[variant_col].map(_normalise_variant).isin(wanted_variants)
    ].copy()
    if selected.empty:
        observed = sorted(set(raw[dataset_col].astype(str)))
        raise ValueError(
            f"No marker rows found for dataset={dataset!r}; observed={observed}"
        )
    selected["_variant"] = selected[variant_col].map(_normalise_variant)

    standard = marker_lists_from_frame(
        selected[selected["_variant"] == _normalise_variant(standard_variant)],
        cluster_col=cluster_col,
        rank_col=rank_col,
        gene_col=gene_col,
        label="Standard DEG",
    )
    curated = marker_lists_from_frame(
        selected[selected["_variant"] == _normalise_variant(llmsc_variant)],
        cluster_col=cluster_col,
        rank_col=rank_col,
        gene_col=gene_col,
        label="LLM-scCurator full_core",
    )
    if not standard:
        raise ValueError(f"No Standard rows found for variant={standard_variant!r}.")
    if not curated:
        raise ValueError(f"No full_core rows found for variant={llmsc_variant!r}.")
    return standard, curated


def load_cd8_audit(path: Path, sheet: str) -> pd.DataFrame:
    audit = pd.read_excel(path, sheet_name=sheet)
    if "Cluster_ID" not in audit.columns:
        raise ValueError(f"{sheet} does not contain Cluster_ID.")
    if "Dataset" in audit.columns:
        keep = audit["Dataset"].astype(str).str.strip().isin(
            ["CD8 T", "CD8", "CD8+ T"]
        )
    else:
        keep = audit["Cluster_ID"].astype(str).str.contains(
            "CD8",
            case=False,
            na=False,
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
            audit["Cluster_ID"].duplicated(),
            "Cluster_ID",
        ].tolist()
        raise ValueError(f"Duplicate CD8 cluster IDs: {duplicates}")
    if len(audit) != 17:
        warnings.warn(
            f"Expected 17 CD8 clusters, found {len(audit)}.",
            stacklevel=2,
        )
    return audit


def read_external_marker_lists(
    path: Path,
    *,
    method: str,
    reference_lists: Mapping[str, Sequence[str]],
    cluster_order: Sequence[str],
    global_panel_policy: str,
    method_filter: str | None = None,
) -> tuple[dict[str, list[str]], str]:
    raw = _read_tabular(path)
    method_col = _find_column(raw, "method", required=False)
    if method_filter is not None and method_col is not None:
        keep = (
            raw[method_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(method_filter.casefold())
        )
        raw = raw.loc[keep].copy()
        if raw.empty:
            raise ValueError(
                f"{path} contains a {method_col!r} column but no "
                f"{method_filter!r} rows."
            )
    gene_col = _find_column(raw, "gene")
    cluster_col = _find_column(raw, "cluster", required=False)
    rank_col = _find_column(raw, "rank", required=False)
    work = raw.copy()
    work["_gene"] = work[gene_col].map(_clean_gene)
    work = work[work["_gene"] != ""].copy()
    if work.empty:
        raise ValueError(f"{method} file contains no usable genes: {path}")

    if cluster_col is not None:
        work["_cluster"] = work[cluster_col].fillna("").astype(str).str.strip()
    else:
        work["_cluster"] = ""

    has_cluster_rows = work["_cluster"].ne("").any()
    has_global_rows = work["_cluster"].eq("").any()
    if has_cluster_rows and has_global_rows:
        raise ValueError(
            f"{method} mixes cluster-specific and global rows. Supply one format "
            "per file so the conversion is auditable."
        )

    if rank_col is None:
        if has_cluster_rows:
            work["_rank"] = work.groupby("_cluster", sort=False).cumcount() + 1
        else:
            work["_rank"] = np.arange(1, len(work) + 1)
    else:
        work["_rank"] = pd.to_numeric(work[rank_col], errors="coerce")
        if work["_rank"].isna().any():
            raise ValueError(f"{method} contains non-numeric ranks.")

    if has_cluster_rows:
        lists = marker_lists_from_frame(
            work,
            cluster_col="_cluster",
            rank_col="_rank",
            gene_col="_gene",
            label=method,
        )
        conversion = "cluster-specific external lists used as supplied"
    else:
        if global_panel_policy == "error":
            raise ValueError(
                f"{method} is a global selected panel without a cluster column. "
                "Repeating it for every cluster would provide identical evidence "
                "to GPTCelltype and is not a valid cluster-annotation comparison. "
                "Supply cluster-specific lists or rerun with "
                "--global-panel-policy deg_intersection to make the conversion "
                "explicit."
            )
        panel = _deduplicate_genes(
            work.sort_values("_rank")["_gene"].tolist(),
            f"{method} global panel",
        )
        reference_sets = {
            cluster: set(reference_lists[cluster]) for cluster in cluster_order
        }
        lists = {
            cluster: [gene for gene in panel if gene in reference_sets[cluster]]
            for cluster in cluster_order
        }
        conversion = (
            "global panel intersected with each cluster's Standard DEG list; "
            "external-panel rank preserved; no length matching"
        )

    return lists, conversion


def _column_label_to_string(label: object) -> str:
    if isinstance(label, tuple):
        parts = [str(value) for value in label if str(value) not in {"", "None"}]
        return parts[-1] if parts else str(label)
    return str(label)


def run_cosg(
    input_h5ad: Path,
    *,
    group_col: str,
    evaluated_clusters: Sequence[str],
    n_top: int,
    key_added: str,
    mu: float,
    expressed_pct: float,
    remove_lowly_expressed: bool,
    use_raw: bool,
    layer: str | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Reproduce the original native COSG lists from the benchmark AnnData."""
    try:
        import cosg
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "COSG-from-h5ad mode requires scanpy and cosg. Install the "
            "manuscript environment or provide --cosg-csv."
        ) from exc

    if not input_h5ad.exists():
        raise FileNotFoundError(f"AnnData input not found: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    if group_col not in adata.obs.columns:
        raise ValueError(f"group_col {group_col!r} not found in adata.obs")
    observed_clusters = set(adata.obs[group_col].astype(str))
    missing = sorted(set(evaluated_clusters) - observed_clusters)
    if missing:
        raise ValueError(f"Evaluated clusters absent from AnnData: {missing}")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"COSG layer {layer!r} not found in adata.layers")

    kwargs: dict[str, Any] = {
        "adata": adata,
        "key_added": key_added,
        "mu": mu,
        "expressed_pct": expressed_pct,
        "remove_lowly_expressed": remove_lowly_expressed,
        "n_genes_user": n_top,
        "groupby": group_col,
        "groups": "all",
        "use_raw": use_raw,
    }
    if layer is not None:
        kwargs["layer"] = layer

    signature = inspect.signature(cosg.cosg)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        unsupported = sorted(set(kwargs) - set(signature.parameters))
        for name in unsupported:
            if name in {"groups", "use_raw", "layer"}:
                warnings.warn(
                    f"Installed COSG does not expose argument {name!r}; omitting it.",
                    stacklevel=2,
                )
                kwargs.pop(name)

    cosg.cosg(**kwargs)
    if key_added not in adata.uns or "names" not in adata.uns[key_added]:
        raise RuntimeError(
            f"COSG did not create adata.uns[{key_added!r}]['names']"
        )

    names = pd.DataFrame(adata.uns[key_added]["names"])
    column_lookup = {
        _column_label_to_string(column): column for column in names.columns
    }
    result: dict[str, list[str]] = {}
    for cluster in evaluated_clusters:
        if cluster not in column_lookup:
            raise ValueError(
                f"COSG output has no column for cluster {cluster!r}; "
                f"observed columns: {list(column_lookup)}"
            )
        raw_genes = names[column_lookup[cluster]].dropna().tolist()
        genes = _deduplicate_genes(raw_genes, f"COSG {cluster}")[:n_top]
        result[cluster] = genes

    metadata = {
        "Source_Mode": "recomputed_from_h5ad",
        "COSG_Version": package_version("cosg"),
        "Scanpy_Version": package_version("scanpy"),
        "Input_H5AD": str(input_h5ad.resolve()),
        "Input_H5AD_SHA256": sha256_file(input_h5ad),
        "Group_Column": group_col,
        "N_Genes_User": str(n_top),
        "Mu": str(mu),
        "Expressed_Pct": str(expressed_pct),
        "Remove_Lowly_Expressed": str(remove_lowly_expressed),
        "Use_Raw": str(use_raw),
        "Layer": "" if layer is None else layer,
    }
    return result, metadata


def ensure_cluster_alignment(
    method_lists: Mapping[str, Mapping[str, Sequence[str]]],
    cluster_order: Sequence[str],
    minimum_genes: int,
) -> None:
    expected = set(cluster_order)
    for method, lists in method_lists.items():
        observed = set(lists)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            raise ValueError(f"{method} is missing evaluated clusters: {missing}")
        if extra:
            warnings.warn(
                f"{method} has extra clusters that will be ignored: {extra}",
                stacklevel=2,
            )
        short = {
            cluster: len(lists[cluster])
            for cluster in cluster_order
            if len(lists[cluster]) < minimum_genes
        }
        if short:
            raise ValueError(
                f"{method} has cluster lists shorter than --minimum-genes="
                f"{minimum_genes}: {short}"
            )


def build_native_length_inputs(
    audit: pd.DataFrame,
    method_lists: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ground_truth = audit.set_index("Cluster_ID")["Ground_Truth"].astype(str)
    rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    cluster_order = audit["Cluster_ID"].astype(str).tolist()
    for condition in CONDITION_ORDER:
        for cluster in cluster_order:
            genes = list(method_lists[condition][cluster])
            length_rows.append(
                {
                    "Condition": condition,
                    "Cluster_ID": cluster,
                    "Ground_Truth": ground_truth.loc[cluster],
                    "N_Genes": len(genes),
                    "Budget_Matched_Across_Methods": False,
                }
            )
            for rank, gene in enumerate(genes, start=1):
                rows.append(
                    {
                        "Condition": condition,
                        "Cluster_ID": cluster,
                        "Ground_Truth": ground_truth.loc[cluster],
                        "Gene": gene,
                        "Rank": rank,
                        "N_Genes_In_List": len(genes),
                        "Budget_Matched_Across_Methods": False,
                    }
                )
    marker_input = pd.DataFrame(rows)
    lengths = pd.DataFrame(length_rows)
    marker_input["Condition"] = pd.Categorical(
        marker_input["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    marker_input = marker_input.sort_values(
        ["Condition", "Cluster_ID", "Rank"]
    ).reset_index(drop=True)
    marker_input["Condition"] = marker_input["Condition"].astype("object")
    return marker_input, lengths


def native_prompt(gene_input: pd.DataFrame, condition: str, tissue: str) -> str:
    subset = gene_input[gene_input["Condition"] == condition]
    rows = []
    for _, cluster_rows in subset.groupby("Cluster_ID", sort=False):
        rows.append(
            ",".join(
                cluster_rows.sort_values("Rank")["Gene"].astype(str)
            )
        )
    n_rows = len(rows)
    return (
        f"Identify cell types of {tissue} cells using the following markers "
        "separately for each row. Only provide the cell type name. Do not show "
        "numbers before the name. Some can be a mixture of multiple cell types.\n"
        f"Return exactly {n_rows} non-empty lines, one cell-type annotation per "
        "input row, in the same order, with no header, bullets, numbering, or "
        "explanation.\n"
        + "\n".join(rows)
    )


def write_prepared_prompts(
    path: Path,
    gene_input: pd.DataFrame,
    model: str,
    tissue: str,
    repeats: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        prompt = native_prompt(gene_input, condition, tissue)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for repeat in range(1, repeats + 1):
            rows.append(
                {
                    "Run_ID": f"{condition}__r{repeat:02d}",
                    "Condition": condition,
                    "Repeat": repeat,
                    "Model_Label": model,
                    "Tissue_Name": tissue,
                    "Fresh_Chat_Required": True,
                    "Prompt_SHA256": prompt_sha256,
                    "Prompt": prompt,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def build_manual_prediction_template(
    audit: pd.DataFrame,
    *,
    model: str,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_order = audit["Cluster_ID"].astype(str).tolist()
    for condition in CONDITION_ORDER:
        for repeat in range(1, repeats + 1):
            run_id = f"{condition}__r{repeat:02d}"
            for row_index, cluster_id in enumerate(cluster_order, start=1):
                rows.append(
                    {
                        "Run_ID": run_id,
                        "Condition": condition,
                        "Repeat": repeat,
                        "Row_Index": row_index,
                        "Cluster_ID": cluster_id,
                        "Raw_Prediction": "",
                        "Model_ID": model,
                        "Run_Date": "",
                        "Fresh_Chat": "",
                        "Notes": "",
                    }
                )
    return pd.DataFrame(rows)


def read_csv_with_encoding_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    """Read common Excel CSV encodings without silently dropping bytes."""
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
        f"Could not decode prediction CSV {path}. Attempts: {' | '.join(errors)}"
    )


def _available_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.stem}.pre_cosg_backup{path.suffix}")
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = path.with_name(
            f"{path.stem}.pre_cosg_backup_{index}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a backup name beside {path}")


def update_manual_prediction_template(
    path: Path,
    audit: pd.DataFrame,
    *,
    model: str,
    repeats: int,
) -> dict[str, Any]:
    """
    Create or extend the manual template while preserving completed predictions.

    Existing four-condition rows are merged into the five-condition design by
    key. The original byte-for-byte CSV is backed up before any migration, and
    the active file is normalized to UTF-8 with BOM for Excel compatibility.
    """
    expected = build_manual_prediction_template(
        audit,
        model=model,
        repeats=repeats,
    )
    if not path.exists():
        expected.to_csv(path, index=False, encoding="utf-8-sig")
        return {
            "created": True,
            "encoding": "new_utf-8-sig",
            "preserved_rows": 0,
            "preserved_nonempty_predictions": 0,
            "added_rows": len(expected),
            "backup_path": None,
        }

    existing, encoding = read_csv_with_encoding_fallback(path)
    key_columns = ["Condition", "Repeat", "Cluster_ID"]
    missing_columns = [
        column for column in key_columns if column not in existing.columns
    ]
    if missing_columns:
        raise ValueError(
            "Existing manual prediction CSV cannot be extended because key "
            f"columns are missing: {missing_columns}"
        )

    existing = existing.copy()
    existing["Condition"] = existing["Condition"].astype(str).str.strip()
    existing["Cluster_ID"] = existing["Cluster_ID"].astype(str).str.strip()
    existing["Repeat"] = pd.to_numeric(
        existing["Repeat"],
        errors="raise",
    ).astype(int)
    duplicated = existing.duplicated(key_columns, keep=False)
    if duplicated.any():
        bad = existing.loc[duplicated, key_columns].head(10)
        raise ValueError(
            "Existing manual prediction CSV has duplicate keys:\n"
            f"{bad.to_string(index=False)}"
        )

    expected_keys = set(map(tuple, expected[key_columns].to_numpy()))
    existing_keys = set(map(tuple, existing[key_columns].to_numpy()))
    unexpected = sorted(existing_keys - expected_keys)
    if unexpected:
        raise ValueError(
            "Existing manual prediction CSV contains rows outside the requested "
            f"five-condition/repeat design. Examples: {unexpected[:10]}"
        )

    expected_indexed = expected.set_index(key_columns)
    existing_indexed = existing.set_index(key_columns)
    structural_columns = {"Run_ID", "Row_Index"}
    for column in existing_indexed.columns:
        if column in structural_columns:
            continue
        if column not in expected_indexed.columns:
            expected_indexed[column] = pd.NA
        expected_indexed.loc[existing_indexed.index, column] = (
            existing_indexed[column].to_numpy()
        )

    merged = expected_indexed.reset_index()
    base_order = list(expected.columns)
    extra_columns = [
        column for column in merged.columns if column not in base_order
    ]
    merged = merged[base_order + extra_columns]
    preserved_nonempty = 0
    if "Raw_Prediction" in existing.columns:
        preserved_nonempty = int(
            existing["Raw_Prediction"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

    added_rows = len(expected_keys - existing_keys)
    if added_rows == 0 and encoding in {"utf-8-sig", "utf-8"}:
        return {
            "created": False,
            "encoding": encoding,
            "preserved_rows": len(existing),
            "preserved_nonempty_predictions": preserved_nonempty,
            "added_rows": 0,
            "backup_path": None,
        }
    backup_path = _available_backup_path(path)
    shutil.copy2(path, backup_path)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "created": False,
        "encoding": encoding,
        "preserved_rows": len(existing),
        "preserved_nonempty_predictions": preserved_nonempty,
        "added_rows": added_rows,
        "backup_path": backup_path,
    }


def write_manual_protocol(path: Path, *, prompt_path: Path, prediction_path: Path) -> None:
    text = f"""Manual o3 protocol for Reviewer Figure L4

1. Open {prompt_path.name}.
2. Complete each Run_ID in a separate new/temporary ChatGPT conversation.
3. Paste only the Prompt. Do not add the condition name, cluster IDs, ground
   truth, earlier answers, or other biological context.
4. Keep browsing, tools, and memory/context carry-over off when the interface
   permits. Do not revise a response because it looks biologically unexpected.
5. The response must contain exactly one non-empty label per input row, in the
   original order, without numbering. Copy those labels into Raw_Prediction for
   the matching Run_ID and Row_Index in {prediction_path.name}.
6. Record the model label displayed by ChatGPT in Model_ID, the date/time in
   Run_Date, and TRUE in Fresh_Chat. If a technical retry is necessary, record
   it in Notes and repeat the complete run in a new chat.
7. Save the CSV and rerun this script. Scoring starts only when all expected
   condition/repeat/cluster rows are present and all predictions are non-empty.

Reporting label:
"Manual o3 annotation using the GPTCelltype-style basic prompt."

Interpretation:
Standard DEG is capped at ranked top 10. COSG, Festem, MarkerMap, and
LLM-scCurator full_core retain their available cluster-specific list lengths.
This is a reviewer-requested descriptive analysis with unequal input budgets,
not a matched-budget performance benchmark and not execution of the
GPTCelltype software package.

If this file was extended from the earlier four-condition design, complete only
the newly added COSG Run_ID rows. Existing Standard_DEG, Festem, MarkerMap, and
LLM_scCurator_full_core predictions are preserved by key.
"""
    path.write_text(text, encoding="utf-8")


def load_noise_definitions() -> tuple[
    dict[str, re.Pattern[str]],
    dict[str, set[str]],
]:
    try:
        from llm_sc_curator.noise_lists import NOISE_LISTS, NOISE_PATTERNS
    except ImportError as exc:
        raise RuntimeError(
            "Could not import llm_sc_curator.noise_lists. Run from the project "
            "environment or use --skip-noise-diagnostic."
        ) from exc
    compiled = {
        str(name): re.compile(pattern)
        for name, pattern in NOISE_PATTERNS.items()
    }
    curated = {
        str(name): set(map(str, genes))
        for name, genes in NOISE_LISTS.items()
    }
    return compiled, curated


def classify_noise(
    gene: str,
    compiled_patterns: Mapping[str, re.Pattern[str]],
    curated_by_module: Mapping[str, set[str]],
) -> tuple[bool, bool, bool, str]:
    regex_modules = [
        name for name, pattern in compiled_patterns.items() if pattern.search(gene)
    ]
    curated_modules = [
        name for name, genes in curated_by_module.items() if gene in genes
    ]
    modules = list(dict.fromkeys(regex_modules + curated_modules))
    return (
        bool(regex_modules),
        bool(curated_modules),
        bool(modules),
        ";".join(modules),
    )


def build_input_diagnostics(
    marker_input: pd.DataFrame,
    *,
    skip_noise: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_level = marker_input.copy()
    if skip_noise:
        gene_level["Is_Regex_Noise"] = pd.NA
        gene_level["Is_Curated_Noise"] = pd.NA
        gene_level["Is_Biological_Noise"] = pd.NA
        gene_level["Noise_Modules"] = ""
    else:
        patterns, curated = load_noise_definitions()
        classifications = [
            classify_noise(gene, patterns, curated)
            for gene in gene_level["Gene"].astype(str)
        ]
        classified = pd.DataFrame(
            classifications,
            columns=[
                "Is_Regex_Noise",
                "Is_Curated_Noise",
                "Is_Biological_Noise",
                "Noise_Modules",
            ],
        )
        gene_level = pd.concat(
            [gene_level.reset_index(drop=True), classified],
            axis=1,
        )

    cluster_rows: list[dict[str, Any]] = []
    for (condition, cluster), sub in gene_level.groupby(
        ["Condition", "Cluster_ID"],
        sort=False,
    ):
        noise_count: int | float
        noise_fraction: float
        if skip_noise:
            noise_count = np.nan
            noise_fraction = np.nan
        else:
            noise_count = int(sub["Is_Biological_Noise"].astype(bool).sum())
            noise_fraction = float(noise_count / len(sub))
        cluster_rows.append(
            {
                "Condition": condition,
                "Cluster_ID": cluster,
                "Ground_Truth": sub["Ground_Truth"].iloc[0],
                "N_Genes": len(sub),
                "Biological_Noise_Count": noise_count,
                "Biological_Noise_Fraction": noise_fraction,
                "Budget_Matched_Across_Methods": False,
            }
        )
    return pd.DataFrame(cluster_rows), gene_level


def validate_predictions(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    repeats: int,
    model: str,
    *,
    manual_mode: bool,
) -> pd.DataFrame:
    required = ["Condition", "Repeat", "Cluster_ID", "Raw_Prediction"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    predictions = predictions.copy()
    predictions["Condition"] = predictions["Condition"].astype(str)
    predictions["Cluster_ID"] = predictions["Cluster_ID"].astype(str)
    predictions["Repeat"] = pd.to_numeric(
        predictions["Repeat"],
        errors="raise",
    ).astype(int)
    predictions["Raw_Prediction"] = (
        predictions["Raw_Prediction"].fillna("").astype(str).str.strip()
    )
    key_columns = ["Condition", "Repeat", "Cluster_ID"]
    duplicated = predictions.duplicated(key_columns, keep=False)
    if duplicated.any():
        bad = predictions.loc[duplicated, key_columns].head(10)
        raise ValueError(
            "Duplicate prediction rows found:\n"
            f"{bad.to_string(index=False)}"
        )

    expected = set(
        itertools.product(
            CONDITION_ORDER,
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
        missing_keys = sorted(expected - observed)[:10]
        extra_keys = sorted(observed - expected)[:10]
        raise ValueError(
            "Predictions do not match the expected five-condition design. "
            f"Missing examples={missing_keys}; extra examples={extra_keys}"
        )
    if predictions["Raw_Prediction"].eq("").any():
        bad = predictions.loc[
            predictions["Raw_Prediction"].eq(""),
            ["Condition", "Repeat", "Cluster_ID"],
        ].head(10)
        raise ValueError(
            "Empty annotator predictions found (first 10 examples):\n"
            f"{bad.to_string(index=False)}"
        )
    if "Model_ID" in predictions.columns:
        models = set(predictions["Model_ID"].dropna().astype(str))
        if models and models != {model}:
            raise ValueError(
                f"Prediction Model_ID values {models} do not match --model {model!r}."
            )
    if manual_mode and "Fresh_Chat" in predictions.columns:
        fresh = (
            predictions["Fresh_Chat"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "t", "yes", "y", "1"})
        )
        if not fresh.all():
            bad = predictions.loc[~fresh, key_columns].head(10)
            raise ValueError(
                "Manual runs must confirm Fresh_Chat=TRUE for every row. "
                "Examples:\n"
                f"{bad.to_string(index=False)}"
            )
    return predictions


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
            "Could not import the frozen manuscript scorer. Run this script from "
            "the repository root with benchmarks/ on PYTHONPATH. S_anno is not "
            "approximated in this reviewer-only comparison."
        ) from exc


def score_predictions(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    cfg, score_fn, parse_major, parse_state, expected_state = load_project_scorer()
    ground_truth = audit[["Cluster_ID", "Ground_Truth"]].copy()
    scored = predictions.merge(
        ground_truth,
        on="Cluster_ID",
        how="left",
        validate="many_to_one",
    )

    scores: list[float] = []
    pred_states: list[str] = []
    pred_majors: list[str] = []
    gt_states: list[str] = []
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
        pred_majors.append(pred_major)
        gt_states.append(gt_state)
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
    boot = array[indices].mean(axis=1)
    return (
        mean,
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    summary_rows: list[dict[str, Any]] = []
    for index, condition in enumerate(CONDITION_ORDER):
        subset = per_cluster[per_cluster["Condition"] == condition]
        mean, low, high = bootstrap_mean_ci(
            subset["Mean_Sanno"].to_numpy(),
            n_bootstrap,
            seed + index,
        )
        summary_rows.append(
            {
                "Condition": condition,
                "N_Clusters": len(subset),
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
            }
        )
    summary = pd.DataFrame(summary_rows)

    wide = per_cluster.pivot(
        index="Cluster_ID",
        columns="Condition",
        values="Mean_Sanno",
    )
    contrast_pairs = [
        ("COSG", "Standard_DEG"),
        ("Festem", "Standard_DEG"),
        ("MarkerMap", "Standard_DEG"),
        ("LLM_scCurator_full_core", "Standard_DEG"),
        ("LLM_scCurator_full_core", "COSG"),
        ("LLM_scCurator_full_core", "Festem"),
        ("LLM_scCurator_full_core", "MarkerMap"),
    ]
    paired_rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(contrast_pairs):
        paired = wide[[left, right]].dropna()
        differences = (
            paired[left] - paired[right]
        ).to_numpy(dtype=float)
        mean, low, high = bootstrap_mean_ci(
            differences,
            n_bootstrap,
            seed + 100 + index,
        )
        try:
            from scipy.stats import wilcoxon

            if np.allclose(differences, 0):
                wilcoxon_p = 1.0
            else:
                wilcoxon_p = float(
                    wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
        except (ImportError, ValueError):
            wilcoxon_p = np.nan
        paired_rows.append(
            {
                "Contrast": f"{left}_minus_{right}",
                "N_Paired_Clusters": len(differences),
                "Mean_Paired_Difference_Sanno": mean,
                "Bootstrap_95CI_Low": low,
                "Bootstrap_95CI_High": high,
                "Paired_Wilcoxon_P": wilcoxon_p,
                "Two_Sided_Sign_Flip_P": sign_flip_pvalue(
                    differences,
                    seed=seed + 200 + index,
                ),
                "N_Improved": int((differences > 0).sum()),
                "N_Unchanged": int(np.isclose(differences, 0).sum()),
                "N_Worsened": int((differences < 0).sum()),
                "Interpretation": (
                    "descriptive native-length contrast; marker budget not matched"
                ),
            }
        )
    return per_cluster, summary, pd.DataFrame(paired_rows)


def _strip_offsets(n_points: int, width: float = 0.14) -> np.ndarray:
    if n_points <= 1:
        return np.zeros(n_points)
    return np.linspace(-width, width, n_points)


def _box_and_strip(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    *,
    ylabel: str,
) -> None:
    values = [
        frame.loc[frame["Condition"] == condition, value_column]
        .dropna()
        .to_numpy(dtype=float)
        for condition in CONDITION_ORDER
    ]
    boxes = axis.boxplot(
        values,
        positions=np.arange(len(CONDITION_ORDER)),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, condition in zip(boxes["boxes"], CONDITION_ORDER):
        patch.set_facecolor(CONDITION_COLORS[condition])
        patch.set_alpha(0.22)
        patch.set_edgecolor(CONDITION_COLORS[condition])
    for index, (condition, condition_values) in enumerate(
        zip(CONDITION_ORDER, values)
    ):
        ordered = np.sort(condition_values)
        axis.scatter(
            index + _strip_offsets(len(ordered)),
            ordered,
            s=22,
            color=CONDITION_COLORS[condition],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )
        if len(ordered):
            axis.scatter(
                [index],
                [float(np.mean(ordered))],
                marker="D",
                s=34,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=4,
            )
    axis.set_xticks(
        np.arange(len(CONDITION_ORDER)),
        [CONDITION_LABELS[condition] for condition in CONDITION_ORDER],
    )
    axis.set_ylabel(ylabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _panel_a(
    axis: plt.Axes,
    diagnostics: pd.DataFrame,
) -> None:
    _box_and_strip(
        axis,
        diagnostics,
        "N_Genes",
        ylabel="Genes supplied per cluster",
    )
    axis.set_title(
        "A  Actual input gene count",
        loc="left",
        fontweight="bold",
    )


def _panel_b(
    axis: plt.Axes,
    diagnostics: pd.DataFrame,
    *,
    noise_available: bool,
) -> None:
    if noise_available:
        _box_and_strip(
            axis,
            diagnostics,
            "Biological_Noise_Fraction",
            ylabel="Biological-noise fraction",
        )
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(
            "B  Biological-noise fraction",
            loc="left",
            fontweight="bold",
        )
    else:
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "Biological-noise diagnostic skipped",
            ha="center",
            va="center",
        )


def _panel_c(
    axis: plt.Axes,
    per_cluster: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    wide = per_cluster.pivot(
        index="Cluster_ID",
        columns="Condition",
        values="Mean_Sanno",
    ).dropna()

    x_positions = np.arange(len(CONDITION_ORDER), dtype=float)

    for _, row in wide.iterrows():
        y = row[CONDITION_ORDER].to_numpy(dtype=float)
        axis.plot(
            x_positions,
            y,
            color="#B8B8B8",
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )
        axis.scatter(
            x_positions,
            y,
            c=[
                CONDITION_COLORS[condition]
                for condition in CONDITION_ORDER
            ],
            s=20,
            zorder=2,
        )

    for index, condition in enumerate(CONDITION_ORDER):
        row = summary.loc[
            summary["Condition"] == condition
        ].iloc[0]

        mean = float(row["Mean_Sanno"])
        low = float(row["Mean_Sanno_Bootstrap_CI_Low"])
        high = float(row["Mean_Sanno_Bootstrap_CI_High"])

        axis.errorbar(
            index,
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            fmt="D",
            markersize=5.5,
            color="black",
            markerfacecolor="white",
            capsize=3,
            linewidth=1.1,
            zorder=4,
        )

    axis.set_xticks(
        x_positions,
        [
            CONDITION_LABELS[condition]
            for condition in CONDITION_ORDER
        ],
    )
    axis.set_ylabel(r"$S_{anno}$")
    axis.set_ylim(-0.04, 1.08)
    axis.set_title(
        "C  Cluster-level annotation scores",
        loc="left",
        fontweight="bold",
    )


def _panel_d(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    legend_outside_right: bool = True,
) -> None:
    metrics = [
        ("Mean_Sanno", r"Mean $S_{anno}$"),
        ("Exact_State_Agreement", "Exact-state\nagreement"),
        ("Major_Lineage_Accuracy", "Major-lineage\naccuracy"),
        (
            "Hierarchy_Consistent_Accuracy",
            "Hierarchy-\nconsistent",
        ),
    ]

    metric_positions = np.arange(len(metrics))
    width = 0.16
    offsets = (
        np.arange(len(CONDITION_ORDER))
        - (len(CONDITION_ORDER) - 1) / 2
    ) * width

    for index, condition in enumerate(CONDITION_ORDER):
        row = summary.loc[
            summary["Condition"] == condition
        ].iloc[0]

        values = [
            100 * float(row[column])
            for column, _ in metrics
        ]

        axis.bar(
            metric_positions + offsets[index],
            values,
            width=width,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition].replace("\n", " "),
        )

    axis.set_xticks(
        metric_positions,
        [label for _, label in metrics],
    )
    axis.set_ylabel("Score or accuracy (%)")
    axis.set_ylim(0, 108)
    axis.set_title(
        "D  Complementary annotation metrics",
        loc="left",
        fontweight="bold",
    )

    if legend_outside_right:
        axis.legend(
            frameon=False,
            fontsize=7.5,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
            borderaxespad=0,
            handlelength=1.5,
            handletextpad=0.6,
            labelspacing=0.7,
        )
    else:
        axis.legend(
            frameon=False,
            fontsize=7.5,
            ncol=1,
            loc="upper left",
        )


def _save_individual_panel(
    draw_panel: Callable[[plt.Axes], None],
    *,
    panel_letter: str,
    out_pdf: Path,
    out_png: Path,
    figsize: tuple[float, float] = (5.4, 4.2),
    left_margin: float = 0.16,
    right_margin: float = 0.97,
) -> None:
    """
    Draw and save one panel as independent PDF and PNG files.

    Files are written beside the combined figure with suffixes such as
    "_A.pdf" and "_A.png".
    """
    figure, axis = plt.subplots(figsize=figsize)
    draw_panel(axis)

    figure.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=0.18,
        top=0.90,
    )

    panel_pdf = out_pdf.with_name(
        f"{out_pdf.stem}_{panel_letter}{out_pdf.suffix}"
    )
    panel_png = out_png.with_name(
        f"{out_png.stem}_{panel_letter}{out_png.suffix}"
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
    diagnostics: pd.DataFrame,
    per_cluster: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    out_pdf: Path,
    out_png: Path,
    noise_available: bool,
    annotator_label: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # ------------------------------------------------------------
    # Combined 2 x 2 figure
    # ------------------------------------------------------------
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 7.2),
    )

    _panel_a(
        axes[0, 0],
        diagnostics,
    )
    _panel_b(
        axes[0, 1],
        diagnostics,
        noise_available=noise_available,
    )
    _panel_c(
        axes[1, 0],
        per_cluster,
        summary,
    )
    _panel_d(
        axes[1, 1],
        summary,
        legend_outside_right=True,
    )

    figure.suptitle(
        (
            "Reviewer-only external marker-list comparison "
            f"using one {annotator_label} annotator"
        ),
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        (
            "Standard DEG uses ranked top 10; COSG, Festem, MarkerMap, "
            "and full_core retain available list lengths. "
            "Budgets were not matched."
        ),
        ha="center",
        fontsize=7.5,
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.92,
        wspace=0.55,
        hspace=0.38,
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
    # Individual panel figures
    # ------------------------------------------------------------
    _save_individual_panel(
        lambda axis: _panel_a(axis, diagnostics),
        panel_letter="A",
        out_pdf=out_pdf,
        out_png=out_png,
    )

    _save_individual_panel(
        lambda axis: _panel_b(
            axis,
            diagnostics,
            noise_available=noise_available,
        ),
        panel_letter="B",
        out_pdf=out_pdf,
        out_png=out_png,
    )

    _save_individual_panel(
        lambda axis: _panel_c(
            axis,
            per_cluster,
            summary,
        ),
        panel_letter="C",
        out_pdf=out_pdf,
        out_png=out_png,
    )

    _save_individual_panel(
        lambda axis: _panel_d(
            axis,
            summary,
            legend_outside_right=True,
        ),
        panel_letter="D",
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(6.6, 4.2),
        left_margin=0.34,
    )

def run_r_helper(
    *,
    rscript: str,
    helper_path: Path,
    marker_path: Path,
    prediction_path: Path,
    r_metadata_path: Path,
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
            "OPENAI_API_KEY is not set. The key is read only from the environment."
        )
    command = [
        rscript,
        str(helper_path),
        str(marker_path),
        str(prediction_path),
        str(r_metadata_path),
        str(prompt_path),
        model,
        tissue,
        str(repeats),
        str(inter_call_sleep_sec),
        str(max_api_retries),
        str(retry_base_wait_sec),
    ]
    subprocess.run(command, check=True, timeout=timeout_sec)


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    marker_table: Path,
    audit: pd.DataFrame,
    diagnostics: pd.DataFrame,
    status: str,
    conversions: Mapping[str, str],
    cosg_metadata: Mapping[str, str],
    r_metadata_path: Path,
) -> None:
    metadata: dict[str, Any] = {
        "Analysis_ID": ANALYSIS_ID,
        "Script_Build": SCRIPT_BUILD,
        "Status": status,
        "Timestamp": now_iso(),
        "Marker_Table": str(marker_table.resolve()),
        "Marker_Table_SHA256": sha256_file(marker_table),
        "Marker_Sheet": args.marker_sheet,
        "Audit_Sheet": args.audit_sheet,
        "Dataset_Filter": args.dataset,
        "N_Clusters": len(audit),
        "Conditions": ";".join(CONDITION_ORDER),
        "Comparison": (
            "Shared annotator and GPTCelltype-style basic prompt; Standard DEG "
            "capped at ranked top 10; other lists retain available native "
            "lengths; no cross-method budget matching"
        ),
        "Interpretation_Boundary": (
            "reviewer-requested descriptive comparison; unequal marker budgets "
            "and different method objectives preclude a fully matched benchmark"
        ),
        "Standard_Variant": args.standard_variant,
        "LLM_scCurator_Variant": args.llmsc_variant,
        "COSG_Conversion": conversions["COSG"],
        "Festem_File": str(Path(args.festem_csv).resolve()),
        "Festem_File_SHA256": sha256_file(Path(args.festem_csv)),
        "Festem_Conversion": conversions["Festem"],
        "MarkerMap_File": str(Path(args.markermap_csv).resolve()),
        "MarkerMap_File_SHA256": sha256_file(Path(args.markermap_csv)),
        "MarkerMap_Conversion": conversions["MarkerMap"],
        "Global_Panel_Policy": args.global_panel_policy,
        "Standard_Max_Genes": args.standard_max_genes,
        "LLM_scCurator_Max_Genes": args.llmsc_max_genes,
        "External_Max_Genes": args.external_max_genes,
        "Minimum_Genes": args.minimum_genes,
        "Noise_Diagnostic_Skipped": args.skip_noise_diagnostic,
        "Annotation_Mode": (
            "native_GPTCelltype_API" if args.run_api else "manual_web_interface"
        ),
        "Annotator_Reporting_Label": (
            "Native GPTCelltype package"
            if args.run_api
            else "Manual o3 annotation using the GPTCelltype-style basic prompt"
        ),
        "Model_ID_or_Displayed_Label": args.model,
        "Tissue_Name": args.tissue,
        "Repeats": args.repeats,
        "Repeat_Aggregation": (
            "cluster-level S_anno and complementary metrics are averaged across "
            "independent repeats; modal raw prediction is retained descriptively"
        ),
        "Expected_Independent_Runs": len(CONDITION_ORDER) * args.repeats,
        "Expected_API_Calls": (
            len(CONDITION_ORDER) * args.repeats if args.run_api else 0
        ),
        "Fresh_Chat_Per_Manual_Run_Required": not args.run_api,
        "Inter_Call_Sleep_sec": args.inter_call_sleep_sec,
        "Max_API_Retries": args.max_api_retries,
        "Retry_Base_Wait_sec": args.retry_base_wait_sec,
        "Sanno_Scorer": (
            "benchmarks.hierarchical_scoring.score_hierarchical "
            "with CD8_HIER_CFG"
        ),
        "GPTCelltype_Repository": "https://github.com/Winnie09/GPTCelltype",
        "COSG_Repository": "https://github.com/genecell/COSG",
        "Festem_Repository": "https://github.com/XiDsLab/Festem",
        "MarkerMap_Repository": (
            "https://github.com/Computational-Morphogenomics-Group/MarkerMap"
        ),
        "Python": sys.version.replace("\n", " "),
        "Python_Executable": sys.executable,
        "Platform": platform.platform(),
        "Hostname": socket.gethostname(),
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "matplotlib_version": package_version("matplotlib"),
        "llm_sc_curator_version": package_version("llm-sc-curator"),
        "API_Key_Recorded": "No",
    }
    for key, value in cosg_metadata.items():
        metadata[f"COSG_{key}"] = value
    for condition in CONDITION_ORDER:
        values = diagnostics.loc[
            diagnostics["Condition"] == condition,
            "N_Genes",
        ].astype(int)
        metadata[f"{condition}_N_Genes_Min"] = int(values.min())
        metadata[f"{condition}_N_Genes_Median"] = float(values.median())
        metadata[f"{condition}_N_Genes_Max"] = int(values.max())
    if r_metadata_path.exists():
        r_metadata = pd.read_csv(r_metadata_path)
        for _, row in r_metadata.iterrows():
            metadata[f"R_{row['Key']}"] = row["Value"]
    pd.DataFrame(
        {"Key": list(metadata), "Value": list(metadata.values())}
    ).to_csv(path, index=False)


def main() -> None:
    print(f"Script build: {SCRIPT_BUILD}")
    args = parse_args()
    marker_table = Path(args.marker_table)
    cosg_path = Path(args.cosg_csv) if args.cosg_csv else None
    input_h5ad = Path(args.input_h5ad) if args.input_h5ad else None
    festem_path = Path(args.festem_csv)
    markermap_path = Path(args.markermap_csv)
    outdir = Path(args.outdir)
    figure_dir = Path(args.figure_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    marker_input_path = outdir / "L4_external_marker_native_inputs.csv"
    length_path = outdir / "L4_external_marker_input_lengths.csv"
    diagnostic_path = outdir / "L4_external_marker_input_diagnostics.csv"
    gene_diagnostic_path = outdir / "L4_external_marker_gene_diagnostics.csv"
    prompt_path = outdir / "L4_shared_annotator_prompt_manifest.csv"
    manual_prediction_path = outdir / "L4_manual_o3_predictions.csv"
    manual_protocol_path = outdir / "L4_manual_o3_protocol.txt"
    api_prediction_path = outdir / "L4_GPTCelltype_API_predictions.csv"
    prediction_path = (
        api_prediction_path if args.run_api else manual_prediction_path
    )
    scored_path = outdir / "L4_shared_annotator_predictions_scored.csv"
    per_cluster_path = outdir / "L4_shared_annotator_per_cluster_metrics.csv"
    summary_path = outdir / "L4_shared_annotator_summary.csv"
    paired_path = outdir / "L4_shared_annotator_paired_comparisons.csv"
    metadata_path = outdir / "L4_external_marker_run_metadata.csv"
    r_metadata_path = outdir / "L4_GPTCelltype_R_metadata.csv"
    helper_path = outdir / "gptcelltype_native_runner.R"
    out_pdf = figure_dir / f"{args.figure_stem}.pdf"
    out_png = figure_dir / f"{args.figure_stem}.png"

    audit = load_cd8_audit(marker_table, args.audit_sheet)
    cluster_order = audit["Cluster_ID"].astype(str).tolist()
    standard, curated = read_existing_marker_lists(
        marker_table,
        args.marker_sheet,
        args.dataset,
        args.standard_variant,
        args.llmsc_variant,
    )
    standard = {
        cluster: _limit_genes(standard[cluster], args.standard_max_genes)
        for cluster in cluster_order
    }
    curated = {
        cluster: _limit_genes(curated[cluster], args.llmsc_max_genes)
        for cluster in cluster_order
    }

    if cosg_path is not None:
        cosg_lists, cosg_conversion = read_external_marker_lists(
            cosg_path,
            method="COSG",
            reference_lists=standard,
            cluster_order=cluster_order,
            global_panel_policy=args.global_panel_policy,
            method_filter="COSG",
        )
        cosg_metadata: dict[str, str] = {
            "Source_Mode": "precomputed_csv",
            "CSV_File": str(cosg_path.resolve()),
            "CSV_File_SHA256": sha256_file(cosg_path),
        }
        short_cosg = {
            cluster: len(cosg_lists[cluster])
            for cluster in cluster_order
            if len(cosg_lists[cluster]) < args.cosg_n_genes
        }
        if short_cosg and cosg_path.name == "L4_external_marker_noise_gene_level.csv":
            warnings.warn(
                "The prior gene-level L4 output contains matched-budget COSG "
                f"rows rather than all native top-{args.cosg_n_genes} rows for "
                f"some clusters: {short_cosg}. For native top-{args.cosg_n_genes}, "
                "use --input-h5ad instead.",
                stacklevel=2,
            )
    else:
        assert input_h5ad is not None
        cosg_lists, cosg_metadata = run_cosg(
            input_h5ad,
            group_col=args.group_col,
            evaluated_clusters=cluster_order,
            n_top=args.cosg_n_genes,
            key_added=args.cosg_key,
            mu=args.cosg_mu,
            expressed_pct=args.cosg_expressed_pct,
            remove_lowly_expressed=args.cosg_remove_lowly_expressed,
            use_raw=args.cosg_use_raw,
            layer=args.cosg_layer,
        )
        cosg_conversion = (
            "native cluster-specific ranked COSG lists reproduced from AnnData; "
            "no cross-method length matching"
        )

    festem, festem_conversion = read_external_marker_lists(
        festem_path,
        method="Festem",
        reference_lists=standard,
        cluster_order=cluster_order,
        global_panel_policy=args.global_panel_policy,
    )
    markermap, markermap_conversion = read_external_marker_lists(
        markermap_path,
        method="MarkerMap",
        reference_lists=standard,
        cluster_order=cluster_order,
        global_panel_policy=args.global_panel_policy,
    )
    festem = {
        cluster: _limit_genes(festem[cluster], args.external_max_genes)
        for cluster in cluster_order
    }
    markermap = {
        cluster: _limit_genes(markermap[cluster], args.external_max_genes)
        for cluster in cluster_order
    }
    cosg_lists = {
        cluster: _limit_genes(cosg_lists[cluster], args.external_max_genes)
        for cluster in cluster_order
    }
    method_lists = {
        "Standard_DEG": standard,
        "COSG": cosg_lists,
        "Festem": festem,
        "MarkerMap": markermap,
        "LLM_scCurator_full_core": curated,
    }
    ensure_cluster_alignment(
        method_lists,
        cluster_order,
        args.minimum_genes,
    )

    marker_input, lengths = build_native_length_inputs(audit, method_lists)
    diagnostics, gene_diagnostics = build_input_diagnostics(
        marker_input,
        skip_noise=args.skip_noise_diagnostic,
    )
    marker_input.to_csv(marker_input_path, index=False)
    lengths.to_csv(length_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)
    gene_diagnostics.to_csv(gene_diagnostic_path, index=False)
    write_prepared_prompts(
        prompt_path,
        marker_input,
        args.model,
        args.tissue,
        args.repeats,
    )
    template_status: dict[str, Any] | None = None
    if args.run_api:
        helper_path.write_text(R_RUNNER, encoding="utf-8")
    else:
        template_status = update_manual_prediction_template(
            manual_prediction_path,
            audit,
            model=args.model,
            repeats=args.repeats,
        )
        write_manual_protocol(
            manual_protocol_path,
            prompt_path=prompt_path,
            prediction_path=manual_prediction_path,
        )

    conversions = {
        "COSG": cosg_conversion,
        "Festem": festem_conversion,
        "MarkerMap": markermap_conversion,
    }
    print("Prepared reviewer-only five-condition shared-annotator inputs.")
    print(f"  Clusters: {len(audit)}")
    for condition in CONDITION_ORDER:
        values = diagnostics.loc[
            diagnostics["Condition"] == condition,
            "N_Genes",
        ]
        print(
            f"  {condition}: genes/cluster "
            f"{int(values.min())}–{int(values.max())} "
            f"(median {values.median():.1f})"
        )
    print(
        f"  Independent runs at repeats={args.repeats}: "
        f"{len(CONDITION_ORDER) * args.repeats}"
    )
    print(f"  Marker input: {marker_input_path}")
    print(f"  Prompt manifest: {prompt_path}")
    if not args.run_api:
        print(f"  Manual protocol: {manual_protocol_path}")
        print(f"  Manual predictions: {manual_prediction_path}")
        assert template_status is not None
        if template_status["created"]:
            print("  A blank manual prediction template was created.")
        else:
            print(
                "  Existing prediction rows preserved: "
                f"{template_status['preserved_rows']} "
                f"({template_status['preserved_nonempty_predictions']} non-empty)"
            )
            print(
                "  Newly added blank rows: "
                f"{template_status['added_rows']}"
            )
            print(
                "  Existing CSV decoded as: "
                f"{template_status['encoding']}"
            )
            if template_status["backup_path"] is not None:
                print(
                    "  Original prediction CSV backup: "
                    f"{template_status['backup_path']}"
                )

    if args.run_api:
        run_needed = True
        if prediction_path.exists() and not args.force_api:
            try:
                validate_predictions(
                    pd.read_csv(prediction_path),
                    audit,
                    args.repeats,
                    args.model,
                    manual_mode=False,
                )
                run_needed = False
                print(
                    "Complete prediction checkpoint found; reusing: "
                    f"{prediction_path}"
                )
            except (
                ValueError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                print(
                    "Partial prediction checkpoint found; resuming: "
                    f"{prediction_path}"
                )
                print(f"  Checkpoint status: {exc}")
        if prediction_path.exists() and args.force_api:
            prediction_path.unlink()
        if run_needed:
            print(f"Running native GPTCelltype with model={args.model!r}.")
            run_r_helper(
                rscript=args.rscript,
                helper_path=helper_path,
                marker_path=marker_input_path,
                prediction_path=prediction_path,
                r_metadata_path=r_metadata_path,
                prompt_path=prompt_path,
                model=args.model,
                tissue=args.tissue,
                repeats=args.repeats,
                inter_call_sleep_sec=args.inter_call_sleep_sec,
                max_api_retries=args.max_api_retries,
                retry_base_wait_sec=args.retry_base_wait_sec,
                timeout_sec=args.api_timeout_sec,
            )

    if not prediction_path.exists():
        write_metadata(
            metadata_path,
            args=args,
            marker_table=marker_table,
            audit=audit,
            diagnostics=diagnostics,
            status="prepared_no_predictions",
            conversions=conversions,
            cosg_metadata=cosg_metadata,
            r_metadata_path=r_metadata_path,
        )
        print("No prediction file is available yet.")
        return

    try:
        prediction_frame = (
            pd.read_csv(prediction_path)
            if args.run_api
            else read_csv_with_encoding_fallback(prediction_path)[0]
        )
        predictions = validate_predictions(
            prediction_frame,
            audit,
            args.repeats,
            args.model,
            manual_mode=not args.run_api,
        )
    except (
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        if args.run_api:
            raise
        write_metadata(
            metadata_path,
            args=args,
            marker_table=marker_table,
            audit=audit,
            diagnostics=diagnostics,
            status="awaiting_complete_manual_predictions",
            conversions=conversions,
            cosg_metadata=cosg_metadata,
            r_metadata_path=r_metadata_path,
        )
        print("Manual prediction template is not complete; no scoring was run.")
        print(f"  Validation status: {exc}")
        print(f"  Complete all rows in: {manual_prediction_path}")
        return
    scored = score_predictions(predictions, audit)
    per_cluster, summary, paired = summarize_predictions(
        scored,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    scored.to_csv(scored_path, index=False)
    per_cluster.to_csv(per_cluster_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    make_figure(
        diagnostics,
        per_cluster,
        summary,
        out_pdf=out_pdf,
        out_png=out_png,
        noise_available=not args.skip_noise_diagnostic,
        annotator_label=(
            "GPTCelltype" if args.run_api else f"manual {args.model}"
        ),
    )
    write_metadata(
        metadata_path,
        args=args,
        marker_table=marker_table,
        audit=audit,
        diagnostics=diagnostics,
        status="completed",
        conversions=conversions,
        cosg_metadata=cosg_metadata,
        r_metadata_path=r_metadata_path,
    )

    print("Reviewer Figure L4 external marker comparison completed.")
    for _, row in summary.iterrows():
        print(
            f"  {row['Condition']}: "
            f"mean S_anno={100 * row['Mean_Sanno']:.1f}%, "
            f"exact state={100 * row['Exact_State_Agreement']:.1f}%, "
            f"major lineage={100 * row['Major_Lineage_Accuracy']:.1f}%"
        )
    print(f"  Figure: {out_pdf}")
    print(f"  Figure: {out_png}")


if __name__ == "__main__":
    main()
