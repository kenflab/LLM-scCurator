#!/usr/bin/env python3
"""Prepare Festem and MarkerMap inputs for reviewer-only Figure L4.

This script reads the same CD8 AnnData object used by the benchmark and writes
the two cluster-specific CSV files consumed by
``make_l4_external_marker_noise.py``:

* ``festem_cd8_markers.csv``
* ``markermap_cd8_markers.csv``

Selection and cluster allocation
--------------------------------
Festem is run on the raw-count layer with the existing cluster labels supplied
as its prior. Its FDR-selected ``clustering_features`` are then allocated to
clusters with Festem::AllocateMarker on the log-normalized layer. Genes assigned
grade A or B are retained, as recommended by the Festem documentation.
Allocation is performed gene by gene through the same Festem Scott-Knott core,
so a gene-specific numerical failure is recorded and excluded instead of
terminating the full run. Genes assigned grade A in every cluster are also
excluded because they are not cluster-discriminating.

Supervised MarkerMap is run on the same cells, log-normalized expression
matrix, and cluster labels. MarkerMap natively selects one global panel rather
than a separate list for every cluster. To make that panel usable as 17
cluster-level LLM input rows, the selected genes are passed through the same
Festem::AllocateMarker A/B allocation rule. This post-selection conversion is
recorded explicitly in the output and metadata; it must not be described as a
native cluster-specific MarkerMap output.

The two selected panels are not truncated to equal lengths. Figure L4 reports
the actual number of genes supplied per cluster.

The global MarkerMap and Festem panels are cached in ``--outdir`` before
cluster allocation. A rerun reuses validated caches by default; pass
``--force-marker-selection`` to repeat both selection procedures.

Example
-------
Run from the R1 directory::

    python paper/revision_metrics/prepare_l4_festem_markermap.py \
      --input-h5ad cd8_benchmark_data.h5ad \
      --outdir paper/revision_tables

Then run Figure L4::

    python paper/revision_metrics/make_l4_external_marker_noise.py \
      --marker-table LetterTables.xlsx \
      --festem-csv paper/revision_tables/festem_cd8_markers.csv \
      --markermap-csv paper/revision_tables/markermap_cd8_markers.csv

Requirements
------------
Python: anndata, numpy, pandas, scipy, torch, markermap
R: Festem (including its dependencies), Matrix

The script uses a temporary local work directory for Matrix Market transfers
and deletes those intermediate files after successful or failed execution.
Only the small final CSV and metadata outputs are retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ANALYSIS_ID = "L4_Festem_MarkerMap_preparation_v2"
FESTEM_REPOSITORY = "https://github.com/XiDsLab/Festem"
MARKERMAP_REPOSITORY = (
    "https://github.com/Computational-Morphogenomics-Group/MarkerMap"
)


R_RUNNER = r'''args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 15) {
  stop(paste(
    "Expected 15 arguments: counts_mtx logcounts_mtx genes_txt labels_txt",
    "markermap_global_csv festem_global_csv festem_out_csv",
    "markermap_out_csv r_metadata_csv festem_fdr allocation_fdr",
    "festem_seed num_threads block_size reuse_festem_global"
  ))
}

counts_mtx <- args[[1]]
logcounts_mtx <- args[[2]]
genes_txt <- args[[3]]
labels_txt <- args[[4]]
markermap_global_csv <- args[[5]]
festem_global_csv <- args[[6]]
festem_out_csv <- args[[7]]
markermap_out_csv <- args[[8]]
r_metadata_csv <- args[[9]]
festem_fdr <- as.numeric(args[[10]])
allocation_fdr <- as.numeric(args[[11]])
festem_seed <- as.integer(args[[12]])
num_threads <- as.integer(args[[13]])
block_size <- as.integer(args[[14]])
reuse_festem_global <- tolower(args[[15]]) %in% c("true", "t", "1", "yes")

if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("R package Matrix is not installed.")
}
if (!requireNamespace("Festem", quietly = TRUE)) {
  stop("R package Festem is not installed.")
}

genes <- readLines(genes_txt, warn = FALSE, encoding = "UTF-8")
labels_raw <- readLines(labels_txt, warn = FALSE, encoding = "UTF-8")
labels <- factor(labels_raw, levels = unique(labels_raw))

counts <- methods::as(Matrix::readMM(counts_mtx), "dgCMatrix")
logcounts <- methods::as(Matrix::readMM(logcounts_mtx), "dgCMatrix")

if (nrow(counts) != length(genes) || nrow(logcounts) != length(genes)) {
  stop("Gene-name count does not match matrix rows.")
}
if (ncol(counts) != length(labels) || ncol(logcounts) != length(labels)) {
  stop("Cluster-label count does not match matrix columns.")
}
rownames(counts) <- genes
rownames(logcounts) <- genes

if (reuse_festem_global && file.exists(festem_global_csv)) {
  message(paste("Reusing cached Festem global panel:", festem_global_csv))
  festem_global <- read.csv(
    festem_global_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  required_festem <- c("gene", "selection_rank")
  if (!all(required_festem %in% colnames(festem_global))) {
    stop(paste(
      "Cached Festem global CSV is missing:",
      paste(
        setdiff(required_festem, colnames(festem_global)),
        collapse = ", "
      )
    ))
  }
  festem_global <- festem_global[
    order(festem_global$selection_rank),
    ,
    drop = FALSE
  ]
  festem_selected <- unique(as.character(festem_global$gene))
  festem_selected <- festem_selected[
    !is.na(festem_selected) & nzchar(festem_selected)
  ]
} else {
  message(sprintf(
    "Running Festem on %d genes, %d cells, %d clusters.",
    nrow(counts), ncol(counts), nlevels(labels)
  ))
  festem_result <- Festem::RunFestem(
    counts,
    G = nlevels(labels),
    prior = labels,
    FDR_level = festem_fdr,
    seed = festem_seed,
    num.threads = num_threads,
    block_size = block_size
  )

  festem_ranked <- unique(as.character(festem_result[["generank"]]))
  festem_selected <- unique(
    as.character(festem_result[["clustering_features"]])
  )
  festem_selected <- festem_selected[
    !is.na(festem_selected) & nzchar(festem_selected)
  ]
  festem_selected <- festem_ranked[festem_ranked %in% festem_selected]
  if (length(festem_selected) == 0) {
    stop("Festem returned no FDR-selected clustering_features.")
  }
  write.csv(
    data.frame(
      gene = festem_selected,
      selection_rank = seq_along(festem_selected),
      method = "Festem",
      stringsAsFactors = FALSE
    ),
    festem_global_csv,
    row.names = FALSE,
    na = ""
  )
  message(paste("Cached Festem global panel:", festem_global_csv))
}

markermap_global <- read.csv(
  markermap_global_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
required_mm <- c("gene", "selection_rank")
if (!all(required_mm %in% colnames(markermap_global))) {
  stop(paste(
    "MarkerMap global CSV is missing:",
    paste(setdiff(required_mm, colnames(markermap_global)), collapse = ", ")
  ))
}
markermap_global <- markermap_global[
  order(markermap_global$selection_rank),
  ,
  drop = FALSE
]
markermap_selected <- unique(as.character(markermap_global$gene))
markermap_selected <- markermap_selected[
  !is.na(markermap_selected) & nzchar(markermap_selected)
]

safe_marker_allocate_core <- function(norm_data, type, sig_level) {
  empty_result <- stats::setNames(
    rep(NA_character_, nlevels(type)),
    levels(type)
  )
  tryCatch(
    {
      result <- Festem:::marker_allocate_core(
        as.numeric(norm_data),
        type = type,
        sig.level = sig_level
      )
      formatted <- empty_result
      result_names <- names(result)
      result <- as.character(result)
      names(result) <- result_names
      if (!is.null(names(result))) {
        formatted[names(result)] <- result
      } else if (length(result) == length(formatted)) {
        formatted[] <- result
      }
      formatted
    },
    error = function(e) empty_result
  )
}

allocate_panel <- function(selected_genes, method_name, output_csv) {
  selected_genes <- selected_genes[selected_genes %in% genes]
  if (length(selected_genes) == 0) {
    stop(paste(method_name, "has no selected genes present in the expression matrix."))
  }

  message(sprintf(
    "Allocating %s panel (%d selected genes) to clusters.",
    method_name,
    length(selected_genes)
  ))
  selected_expression <- logcounts[selected_genes, , drop = FALSE]
  allocation_cluster <- parallel::makeCluster(
    getOption("cl.cores", num_threads)
  )
  allocation <- tryCatch(
    {
      parallel::clusterEvalQ(
        allocation_cluster,
        requireNamespace("Festem", quietly = TRUE)
      )
      if (requireNamespace("pbapply", quietly = TRUE)) {
        pbapply::pbapply(
          selected_expression,
          1,
          safe_marker_allocate_core,
          type = labels,
          sig_level = allocation_fdr,
          cl = allocation_cluster
        )
      } else {
        parallel::parApply(
          allocation_cluster,
          selected_expression,
          1,
          safe_marker_allocate_core,
          type = labels,
          sig_level = allocation_fdr
        )
      }
    },
    finally = parallel::stopCluster(allocation_cluster)
  )
  allocation <- t(as.matrix(allocation))
  rownames(allocation) <- selected_genes
  colnames(allocation) <- levels(labels)

  failed_flag <- apply(allocation, 1, function(x) any(is.na(x)))
  failed_genes <- rownames(allocation)[failed_flag]
  allocation <- allocation[!failed_flag, , drop = FALSE]

  all_a_flag <- apply(
    allocation,
    1,
    function(x) all(!is.na(x)) && all(toupper(as.character(x)) == "A")
  )
  noninformative_genes <- rownames(allocation)[all_a_flag]
  allocation <- allocation[!all_a_flag, , drop = FALSE]
  if (nrow(allocation) == 0) {
    stop(paste(
      method_name,
      "has no allocatable cluster-discriminating genes after excluding",
      "failed and all-A genes."
    ))
  }

  output_rows <- list()
  output_index <- 1L
  for (cluster_id in levels(labels)) {
    if (!cluster_id %in% colnames(allocation)) {
      next
    }
    grades <- toupper(as.character(allocation[, cluster_id]))
    names(grades) <- rownames(allocation)
    retained <- selected_genes[
      selected_genes %in% names(grades)[grades %in% c("A", "B")]
    ]
    if (length(retained) == 0) {
      next
    }
    output_rows[[output_index]] <- data.frame(
      method = method_name,
      cluster = cluster_id,
      rank = seq_along(retained),
      gene = retained,
      selection_rank = match(retained, selected_genes),
      allocation_grade = unname(grades[retained]),
      allocation_rule = paste(
        "Festem marker_allocate_core grade A or B;",
        "failed and all-A genes excluded"
      ),
      stringsAsFactors = FALSE
    )
    output_index <- output_index + 1L
  }

  if (length(output_rows) == 0) {
    stop(paste(method_name, "allocation returned no cluster-specific markers."))
  }
  output <- do.call(rbind, output_rows)
  missing_clusters <- setdiff(levels(labels), unique(output$cluster))
  if (length(missing_clusters) > 0) {
    stop(paste(
      method_name,
      "has no A/B-allocated markers for clusters:",
      paste(missing_clusters, collapse = ", "),
      "Increase --markermap-k for MarkerMap or inspect the allocation."
    ))
  }
  write.csv(output, output_csv, row.names = FALSE, na = "")
  message(sprintf(
    paste0(
      "%s allocation completed: %d retained rows; ",
      "%d failed genes removed; %d all-A genes removed."
    ),
    method_name,
    nrow(output),
    length(failed_genes),
    length(noninformative_genes)
  ))
  invisible(list(
    markers = output,
    failed_genes = failed_genes,
    noninformative_genes = noninformative_genes
  ))
}

festem_output <- allocate_panel(
  festem_selected,
  "Festem",
  festem_out_csv
)
markermap_output <- allocate_panel(
  markermap_selected,
  "MarkerMap",
  markermap_out_csv
)

r_metadata <- data.frame(
  Key = c(
    "R_version",
    "Festem_version",
    "Matrix_version",
    "N_cells",
    "N_genes",
    "N_clusters",
    "Festem_FDR",
    "Allocation_FDR",
    "Festem_seed",
    "Num_threads",
    "Block_size",
    "Festem_global_selected_genes",
    "MarkerMap_global_selected_genes",
    "Festem_cluster_list_min",
    "Festem_cluster_list_median",
    "Festem_cluster_list_max",
    "Festem_allocation_failed_n",
    "Festem_allocation_failed_genes",
    "Festem_allocation_all_A_n",
    "Festem_allocation_all_A_genes",
    "MarkerMap_cluster_list_min",
    "MarkerMap_cluster_list_median",
    "MarkerMap_cluster_list_max",
    "MarkerMap_allocation_failed_n",
    "MarkerMap_allocation_failed_genes",
    "MarkerMap_allocation_all_A_n",
    "MarkerMap_allocation_all_A_genes"
  ),
  Value = c(
    R.version.string,
    as.character(utils::packageVersion("Festem")),
    as.character(utils::packageVersion("Matrix")),
    as.character(ncol(counts)),
    as.character(nrow(counts)),
    as.character(nlevels(labels)),
    as.character(festem_fdr),
    as.character(allocation_fdr),
    as.character(festem_seed),
    as.character(num_threads),
    as.character(block_size),
    as.character(length(festem_selected)),
    as.character(length(markermap_selected)),
    as.character(min(table(festem_output$markers$cluster))),
    as.character(stats::median(table(festem_output$markers$cluster))),
    as.character(max(table(festem_output$markers$cluster))),
    as.character(length(festem_output$failed_genes)),
    paste(festem_output$failed_genes, collapse = ";"),
    as.character(length(festem_output$noninformative_genes)),
    paste(festem_output$noninformative_genes, collapse = ";"),
    as.character(min(table(markermap_output$markers$cluster))),
    as.character(stats::median(table(markermap_output$markers$cluster))),
    as.character(max(table(markermap_output$markers$cluster))),
    as.character(length(markermap_output$failed_genes)),
    paste(markermap_output$failed_genes, collapse = ";"),
    as.character(length(markermap_output$noninformative_genes)),
    paste(markermap_output$noninformative_genes, collapse = ";")
  ),
  stringsAsFactors = FALSE
)
write.csv(r_metadata, r_metadata_csv, row.names = FALSE, na = "")
message("Festem and MarkerMap cluster-specific CSV files were written.")
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cluster-specific Festem and MarkerMap CSV files from the "
            "CD8 benchmark AnnData object."
        )
    )
    parser.add_argument("--input-h5ad", default="cd8_benchmark_data.h5ad")
    parser.add_argument("--cluster-col", default="meta.cluster")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--expression-layer", default="logcounts")
    parser.add_argument("--outdir", default="paper/revision_tables")
    parser.add_argument("--festem-output", default="festem_cd8_markers.csv")
    parser.add_argument("--markermap-output", default="markermap_cd8_markers.csv")
    parser.add_argument(
        "--festem-global-output",
        default="festem_cd8_global_panel.csv",
    )
    parser.add_argument(
        "--markermap-global-output",
        default="markermap_cd8_global_panel.csv",
    )
    parser.add_argument(
        "--metadata-output",
        default="L4_Festem_MarkerMap_preparation_metadata.csv",
    )
    parser.add_argument("--markermap-k", type=int, default=50)
    parser.add_argument("--markermap-hidden-size", type=int, default=64)
    parser.add_argument("--markermap-z-size", type=int, default=16)
    parser.add_argument("--markermap-batch-size", type=int, default=64)
    parser.add_argument("--markermap-min-epochs", type=int, default=10)
    parser.add_argument("--markermap-max-epochs", type=int, default=100)
    parser.add_argument("--markermap-seed", type=int, default=42)
    parser.add_argument("--festem-fdr", type=float, default=0.05)
    parser.add_argument("--allocation-fdr", type=float, default=0.05)
    parser.add_argument("--festem-seed", type=int, default=321)
    parser.add_argument(
        "--festem-threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--festem-block-size", type=int, default=40000)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-timeout-sec", type=int, default=14400)
    parser.add_argument(
        "--force-marker-selection",
        action="store_true",
        help=(
            "Rerun MarkerMap and Festem selection even when validated global "
            "panel caches already exist."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate and summarize the AnnData object without running either method.",
    )
    args = parser.parse_args()

    positive_ints = [
        "markermap_k",
        "markermap_hidden_size",
        "markermap_z_size",
        "markermap_batch_size",
        "markermap_min_epochs",
        "markermap_max_epochs",
        "festem_threads",
        "festem_block_size",
        "r_timeout_sec",
    ]
    for name in positive_ints:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.markermap_max_epochs <= args.markermap_min_epochs:
        parser.error("--markermap-max-epochs must exceed --markermap-min-epochs.")
    for name in ("festem_fdr", "allocation_fdr"):
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")
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


def matrix_summary(matrix: Any, name: str) -> dict[str, Any]:
    import numpy as np
    from scipy import sparse

    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values.")
    sample = finite[: min(500_000, finite.size)]
    return {
        "name": name,
        "min": float(sample.min()),
        "max": float(sample.max()),
        "integer_fraction": float(
            np.mean(np.isclose(sample, np.round(sample)))
        ),
        "sampled_values": int(sample.size),
    }


def validate_adata(adata: Any, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    if args.cluster_col not in adata.obs.columns:
        raise ValueError(
            f"Missing cluster column {args.cluster_col!r}; "
            f"observed={list(adata.obs.columns)}"
        )
    for layer in (args.counts_layer, args.expression_layer):
        if layer not in adata.layers:
            raise ValueError(
                f"Missing AnnData layer {layer!r}; "
                f"observed={list(adata.layers.keys())}"
            )
    labels = (
        adata.obs[args.cluster_col]
        .astype("string")
        .fillna("")
        .astype(str)
    )
    if labels.eq("").any():
        raise ValueError("Empty cluster labels are present.")
    if labels.nunique() < 2:
        raise ValueError("At least two clusters are required.")
    if adata.var_names.astype(str).duplicated().any():
        duplicates = adata.var_names[
            adata.var_names.astype(str).duplicated()
        ][:10]
        raise ValueError(f"Duplicate gene identifiers found: {list(duplicates)}")

    counts_summary = matrix_summary(
        adata.layers[args.counts_layer],
        args.counts_layer,
    )
    expression_summary = matrix_summary(
        adata.layers[args.expression_layer],
        args.expression_layer,
    )
    if counts_summary["min"] < 0:
        raise ValueError("The counts layer contains negative values.")
    if counts_summary["integer_fraction"] < 0.999:
        raise ValueError(
            f"{args.counts_layer!r} does not appear to contain raw integer counts "
            f"(integer fraction={counts_summary['integer_fraction']:.4f})."
        )
    if expression_summary["min"] < 0:
        raise ValueError("The expression layer contains negative values.")
    if not np.isfinite(expression_summary["max"]):
        raise ValueError("The expression layer contains non-finite values.")

    cluster_counts = labels.value_counts().sort_index()
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_clusters": int(labels.nunique()),
        "cluster_counts": {
            str(key): int(value) for key, value in cluster_counts.items()
        },
        "counts_summary": counts_summary,
        "expression_summary": expression_summary,
    }


def write_matrix_market(matrix: Any, path: Path) -> None:
    import numpy as np
    from scipy import sparse
    from scipy.io import mmwrite

    if sparse.issparse(matrix):
        gene_by_cell = matrix.transpose().tocoo()
    else:
        gene_by_cell = sparse.coo_matrix(matrix.transpose())
    if path.name.startswith("counts"):
        gene_by_cell = gene_by_cell.astype(np.int64)
        field = "integer"
    else:
        gene_by_cell = gene_by_cell.astype(np.float32)
        field = "real"
    mmwrite(path, gene_by_cell, field=field)


def run_markermap(
    adata: Any,
    args: argparse.Namespace,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from scipy import sparse

    try:
        import torch
        from markermap.utils import split_data
        from markermap.vae_models import MarkerMap, train_model
    except ImportError as exc:
        raise ImportError(
            "MarkerMap could not be imported with its runtime dependencies. "
            "Confirm with: python -c 'from markermap.vae_models import "
            "MarkerMap, train_model; print(\"MarkerMap runtime: OK\")'"
        ) from exc

    np.random.seed(args.markermap_seed)
    torch.manual_seed(args.markermap_seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass

    expression = adata.layers[args.expression_layer]
    dense = (
        expression.toarray()
        if sparse.issparse(expression)
        else np.asarray(expression)
    )
    dense = np.asarray(dense, dtype=np.float32)
    if dense.shape != adata.shape:
        raise ValueError(
            f"MarkerMap expression shape {dense.shape} != AnnData shape {adata.shape}."
        )

    labels = (
        adata.obs[args.cluster_col]
        .astype("category")
        .cat.remove_unused_categories()
    )
    adata.obs[args.cluster_col] = labels
    adata.X = dense

    train_idx, val_idx, test_idx = split_data(
        dense,
        labels.astype(str).to_numpy(),
        [0.7, 0.1, 0.2],
        seed=args.markermap_seed,
        min_groups=[1, 1, 1],
    )
    train_loader, val_loader = MarkerMap.prepareData(
        adata,
        train_idx,
        val_idx,
        args.cluster_col,
        None,
        batch_size=args.markermap_batch_size,
    )
    model = MarkerMap(
        input_size=adata.n_vars,
        hidden_layer_size=args.markermap_hidden_size,
        z_size=args.markermap_z_size,
        num_classes=labels.cat.categories.size,
        k=args.markermap_k,
        loss_tradeoff=0,
    )
    train_model(
        model,
        train_loader,
        val_loader,
        min_epochs=args.markermap_min_epochs,
        max_epochs=args.markermap_max_epochs,
        early_stopping_patience=3,
        verbose=True,
    )

    # MarkerMap's public ``markers()`` method calls ``top_logits()`` and ranks
    # the resulting aggregate feature weights.  Call ``top_logits()`` exactly
    # once so that the saved scores and selected indices come from the same
    # Gumbel draw.  Resetting the seed makes this final discrete extraction
    # reproducible independently of how many random numbers training consumed.
    torch.manual_seed(args.markermap_seed)
    with torch.no_grad():
        _, aggregate_scores = model.top_logits()
        selected_indices_tensor = torch.argsort(
            aggregate_scores,
            descending=True,
        )[: args.markermap_k]
        selected_scores_tensor = aggregate_scores[selected_indices_tensor]
    selected_indices = (
        selected_indices_tensor.detach().cpu().numpy().astype(int)
    )
    selected_scores = (
        selected_scores_tensor.detach().cpu().numpy().astype(float)
    )
    if len(selected_indices) != args.markermap_k:
        raise RuntimeError(
            f"MarkerMap returned {len(selected_indices)} markers; "
            f"expected {args.markermap_k}."
        )

    result = __import__("pandas").DataFrame(
        {
            "gene": adata.var_names[selected_indices].astype(str),
            "selection_rank": np.arange(1, len(selected_indices) + 1),
            "selection_score": selected_scores,
            "feature_index": selected_indices,
            "method": "MarkerMap_supervised",
        }
    )
    result.to_csv(output_path, index=False)
    return {
        "reused_cached_panel": False,
        "train_cells": int(len(train_idx)),
        "validation_cells": int(len(val_idx)),
        "held_out_cells_not_used_for_selection": int(len(test_idx)),
        "selected_genes": int(len(result)),
    }


def validate_cached_global_panel(
    path: Path,
    *,
    expected_method: str,
    expected_gene_count: int,
    available_genes: set[str],
) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"gene", "selection_rank", "method"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Cached {expected_method} panel {path} is missing columns: {missing}. "
            "Use --force-marker-selection to regenerate it."
        )
    methods = set(frame["method"].dropna().astype(str))
    if methods != {expected_method}:
        raise ValueError(
            f"Cached panel {path} has method={methods}, expected "
            f"{expected_method!r}. Use --force-marker-selection."
        )
    genes = frame["gene"].fillna("").astype(str).str.strip()
    ranks = pd.to_numeric(frame["selection_rank"], errors="coerce")
    if genes.eq("").any() or genes.duplicated().any():
        raise ValueError(
            f"Cached panel {path} has empty or duplicated genes. "
            "Use --force-marker-selection."
        )
    if ranks.isna().any() or ranks.duplicated().any():
        raise ValueError(
            f"Cached panel {path} has invalid or duplicated ranks. "
            "Use --force-marker-selection."
        )
    if len(frame) != expected_gene_count:
        raise ValueError(
            f"Cached {expected_method} panel contains {len(frame)} genes, "
            f"but the requested run expects {expected_gene_count}. "
            "Use --force-marker-selection."
        )
    absent = sorted(set(genes) - available_genes)
    if absent:
        raise ValueError(
            f"Cached panel {path} contains genes absent from the AnnData object: "
            f"{absent[:10]}. Use --force-marker-selection."
        )
    return {
        "reused_cached_panel": True,
        "selected_genes": int(len(frame)),
        "cached_panel": str(path.resolve()),
    }


def validate_external_output(
    path: Path,
    *,
    expected_method: str,
    expected_clusters: set[str],
) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {
        "method",
        "cluster",
        "rank",
        "gene",
        "selection_rank",
        "allocation_grade",
        "allocation_rule",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")
    methods = set(frame["method"].astype(str))
    if methods != {expected_method}:
        raise ValueError(f"{path} has unexpected method values: {methods}")
    observed_clusters = set(frame["cluster"].astype(str))
    if observed_clusters != expected_clusters:
        raise ValueError(
            f"{path} cluster mismatch: "
            f"missing={sorted(expected_clusters - observed_clusters)}, "
            f"extra={sorted(observed_clusters - expected_clusters)}"
        )
    if frame.duplicated(["cluster", "rank"]).any():
        raise ValueError(f"{path} contains duplicate cluster/rank rows.")
    lengths = frame.groupby("cluster").size()
    return {
        "rows": int(len(frame)),
        "clusters": int(lengths.size),
        "min_genes_per_cluster": int(lengths.min()),
        "median_genes_per_cluster": float(lengths.median()),
        "max_genes_per_cluster": int(lengths.max()),
    }


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    input_path: Path,
    adata_summary: dict[str, Any],
    markermap_summary: dict[str, Any],
    festem_output_summary: dict[str, Any],
    markermap_output_summary: dict[str, Any],
    r_metadata_path: Path,
) -> None:
    import pandas as pd

    festem_global_path = Path(args.outdir) / args.festem_global_output
    markermap_global_path = Path(args.outdir) / args.markermap_global_output
    metadata: dict[str, Any] = {
        "Analysis_ID": ANALYSIS_ID,
        "Timestamp": now_iso(),
        "Input_h5ad": str(input_path.resolve()),
        "Input_h5ad_SHA256": sha256_file(input_path),
        "Cluster_Column": args.cluster_col,
        "Counts_Layer": args.counts_layer,
        "Expression_Layer": args.expression_layer,
        "N_Cells": adata_summary["n_cells"],
        "N_Genes": adata_summary["n_genes"],
        "N_Clusters": adata_summary["n_clusters"],
        "Cluster_Counts_JSON": json.dumps(
            adata_summary["cluster_counts"],
            sort_keys=True,
        ),
        "Festem_Selection": (
            "RunFestem on raw counts with existing cluster labels as prior; "
            "FDR-selected clustering_features"
        ),
        "MarkerMap_Selection": (
            "supervised MarkerMap on logcounts with 70% train, 10% validation, "
            "20% held out from marker selection"
        ),
        "Cluster_Allocation": (
            "Robust per-gene wrapper around Festem marker_allocate_core on "
            "logcounts; retain grades A/B; exclude allocation failures and "
            "all-A non-discriminating genes"
        ),
        "Interpretation_Boundary": (
            "MarkerMap natively returns a global panel; its cluster-specific "
            "lists are a prespecified post-selection conversion using the same "
            "allocation rule as the Festem panel"
        ),
        "Festem_FDR": args.festem_fdr,
        "Allocation_FDR": args.allocation_fdr,
        "Festem_Seed": args.festem_seed,
        "Festem_Threads": args.festem_threads,
        "Festem_Block_Size": args.festem_block_size,
        "MarkerMap_k": args.markermap_k,
        "MarkerMap_Hidden_Size": args.markermap_hidden_size,
        "MarkerMap_Z_Size": args.markermap_z_size,
        "MarkerMap_Batch_Size": args.markermap_batch_size,
        "MarkerMap_Min_Epochs": args.markermap_min_epochs,
        "MarkerMap_Max_Epochs": args.markermap_max_epochs,
        "MarkerMap_Seed": args.markermap_seed,
        "Festem_Global_Panel": str(festem_global_path.resolve()),
        "Festem_Global_Panel_SHA256": sha256_file(festem_global_path),
        "MarkerMap_Global_Panel": str(markermap_global_path.resolve()),
        "MarkerMap_Global_Panel_SHA256": sha256_file(
            markermap_global_path
        ),
        "MarkerMap_Run_JSON": json.dumps(markermap_summary, sort_keys=True),
        "Festem_Output_JSON": json.dumps(
            festem_output_summary,
            sort_keys=True,
        ),
        "MarkerMap_Output_JSON": json.dumps(
            markermap_output_summary,
            sort_keys=True,
        ),
        "Festem_Repository": FESTEM_REPOSITORY,
        "MarkerMap_Repository": MARKERMAP_REPOSITORY,
        "Python": sys.version.replace("\n", " "),
        "Python_Executable": sys.executable,
        "Platform": platform.platform(),
        "Hostname": socket.gethostname(),
        "anndata_version": package_version("anndata"),
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "scipy_version": package_version("scipy"),
        "torch_version": package_version("torch"),
        "markermap_version": package_version("markermap"),
    }
    if r_metadata_path.exists():
        r_metadata = pd.read_csv(r_metadata_path)
        for _, row in r_metadata.iterrows():
            metadata[f"R_{row['Key']}"] = row["Value"]
    pd.DataFrame(
        {"Key": list(metadata), "Value": list(metadata.values())}
    ).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_h5ad)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Install the Python analysis dependencies in the active environment."
        ) from exc

    print(f"Reading AnnData: {input_path.resolve()}")
    adata = ad.read_h5ad(input_path)
    adata_summary = validate_adata(adata, args)
    print(
        f"Validated {adata_summary['n_cells']:,} cells, "
        f"{adata_summary['n_genes']:,} genes, "
        f"{adata_summary['n_clusters']} clusters."
    )
    print(
        f"  counts integer fraction: "
        f"{adata_summary['counts_summary']['integer_fraction']:.4f}"
    )
    print(
        f"  expression range: "
        f"{adata_summary['expression_summary']['min']:.4g} to "
        f"{adata_summary['expression_summary']['max']:.4g}"
    )
    if args.inspect_only:
        print(json.dumps(adata_summary, indent=2, sort_keys=True))
        return

    if shutil.which(args.rscript) is None:
        raise FileNotFoundError(f"Rscript executable not found: {args.rscript}")

    festem_output = outdir / args.festem_output
    markermap_output = outdir / args.markermap_output
    festem_global_output = outdir / args.festem_global_output
    markermap_global_output = outdir / args.markermap_global_output
    metadata_output = outdir / args.metadata_output
    r_metadata_output = outdir / "L4_Festem_R_metadata.csv"
    reuse_festem_global = (
        festem_global_output.exists()
        and not args.force_marker_selection
    )

    with tempfile.TemporaryDirectory(prefix="llmsccurator-l4-external-") as temp:
        workdir = Path(temp)
        counts_mtx = workdir / "counts_gene_by_cell.mtx"
        logcounts_mtx = workdir / "logcounts_gene_by_cell.mtx"
        genes_txt = workdir / "genes.txt"
        labels_txt = workdir / "labels.txt"
        r_runner = workdir / "run_festem_and_allocate.R"

        print("Writing temporary sparse matrices for Festem.")
        write_matrix_market(
            adata.layers[args.counts_layer],
            counts_mtx,
        )
        write_matrix_market(
            adata.layers[args.expression_layer],
            logcounts_mtx,
        )
        genes_txt.write_text(
            "\n".join(adata.var_names.astype(str)) + "\n",
            encoding="utf-8",
        )
        labels_txt.write_text(
            "\n".join(
                adata.obs[args.cluster_col]
                .astype("string")
                .fillna("")
                .astype(str)
            )
            + "\n",
            encoding="utf-8",
        )

        if (
            markermap_global_output.exists()
            and not args.force_marker_selection
        ):
            print(
                "Reusing cached supervised MarkerMap global panel: "
                f"{markermap_global_output}"
            )
            markermap_summary = validate_cached_global_panel(
                markermap_global_output,
                expected_method="MarkerMap_supervised",
                expected_gene_count=args.markermap_k,
                available_genes=set(adata.var_names.astype(str)),
            )
        else:
            print(
                f"Training supervised MarkerMap (k={args.markermap_k}, "
                f"seed={args.markermap_seed})."
            )
            markermap_summary = run_markermap(
                adata,
                args,
                markermap_global_output,
            )
            print(
                "Cached MarkerMap global panel: "
                f"{markermap_global_output}"
            )

        r_runner.write_text(R_RUNNER, encoding="utf-8")
        command = [
            args.rscript,
            str(r_runner),
            str(counts_mtx),
            str(logcounts_mtx),
            str(genes_txt),
            str(labels_txt),
            str(markermap_global_output.resolve()),
            str(festem_global_output.resolve()),
            str(festem_output.resolve()),
            str(markermap_output.resolve()),
            str(r_metadata_output.resolve()),
            str(args.festem_fdr),
            str(args.allocation_fdr),
            str(args.festem_seed),
            str(args.festem_threads),
            str(args.festem_block_size),
            str(reuse_festem_global),
        ]
        print(
            f"Running Festem and common cluster allocation with "
            f"{args.festem_threads} R workers."
        )
        subprocess.run(
            command,
            check=True,
            timeout=args.r_timeout_sec,
        )

    expected_clusters = set(
        adata.obs[args.cluster_col]
        .astype("string")
        .fillna("")
        .astype(str)
    )
    festem_summary = validate_external_output(
        festem_output,
        expected_method="Festem",
        expected_clusters=expected_clusters,
    )
    markermap_output_summary = validate_external_output(
        markermap_output,
        expected_method="MarkerMap",
        expected_clusters=expected_clusters,
    )
    write_metadata(
        metadata_output,
        args=args,
        input_path=input_path,
        adata_summary=adata_summary,
        markermap_summary=markermap_summary,
        festem_output_summary=festem_summary,
        markermap_output_summary=markermap_output_summary,
        r_metadata_path=r_metadata_output,
    )

    print("External marker preparation completed.")
    print(
        f"  Festem: {festem_summary['min_genes_per_cluster']}–"
        f"{festem_summary['max_genes_per_cluster']} genes/cluster "
        f"(median {festem_summary['median_genes_per_cluster']:.1f})"
    )
    print(
        f"  MarkerMap: {markermap_output_summary['min_genes_per_cluster']}–"
        f"{markermap_output_summary['max_genes_per_cluster']} genes/cluster "
        f"(median {markermap_output_summary['median_genes_per_cluster']:.1f})"
    )
    print(f"  Festem CSV: {festem_output}")
    print(f"  MarkerMap CSV: {markermap_output}")
    print(f"  Festem global cache: {festem_global_output}")
    print(f"  MarkerMap global cache: {markermap_global_output}")
    print(f"  Metadata: {metadata_output}")


if __name__ == "__main__":
    main()
