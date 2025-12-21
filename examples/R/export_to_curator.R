# export_to_curator.R
# Seurat -> AnnData-ready export (file-based handoff for LLM-scCurator)
#
# Outputs (outdir/):
#   counts.mtx      genes x cells (MatrixMarket)
#   features.tsv    gene names (1-col)
#   barcodes.tsv    cell ids (1-col)
#   obs.csv         Seurat meta.data + cell_id (+ cluster column)
#   umap.csv        cell_id, UMAP1, UMAP2 (optional; if present and not disabled)
#   sessionInfo.txt R session info
#
# Usage:
#   Rscript /content/export_to_curator.R --in_rds obj.rds --outdir out_seurat --cluster_col seurat_clusters
#
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

stopf <- function(...) stop(sprintf(...), call. = FALSE)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    in_rds = NULL,
    outdir = "out_seurat",
    cluster_col = "seurat_clusters",
    assay = NULL,          # NULL -> RNA > Spatial > DefaultAssay
    save_umap = TRUE
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    val <- if (i + 1 <= length(args)) args[[i + 1]] else NULL

    if (key %in% c("-i", "--in_rds")) { out$in_rds <- val; i <- i + 2; next }
    if (key %in% c("-o", "--outdir")) { out$outdir <- val; i <- i + 2; next }
    if (key %in% c("-c", "--cluster_col")) { out$cluster_col <- val; i <- i + 2; next }
    if (key %in% c("-a", "--assay")) { out$assay <- val; i <- i + 2; next }
    if (key == "--no_umap") { out$save_umap <- FALSE; i <- i + 1; next }

    if (key %in% c("-h", "--help")) {
      cat("
Export Seurat object to an AnnData-ready folder (LLM-scCurator)

Usage:
  Rscript /content/export_to_curator.R --in_rds <obj.rds> --outdir <dir> [options]

Required:
  --in_rds, -i      Path to Seurat .rds file

Options:
  --outdir, -o      Output directory (default: out_seurat)
  --cluster_col, -c Metadata column name for clusters (default: seurat_clusters)
  --assay, -a       Assay to export (default: auto: RNA > Spatial > DefaultAssay)
  --no_umap         Do not export UMAP even if present

Outputs (outdir/):
  counts.mtx        genes x cells (MatrixMarket)
  features.tsv      gene names (1-col)
  barcodes.tsv      cell ids (1-col)
  obs.csv           Seurat meta.data + cell_id
  umap.csv          cell_id, UMAP1, UMAP2 (if present and not disabled)
  sessionInfo.txt   R session info
\n")
      quit(status = 0)
    }

    stopf("Unknown argument: %s (use --help)", key)
  }

  if (is.null(out$in_rds) || is.na(out$in_rds) || out$in_rds == "") {
    stopf("Missing --in_rds. Use --help.")
  }
  out
}

pick_assay <- function(obj, requested = NULL) {
  assays <- Assays(obj)
  if (!is.null(requested)) {
    if (!(requested %in% assays)) stopf("Requested assay '%s' not found. Available: %s",
                                        requested, paste(assays, collapse = ", "))
    return(requested)
  }
  if ("RNA" %in% assays) return("RNA")
  if ("Spatial" %in% assays) return("Spatial")
  DefaultAssay(obj)
}

get_counts_any <- function(obj, assay) {
  counts <- NULL
  a <- obj[[assay]]

  # Seurat v5: Assay5 + layer
  if (inherits(a, "Assay5")) {
    if ("counts" %in% Layers(a)) {
      counts <- tryCatch(
        LayerData(obj, assay = assay, layer = "counts"),
        error = function(e) NULL
      )
    }
  }

  # Seurat v4 fallback: slot
  if (is.null(counts)) {
    counts <- tryCatch(
      GetAssayData(obj, assay = assay, slot = "counts"),
      error = function(e) NULL
    )
  }

  if (is.null(counts) || nrow(counts) == 0 || ncol(counts) == 0) {
    stopf("Could not retrieve counts from assay '%s'.", assay)
  }
  if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")
  counts
}

write_export <- function(seurat_obj, outdir, cluster_col, assay = NULL, save_umap = TRUE) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  message("Preparing export for LLM-scCurator (file-based handoff)...")

  # 1) assay
  chosen_assay <- pick_assay(seurat_obj, assay)
  message(sprintf(" -> assay = %s", chosen_assay))

  # 2) counts
  counts <- get_counts_any(seurat_obj, chosen_assay)
  message(sprintf(" -> counts: %d genes x %d cells", nrow(counts), ncol(counts)))

  if (nrow(counts) < 5000) {
    warning(sprintf(
      "Only %d genes found. If this is WTA scRNA-seq, you may be exporting a reduced gene set.\nIf this is targeted (Xenium/CosMx), this may be expected.",
      nrow(counts)
    ), call. = FALSE)
  }

  # 3) meta
  meta <- seurat_obj@meta.data
  meta$cell_id <- rownames(meta)
  if (!(cluster_col %in% colnames(meta))) {
    stopf("Cluster column '%s' not found in meta.data.", cluster_col)
  }
  meta[[cluster_col]] <- as.character(meta[[cluster_col]])

  # 4) UMAP
  umap_df <- NULL
  if (save_umap && ("umap" %in% Reductions(seurat_obj))) {
    um <- Embeddings(seurat_obj, "umap")
    umap_df <- data.frame(cell_id = rownames(um), UMAP1 = um[, 1], UMAP2 = um[, 2])
    message(" -> UMAP found: exporting umap.csv")
  } else {
    message(" -> UMAP not exported (missing or disabled)")
  }

  # 5) write files
  Matrix::writeMM(counts, file.path(outdir, "counts.mtx"))
  write.table(rownames(counts), file.path(outdir, "features.tsv"),
              quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)
  write.table(colnames(counts), file.path(outdir, "barcodes.tsv"),
              quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)
  write.csv(meta, file.path(outdir, "obs.csv"), row.names = FALSE)

  if (!is.null(umap_df)) write.csv(umap_df, file.path(outdir, "umap.csv"), row.names = FALSE)

  writeLines(capture.output(sessionInfo()), file.path(outdir, "sessionInfo.txt"))

  message(sprintf("✅ Export completed: %s", normalizePath(outdir)))
  invisible(TRUE)
}

# ---- Main ----
cfg <- parse_args()

if (!file.exists(cfg$in_rds)) {
  stopf("Input file not found: %s", cfg$in_rds)
}

obj <- readRDS(cfg$in_rds)
if (!inherits(obj, "Seurat")) {
  stopf("Input is not a Seurat object: %s", cfg$in_rds)
}

write_export(
  seurat_obj = obj,
  outdir = cfg$outdir,
  cluster_col = cfg$cluster_col,
  assay = cfg$assay,
  save_umap = cfg$save_umap
)
