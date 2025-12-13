# Required libraries:
# The 'anndata' package allows manipulation of AnnData objects within R via reticulate.
# if (!require("anndata")) install.packages("anndata")
# if (!require("Seurat")) install.packages("Seurat")

library(Seurat)
library(anndata)
library(Matrix)

#' Export Seurat Object for LLM-scCurator Pipeline
#'
#' This function converts a processed Seurat object into an AnnData (.h5ad) file,
#' compatible with the LLM-scCurator Python framework.
#' It extracts raw count matrices and metadata, ensuring seamless integration
#' between R-based preprocessing and Python-based LLM annotation.
#'
#' @param seurat_obj A Seurat object.
#' @param save_path Output path for .h5ad file.
#' @param cluster_col Metadata column for cluster labels.
#'
#' @return None. Saves an .h5ad file to the specified path.
#' @export
#'
#' @examples
#' \dontrun{
#' export_for_llm_curator(seurat_obj, "my_data.h5ad", cluster_col = "seurat_clusters")
#' }
export_for_llm_curator <- function(seurat_obj, save_path, cluster_col = "seurat_clusters") {
  
  message(paste0("Preparing export for LLM-scCurator..."))
  
  # --- 1. Assay Selection Strategy (Prioritize Raw Data) ---
  # Users often set 'SCT' or 'Integrated' as default, which may lack genes or raw counts.
  # We enforce a search for 'RNA' or 'Spatial' first.
  
  available_assays <- Assays(seurat_obj)
  target_assay <- NULL
  
  if ("RNA" %in% available_assays) {
    target_assay <- "RNA"
    message("   -> Found 'RNA' assay. Using this for raw counts (Recommended).")
  } else if ("Spatial" %in% available_assays) {
    target_assay <- "Spatial" # For Visium
    message("   -> Found 'Spatial' assay. Using this for raw counts.")
  } else {
    target_assay <- DefaultAssay(seurat_obj)
    message(paste0("   -> Warning: 'RNA' assay not found. Using default assay: '", target_assay, "'"))
    message("      (Ensure this assay contains UNCORRECTED raw counts for all genes.)")
  }
  
  # --- 2. Extract Raw Counts ---
  counts_matrix <- NULL
  
  # Handle Seurat v5 vs v4 structure
  # Try to get the 'counts' layer/slot from the target assay
  obj_assay <- seurat_obj[[target_assay]]
  
  # Attempt 1: Seurat v5 style (Layers)
  if (is(obj_assay, "Assay5")) {
     if ("counts" %in% Layers(obj_assay)) {
       counts_matrix <- LayerData(seurat_obj, assay = target_assay, layer = "counts")
     }
  } 
  
  # Attempt 2: Seurat v3/v4 style (Slots) -> Fallback if v5 method fails or returns nothing
  if (is.null(counts_matrix)) {
     try({
       counts_matrix <- GetAssayData(seurat_obj, assay = target_assay, slot = "counts")
     }, silent = TRUE)
  }
  
  # Final check
  if (is.null(counts_matrix) || nrow(counts_matrix) == 0) {
    stop(paste0("Error: Could not find raw 'counts' in assay '", target_assay, "'. Please verify your object."))
  }

  # Check for gene count (Handling both scRNA-seq and Spatial Panels)
  n_genes <- nrow(counts_matrix)
  
  if (n_genes < 5000) {
    warning(paste0("⚠Note: Only ", n_genes, " genes found.\n",
                   "   - If this is whole-transcriptome scRNA-seq: You might be exporting only Variable Features. ",
                   "We recommend exporting the FULL transcriptome for optimal noise detection.\n",
                   "   - If this is a targeted spatial panel (e.g., Xenium, CosMx): This is expected. You can proceed."))
  }
  
  # --- 3. Transpose & Save ---
  message(paste0("   -> Exporting ", nrow(counts_matrix), " genes x ", ncol(counts_matrix), " cells..."))
  
  counts_matrix <- Matrix::t(counts_matrix)
  
  # Metadata validation
  if (!cluster_col %in% colnames(seurat_obj@meta.data)) {
    stop(paste0("Error: Cluster column '", cluster_col, "' not found."))
  }
  
  meta_data <- seurat_obj@meta.data
  meta_data[[cluster_col]] <- as.factor(meta_data[[cluster_col]])
  
  ad <- AnnData(
    X = counts_matrix,
    obs = meta_data
  )
  
  tryCatch({
    write_h5ad(ad, save_path)
    message(paste0("Saved to: ", save_path))
  }, error = function(e) {
    stop(paste0("Write failed: ", e$message))
  })
}
