# User guide

## Typical workflows

### Scanpy (clusters → curated markers → label)
1. Start from a clustered `AnnData` (e.g., `adata.obs["leiden"]`).
2. Curate markers per cluster (mask/rescue/leakage filters).
3. Annotate with LLM using lineage-aware context.

### Seurat (recommended)
Export Seurat → `.h5ad` and run the Python workflow.
(See `examples/R/export_script.R` in the repository.)

### Spatial (Visium / Xenium)
Use the same pipeline on spatially-resolved cell/spot matrices:
- Visium: spot-level annotations + pseudo-bulk validation
- Xenium: cell-level annotations with spatial coordinates preserved

## Manuscript reproduction (paper/)
For deterministic reproduction of benchmarks, figures, and Source Data, follow:
- `paper/README.md`
- Figures are generated from `paper/source_data/` (re-running LLM calls is optional).
