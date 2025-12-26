# User guide

This page summarizes common usage patterns and helps you choose the right entry point.  
For the conceptual model (masking, rescue, leakage filtering, hierarchical inference), see **[Concepts](concepts.md)**.

---

## Typical workflows

### Scanpy (clusters → curated markers → LLM labels)

Use this when you already work with `AnnData` (Scanpy/Scanpy-compatible pipelines).

1. Start from a clustered `AnnData` (e.g., `adata.obs["leiden"]`).
2. Distill per-cluster marker lists (mask → rescue → leakage filter).
3. Query an LLM with lineage-aware context to obtain `major_type` / `fine_type` labels.

Tip:
- See **[Concepts](concepts.md)** for what “mask”, “rescue”, and “leakage” mean in practice.

---

### Seurat (recommended)

Use this when your primary workflow is Seurat and you want to keep downstream plotting and analysis in R.  
LLM-scCurator operates on `AnnData`, so the recommended pattern is:

**Export from Seurat → run the Python workflow → re-import labels into Seurat**.

- R helper (export): **[examples/R/export_to_curator.R](https://github.com/kenflab/LLM-scCurator/blob/master/examples/R/export_to_curator.R)**

---

### Spatial (Visium / Xenium)

Use the same workflow on spatially resolved matrices; the distillation and hierarchical annotation logic is unchanged.

- **Visium:** spot-level annotation with optional pseudo-bulk validation
- **Xenium:** cell-level annotation with spatial coordinates preserved

See **[Tutorials](tutorials.md)** for paper spatial validation notebooks (Xenium/Visium).


---

## Manuscript reproduction (paper/)

For deterministic reproduction of benchmarks, figures, and Source Data, follow:

- **[paper/README.md](https://github.com/kenflab/LLM-scCurator/blob/master/paper/README.md)**
- Figures are generated from [`paper/source_data/`](https://github.com/kenflab/LLM-scCurator/blob/master/paper/source_data/) (re-running LLM calls is optional).
