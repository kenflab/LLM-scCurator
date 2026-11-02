# User guide

This page summarizes common usage patterns and helps you choose the right entry point.  
For marker curation, lineage-context supplementation and coarse-to-fine inference, see **[Concepts](concepts.md)** and the **[integrated workflow guide](integrated-workflow.md)**.

---

## Typical workflows

### Scanpy (clusters → curated markers → LLM labels)

Use this when you already work with `AnnData` (Scanpy/Scanpy-compatible pipelines).

1. Start from a clustered `AnnData` (e.g., `adata.obs["leiden"]`).
2. Curate the marker list for one cluster; this also estimates its broad lineage context from the target expression matrix.
3. Annotate that same cluster with `use_auto_context=True` before preparing the next cluster. Export the cluster annotation table and review the predictions against expression evidence.

The separate hierarchical discovery entry point performs coarse and fine clustering and returns `major_type` / `fine_type`. Annotating existing clusters does not by itself create these two hierarchy columns.

Tip:
- See **[Concepts](concepts.md)** for masking, rescue and optional cross-lineage expression filtering. The latter requires coarse labels with documented provenance.

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
- **[Evaluation scope](https://github.com/kenflab/LLM-scCurator/blob/main/paper/EVALUATION_SCOPE.md)** distinguishes integrated workflow comparisons from marker-curation-only comparisons.
- Figures are generated from [`paper/source_data/`](https://github.com/kenflab/LLM-scCurator/blob/master/paper/source_data/) (re-running LLM calls is optional).

## Record new annotation calls

Wrap a backend with the optional `RecordingBackend` to retain exact prompts,
configured settings and raw model-response text before parsing. See
[Response recording](response-recording.md) for an example and the capture scope.
