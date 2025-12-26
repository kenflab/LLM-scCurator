# paper/: manuscript-facing assets

This directory contains the minimal, manuscript-facing assets to **inspect and verify** the benchmarks, figures, and Source Data.

## What is versioned here
- [`README.md`](README.md): this guide
- [`FIGURE_MAP.csv`](FIGURE_MAP.csv): panel → Source Data file (and optional notebook provenance)
- [`config/`](config): dataset pointers, parameters, and deterministic label maps
- [`scripts/`](scripts): optional, small utilities to (re)generate selected intermediates and render a subset of panels
- [`notebooks/`](notebooks): optional, read-only notebooks for provenance/inspection
- [`source_data/subsampled_ids/`](source_data/subsampled_ids): the fixed cell sets used in the manuscript (Source Data)
- [`source_data/figure_data/`](source_data/figure_data): panel-level CSVs underlying each figure panel (Source Data)

## What is NOT versioned here
Raw expression matrices (e.g., `.h5ad`) are not distributed in this repository. All input datasets are publicly available from their original repositories (see [`config/datasets.tsv`](config/datasets.tsv)). Reproducibility is anchored on the subsampled cell ID lists in [`source_data/subsampled_ids/`](source_data/subsampled_ids).

## How to review
- Inspect the exact numeric values used for plotting in [`source_data/figure_data/`](source_data/figure_data) (and the per-figure Excel workbooks in [`source_data/`](source_data), if provided).
- Use [`FIGURE_MAP.csv`](FIGURE_MAP.csv) to locate the Source Data file (and the corresponding notebook, when available) for any panel.

### Optional: render supported panels
Some panels can be rendered from precomputed Source Data via:
```bash
python scripts/make_figures.py --make-fig2a --make-confusions
```
## Reviewer notes: benchmarking & evaluation

The evaluation logic is intentionally **deterministic** and **backend-agnostic**: it scores
already-produced prediction tables and does not call any LLM APIs during evaluation.

### Ground-truth harmonization (single source of truth)
Ground-truth labels (`Ground_Truth`) are derived **only from the author-provided cluster name strings**
using deterministic mapping functions in [`../benchmarks/gt_mappings.py`](../benchmarks/gt_mappings.py). These mappings are conservative
and prefer stable, interpretable categories over overfitting fine subtypes.

### Ontology-aware hierarchical scoring
We score predictions using a simple two-level ontology:
(i) **major lineage** and (ii) **within-lineage state**. Scoring is computed by
[`../benchmarks/hierarchical_scoring.py`](../benchmarks/hierarchical_scoring.py) with dataset-specific `HierarchyConfig` objects:

- CD8: [`../benchmarks/cd8_config.py`](../benchmarks/cd8_config.py) (w_lineage=0.7, w_state=0.3; strict T vs NK penalties)
- CD4: [`../benchmarks/cd4_config.py`](../benchmarks/cd4_config.py) (w_lineage=0.7, w_state=0.3; strict cross-lineage penalties)
- CAF/MSC: [`../benchmarks/caf_config.py`](../benchmarks/caf_config.py) (w_lineage=0.3, w_state=0.7; Fibroblast↔Endothelial partial lineage credit)
- Mouse B-lineage (decoy robustness): [`../benchmarks/mouse_b_config.py`](../benchmarks/mouse_b_config.py) (w_lineage=0.5, w_state=0.5)

This scheme awards full credit only when both the expected major lineage and state match, and applies
hard cross-lineage penalties for biologically incompatible calls.

### Reproducibility notes
- All preprocessing and subsampling are fixed by the ID lists in [`source_data/subsampled_ids/`](source_data/subsampled_ids/).
- Local preprocessing uses fixed random seeds where applicable; however, LLM outputs may still vary
  across runs even with temperature=0 depending on the backend/provider.
- Evaluation outputs are derived artifacts written from integrated CSVs and do not modify upstream inputs.
