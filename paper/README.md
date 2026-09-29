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

## Current Figure 3abc

Run from the repository root after generating the full-precision L4 CSV and
`L4_prediction_audit.csv` with `paper/revision_metrics/make_l4_l5_letter_tables.py`.
Both inputs must come from the same evaluation run. No LLM calls are made.
Validated input snapshots are also versioned in
`paper/source_data/figure_data/Fig3/inputs/`; use these paths to reproduce the
committed Source Data without the original workbook.

```bash
python paper/scripts/make_figure3.py \
  --metrics-csv paper/revision_tables/L4_complementary_metrics.csv \
  --audit-csv paper/revision_tables/L4_prediction_audit.csv \
  --outdir paper/revision_figures/Fig3 \
  --source-data-dir paper/source_data/figure_data/Fig3 \
  --n-bootstrap 20000 --seed 42
```

The script checks all 250 available prediction records against L4 before writing
`Fig3a_c.pdf` and `.png`. It retains all 52 predefined clusters. Panel a shows mean
Sanno and percentile 95% bootstrap confidence intervals (20,000 cluster resamples
within each dataset/method, seed 42). These intervals describe each method's mean;
they are distinct from confidence intervals for paired method differences.
Panels b and c show hierarchy-consistent and strict major-lineage accuracy.
Unassessed mouse B CellTypist/Azimuth entries are NA, not zero.

`Fig3abc_data.csv` and `Fig3a_data.csv` through `Fig3c_data.csv` contain the plotted
values in fractions (0–1); the plot displays percentages. `Fig3_metadata.json`
records input/script hashes, software versions, and bootstrap settings.
The notebook `paper/notebooks/06_Fig3_minimal.ipynb` calls this same script.
HLCA panels 3d–e use the separate HLCA pipeline. Figure L1 uses
`make_l4_l5_review_figures.py`. The legacy `make_figures.py` renderer does not
regenerate the current Figure 3.

Changing from 5,000 to 20,000 bootstrap resamples may change the interval endpoints;
it does not change the means or complementary metrics. The Table L5 weight-sensitivity
analysis keeps its separately documented 10,000-resample setting.
