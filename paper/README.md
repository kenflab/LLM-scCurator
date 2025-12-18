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
