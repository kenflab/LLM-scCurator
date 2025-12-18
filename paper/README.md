# paper/: manuscript-facing assets

This directory contains the minimal, manuscript-facing assets to reproduce the benchmarks, figures, and Source Data.

## What is versioned here
- [`README.md`](README.md): this guide
- [`FIGURE_MAP.csv`](FIGURE_MAP.csv): panel → Source Data file (and notebook provenance)
- [`config/`](config): dataset pointers, parameters, and deterministic label maps
- [`scripts/`](scripts): selected entrypoints to regenerate intermediate benchmark tables and a subset of figures
- [`notebooks/`](notebooks): optional, read-only development notebooks (not the canonical pipeline)
- [`source_data/subsampled_ids/`](source_data/subsampled_ids): the fixed cell sets used in the manuscript (Source Data)
- [`source_data/figure_data/`](source_data/figure_data): numeric data underlying each figure panel (Source Data)

- Note: for some panels (e.g., Extended Data Fig. 2), `*_data.csv` files are **per-cluster scored tables**
(exports of `*_SCORED.csv`), which are aggregated at plot time (no pre-aggregated confusion CSV is stored).

## What is NOT versioned here
Raw expression matrices (e.g., `.h5ad`) are not distributed in this repository. All input datasets are publicly available from their original repositories (see [`config/datasets.tsv`](config/datasets.tsv)). Reproducibility is anchored on the subsampled cell ID lists in [`source_data/subsampled_ids/`](source_data/subsampled_ids).

## How to reproduce
- Inspect the exact numeric values used for plotting in [`source_data/figure_data/`](source_data/figure_data) (and the per-figure Excel workbooks in [`source_data/`](source_data), if provided).
- Use [`FIGURE_MAP.csv`](FIGURE_MAP.csv) to locate the Source Data file (and the notebook section) for any panel.

### Optional: render supported panels
Some panels can be rendered from precomputed Source Data via:
```bash
python scripts/make_figures.py --make-fig2a --make-confusions
```
