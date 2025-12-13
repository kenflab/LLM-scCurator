# paper/

This directory contains the minimal, manuscript-facing assets to reproduce the benchmarks, figures, and Source Data.

## What is versioned here
- `README.md`: this guide
- `config/`: dataset pointers, parameters, and deterministic label maps
- `scripts/`: entrypoints to reproduce benchmark outputs and figures
- `notebooks/`: optional, read-only development notebooks (not the canonical pipeline)
- `source_data/subsampled_ids/`: the fixed cell sets used in the manuscript (Source Data)
- `source_data/figure_data/`: numeric data underlying each figure panel (Source Data)

## What is NOT versioned here
Raw expression matrices (e.g., `.h5ad`) are not distributed in this repository. All input datasets are publicly available from their original repositories (see `config/datasets.tsv`). Reproducibility is anchored on the subsampled cell ID lists in `source_data/subsampled_ids/`.

## How to reproduce
1. Download the public datasets listed in `config/datasets.tsv` (or use your own local mirrors).
2. Recreate the exact cell sets using `source_data/subsampled_ids/*.csv` (or regenerate them with `scripts/`).
3. Run the scripts in `scripts/` to produce benchmark tables and figure-ready outputs.
4. The numeric values plotted in each figure panel are exported to `source_data/figure_data/`.
