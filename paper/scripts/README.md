# scripts/

Small utilities used during manuscript preparation.

Most review/verification can be done directly from [`../source_data/`](../source_data/`) (canonical Source Data),
indexed by [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv). Re-running scripts is optional.

> Tip: run commands from the repository root to keep paths consistent.

## Scripts

- ### apply_label_map.py
Applies a YAML label map (substring, case-insensitive) to a text field or a CSV column.


- ### export_subsampled_ids.py
Exports the fixed cell/spot identifier lists used in the manuscript ([`../source_data/subsampled_ids/`](`../source_data/subsampled_ids)).


- ### run_benchmarks.py (optional; advanced)
Optional re-run entrypoint that may regenerate benchmark intermediates from large public inputs.
This typically requires downloading datasets listed in [`../config/datasets.tsv`](../config/datasets.tsv) and setting API keys (if applicable).
Outputs are not required for Source Data inspection.


- ### make_figures.py (optional; best-effort rendering)
A lightweight renderer for a subset of panels from precomputed Source Data (see [`../source_data/figure_data/`](../source_data/figure_data)).
If it does not run in your environment, you can still verify all numeric values directly from [`../source_data/`](`../source_data).

Example:
```bash
python paper/scripts/make_figures.py --help
```
If supported in your setup:
```
python paper/scripts/make_figures.py --make-fig2a --make-confusions
```
