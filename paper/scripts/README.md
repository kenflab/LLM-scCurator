# scripts/

Small utilities used during manuscript preparation.

Most review/verification can be done directly from [`../source_data/`](../source_data/`) (canonical Source Data),
indexed by [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv). Re-running scripts is optional.

> Tip: run commands from the repository root to keep paths consistent.

## Environment (optional)

If you want to re-run scripts locally, please follow the main README for setup:
- **Installation / Docker**: [`README.md`](../../README.md#-installation)
- **Docker (official environment)**: [`README.md`](../../README.md#-docker-official-environment)
- **Backends (LLM API keys)** (if applicable): [`README.md`](../../README.md#-backends-llm-api-keys-setup)

## Scripts

- [apply_label_map.py](apply_label_map.py) <br> 
  Applies a YAML label map (substring, case-insensitive) to a text field or a CSV column. <br>

- [export_subsampled_ids.py](export_subsampled_ids.py) <br> 
  Exports the fixed cell/spot identifier lists used in the manuscript ([`../source_data/subsampled_ids/`](../source_data/subsampled_ids)). <br>

- [run_benchmarks.py](run_benchmarks.py) (optional; advanced) <br> 
  Optional re-run entrypoint that may regenerate benchmark intermediates from large public inputs. <br>
  This typically requires downloading datasets listed in [`../config/datasets.tsv`](../config/datasets.tsv) and setting
  Backends (LLM API keys) (if applicable). <br>
  Outputs are not required for Source Data inspection. <br>

- [make_figures.py](make_figures.py) (optional) <br> 
  A lightweight renderer for a subset of panels from precomputed Source Data (see [`../source_data/figure_data/`](../source_data/figure_data)). <br>
  If it does not run in your environment, you can still verify all numeric values directly from [`../source_data/`](../source_data). <br>

  Example:
  ```bash
  python paper/scripts/make_figures.py --help
  ```
  If supported in your setup:
  ```
  python paper/scripts/make_figures.py --make-fig2a --make-confusions
  ```
  <br>

- [example_subsampled_ids_with_gt.py](example_subsampled_ids_with_gt.py) (example) <br> 
  End-to-end example that uses [`export_subsampled_ids.py`](export_subsampled_ids.py) and [`apply_label_map.py`](apply_label_map.py) <br> 
  to export subsampled IDs and add deterministic `Ground_Truth` labels using YAML maps in [`../config/label_maps/`](../config/label_maps/). <br> 

  - Script: [`example_subsampled_ids_with_gt.py`](example_subsampled_ids_with_gt.py) 
  - Notebook log (provenance): [`example_subsampled_ids_with_gt.ipynb`](example_subsampled_ids_with_gt.ipynb)
