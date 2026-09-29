# scripts/

Small utilities used during manuscript preparation.

Most review/verification can be done directly from [`../source_data/`](../source_data/) (canonical Source Data),
indexed by [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv). Re-running scripts is optional.

> Tip: run commands from the repository root to keep paths consistent.

## Environment (optional)

If you want to re-run scripts locally, follow the main README for setup:
- **Install (pip)**: [`README.md`](../../README.md#-installation)
- **Docker (prebuilt image or local build)**: [`README.md`](../../README.md#-docker-official-environment)
- **Backends (LLM API keys)** (if applicable): [`README.md`](../../README.md#-backends-llm-api-keys-setup)


## Scripts

- [`apply_label_map.py`](apply_label_map.py) <br> 
  Applies a YAML label map (substring, case-insensitive) to a text field or a CSV column. <br>

- [`export_subsampled_ids.py`](export_subsampled_ids.py) <br> 
  Exports the fixed cell/spot identifier lists used in the manuscript ([`../source_data/subsampled_ids/`](../source_data/subsampled_ids)). <br>

- [`example_subsampled_ids_with_gt.py`](example_subsampled_ids_with_gt.py) (example) <br> 
  End-to-end example that uses [`export_subsampled_ids.py`](export_subsampled_ids.py) and [`apply_label_map.py`](apply_label_map.py) <br> 
  to export subsampled IDs and add deterministic `Ground_Truth` labels using YAML maps in [`../config/label_maps/`](../config/label_maps/). <br> 

  - Script: [`example_subsampled_ids_with_gt.py`](example_subsampled_ids_with_gt.py) 
  - Notebook log (provenance): [`example_subsampled_ids_with_gt.ipynb`](example_subsampled_ids_with_gt.ipynb) <br> 

- [`run_benchmarks.py`](run_benchmarks.py) (optional; advanced) <br> 
  Optional re-run entrypoint that may regenerate benchmark intermediates from large public inputs. <br>
  This typically requires downloading datasets listed in [`../config/datasets.tsv`](../config/datasets.tsv) and setting
  [Backends (LLM API keys)](../../README.md#-backends-llm-api-keys-setup) (if applicable). <br>
  Outputs are not required for Source Data inspection. <br>
  - Script: [`run_benchmarks.py`](run_benchmarks.py)
  - Notebook log (provenance): [`run_benchmarks.ipynb`](run_benchmarks.ipynb)
  
  Example (If supported in your setup):
  ```bash
  export GEMINI_API_KEY="YOUR_KEY_HERE"

  python paper/scripts/run_benchmarks.py \
    --config paper/config/benchmarks.yaml \
    --repo-root /work \
    --out-results paper/scripts/results \
    --cache-dir  paper/scripts/figures/cache/llm_calls \
    --datasets cd8 \
    --seed 42
  ```
  Outputs (written to scripts/figures/):
  - cd8_benchmark_results_integrated.csv
  - cd8_benchmark_run_metadata.json
   
- [`make_figures.py`](make_figures.py) (optional) <br> 
  A lightweight renderer for a subset of panels from precomputed Source Data (see [`../source_data/figure_data/`](../source_data/figure_data)). <br>
  If it does not run in your environment, you can still verify all numeric values directly from [`../source_data/`](../source_data)(see [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv)) <br>
  - Script: [`make_figures.py`](make_figures.py)
  - Notebook log (provenance): [`run_make_figures.ipynb`](run_make_figures.ipynb)
  
  Example:
  ```bash
  python paper/scripts/make_figures.py --help
  ```
  If supported in your setup:
  ```
  python paper/scripts/make_figures.py --make-fig2a --make-confusions
  ```
  Outputs (written to scripts/figures/):
  - Fig2a_d.pdf
  - Fig2a_d.png
  - EDFig2a_confusion.pdf
  - EDFig2a_confusion.png
  - EDFig2b_confusion.pdf
  - EDFig2b_confusion.png
  - EDFig2c_confusion.pdf
  - EDFig2c_confusion.png

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
