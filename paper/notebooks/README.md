# notebooks/

Read-only provenance notebooks captured during manuscript preparation (runs, logs, and figure assembly notes).

These notebooks are optional and provided for transparency.
Panel verification is supported by the exported Source Data in [`../source_data/`](../source_data), indexed by [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv).
Some notebooks may depend on large public inputs and computing.

Note on LLM calls: we fix local random seeds for deterministic preprocessing; however, LLM API outputs may still vary across runs even with temperature=0.


## Table of contents

### 00 — Export fixed subsampled IDs
- CD8: [`00_export_subsample_cd8.ipynb`](00_export_subsample_cd8.ipynb)
- CD4: [`00_export_subsample_cd4.ipynb`](00_export_subsample_cd4.ipynb)
- BRCA MSC: [`00_export_subsample_brca_msc.ipynb`](00_export_subsample_brca_msc.ipynb)
- Mouse B: [`00_export_subsample_mouse_b.ipynb`](00_export_subsample_mouse_b.ipynb)

### 01 — Ground-truth QC / mapping checks
- CD8: [`01_cd8_gt_qc.ipynb`](01_cd8_gt_qc.ipynb)
- CD4: [`01_cd4_gt_qc.ipynb`](01_cd4_gt_qc.ipynb)
- BRCA MSC: [`01_brca_msc_gt_qc.ipynb`](01_brca_msc_gt_qc.ipynb)
- Mouse B: [`01_mouse_b_gt_qc.ipynb`](01_mouse_b_gt_qc.ipynb)

### 02 — Run benchmarks (development / optional re-run)
- CD8: [`02_run_cd8_benchmark.ipynb`](02_run_cd8_benchmark.ipynb)
- CD4: [`02_run_cd4_benchmark.ipynb`](02_run_cd4_benchmark.ipynb)
- BRCA MSC: [`02_run_brca_msc_benchmark.ipynb`](02_run_brca_msc_benchmark.ipynb)
- Mouse B: [`02_run_mouse_b_benchmark.ipynb`](02_run_mouse_b_benchmark.ipynb)

### 03 — Evaluate benchmarks / scoring exports
- [`03_evaluate_benchmarks.ipynb`](03_evaluate_benchmarks.ipynb): computes ontology-aware scores and summary tables used in Source Data and figure panels (Fig. 2a-d) 

### 04 — Landscape (figure assembly helper)
- [`04_landscape.ipynb`](04_landscape.ipynb): generates gene landscap (Fig1b, c, and EDFig. 1a)

### 05 — Compare top-N genes (figure assembly helper)
- [`05_compare_top_N_genes.ipynb`](05_compare_top_N_genes.ipynb): compares top-N input gene lists across methods and exports noise-category breakdowns for plotting (Fig. 2e, f)

### 06 — Simulation
- [`06_Simulation.ipynb`](06_simulation.ipynb): — runs marker noise injection to stress-test robustness and generate plot-ready summaries (Fig. 2g)


### 07 — Colon Xenium (spatial validation)
- [`07_Colon_Xenium.ipynb`](07_Colon_Xenium.ipynb): generate Xenium spatial plots and pseudo-bulk heatmap matrices (Fig. 2h and ED Fig. 3d).

### 08 — OSCC Visium (spatial validation)
- [`08_OSCC_Visium.ipynb`](08_OSCC_Visium.ipynb): generate Visium spatial plots and pseudo-bulk heatmap matrices (ED Fig. 3a–c).

### 09 — Marker_effects (spatial validation)
- [`09_marker_effects.ipynb`](09_marker_effects.ipynb): compute one-vs-rest marker effect sizes (AUROC, log2FC, Δdet) and export *_marker_effects.csv (Source Data for ED Fig. 3c–d).
