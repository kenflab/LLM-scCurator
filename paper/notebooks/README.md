# notebooks/

Read-only provenance notebooks captured during manuscript preparation (runs, logs, and figure assembly notes).

These notebooks are optional and provided for transparency.
Panel verification is supported by the exported Source Data in [`../source_data/`](../source_data), indexed by [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv).
Some notebooks may depend on large public inputs and computing.

Note on LLM calls: we fix local random seeds for deterministic preprocessing; however, LLM API outputs may still vary across runs even with temperature = 0.

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
- [`03_evaluate_benchmarks.ipynb`](03_evaluate_benchmarks.ipynb): computes ontology-aware scores and benchmark summary tables used downstream in Source Data and figure panels.

### 04 — Fig. 2a–d and Fig. S1 assembly
- [`04_Fig2a_d_FigS1.ipynb`](04_Fig2a_d_FigS1.ipynb): exports stepwise feature-level summaries, representative rank-shift tables, and supporting CD8 burden landscapes for Fig. 2a–d and Fig. S1.

### 05 — Fig. 2e–f stepwise task-level analyses
- [`05_Fig2e_f_StepwiseAnalyses.ipynb`](05_Fig2e_f_StepwiseAnalyses.ipynb): exports task-level Sanno summaries and cluster-by-variant scoring tables for Fig. 2e–f.

### 06 — Fig. 3 cross-benchmark comparison
- [`06_Fig3_minimal.ipynb`](06_Fig3_minimal.ipynb): generates cross-benchmark summary tables for Standard, LLM-scCurator, and reference-based comparators used in Fig. 3.

### 07 — Fig. 4 robustness / ambiguity analyses
- [`07_Fig4_minimal.ipynb`](07_Fig4_minimal.ipynb): exports top-N stress-test summaries, low-consistency rates, in-silico biological-noise injection results, and ambiguity-prone state comparisons for Fig. 4.

### 08 — Fig. S3 cross-dataset feature-distillation summaries
- [`08_FigS3_minimal.ipynb`](08_FigS3_minimal.ipynb): exports cross-dataset backend-free summaries of biological-noise fraction, low-Gini burden, canonical marker recall, and supporting ranked gene lists for Fig. S3.

### 09 — Colon Xenium (spatial validation)
- [`09_Colon_Xenium.ipynb`](10_Colon_Xenium.ipynb): generates Xenium spatial maps and pseudobulk heatmap matrices for Fig. 5.

### 10 — OSCC Visium (spatial validation)
- [`10_OSCC_Visium.ipynb`](09_OSCC_Visium.ipynb): generates Visium spatial maps and pseudobulk heatmap matrices for Fig. S5.

### 11 — Marker effects (spatial validation)
- [`11_marker_effects.ipynb`](11_marker_effects.ipynb): computes one-vs-rest marker effect sizes (AUROC, log2FC, Δdet) and exports `*_marker_effects.csv` files used in Source Data for Fig. 5b and Fig. S5c.