# source_data/

This directory contains the **source data files** supporting the figures in the LLM-scCurator manuscript.

Our goal is that a reader can (i) identify which file supports each figure/panel and (ii) see the **exact numeric values used to generate the plotted points/bars/summary statistics**, together with minimal metadata (units, n, and statistical tests).

## What to submit
### Per-figure Excel files (primary submission artifacts)
We provide **one Excel file per relevant figure**, placed in this directory:

- `SourceData_Fig1.xlsx`
- `SourceData_Fig2.xlsx`
- `SourceData_EDFig1.xlsx` (Extended Data)
- `SourceData_EDFig2.xlsx` (Extended Data)
- `SourceData_EDFig3.xlsx` (Extended Data)

**Inside each workbook**
- The first sheet is always named `README` and includes:
- The linked figure (e.g., “This file corresponds to Fig. 2.”)
- A panel-to-sheet mapping (e.g., Fig. 2a → `Fig2a_data`)
- Column definitions, units, sample/cluster identifiers (if any), and **n**

Data sheets contain the underlying values used for plotting, and optional `*_stats` sheets contain the corresponding test results.

## Repository context (non-submission helpers)
This repository also includes helper outputs used during figure generation:

- `subsampled_ids/` — fixed cell/spot subsets used for deterministic benchmarks and plots. 
- `figure_data/` — intermediate CSV snapshots used for figure assembly/debugging.

These helper folders are kept for reproducibility, but the **per-figure Excel workbooks** listed above are the intended “Source Data” deliverables for journal submission.
