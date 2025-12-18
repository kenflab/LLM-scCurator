# source_data/

This directory contains the **source data files** supporting the figures in the LLM-scCurator manuscript.

Our goal is that a reader can (i) identify which file supports each figure/panel and (ii) see the **exact numeric values used to generate the plotted points/bars/summary**, together with minimal metadata (units and n, where applicable).

Panel mapping
For a panel-to-file index, see [`../FIGURE_MAP.csv`](../FIGURE_MAP.csv).

## What to submit
### Per-figure Excel files (primary submission artifacts)
We provide **one Excel file per relevant figure**, placed in this directory:

- [`SourceData_Fig1.xlsx`](SourceData_Fig1.xlsx)
- [`SourceData_Fig2.xlsx`](SourceData_Fig2.xlsx)
- [`SourceData_EDFig1.xlsx`](SourceData_EDFig1.xlsx) (Extended Data)
- [`SourceData_EDFig2.xlsx`](SourceData_EDFig2.xlsx) (Extended Data)
- [`SourceData_EDFig3.xlsx`](SourceData_EDFig3.xlsx) (Extended Data)

**Inside each workbook**
- The first sheet is always named `README` and includes:
- The linked figure (e.g., “This file corresponds to Fig. 2.”)
- A panel-to-sheet mapping (e.g., Fig. 2a → `Fig2a_data`)
- Column definitions, units, sample/cluster identifiers (if any), and **n**

## Repository context (non-submission helpers)
This repository also includes helper outputs used during figure generation:

- [`subsampled_ids/`](subsampled_ids) — fixed cell/spot subsets used for deterministic benchmarks and plots. 
- [`figure_data/`](figure_data) — panel-level CSV inputs used for plotting and traceability (including a small number of panel-specific scored tables).

Extended Data Fig. 2 note:
- [`figure_data/EDFig2/EDFig2a_data.csv`](figure_data/EDFig2/EDFig2a_data.csv), [`figure_data/EDFig2/EDFig2b_data.csv`](figure_data/EDFig2/EDFig2b_data.csv), and [`figure_data/EDFig2/EDFig2c_data.csv`](figure_data/EDFig2/EDFig2c_data.csv) are per-cluster scored tables used to compute confusion matrices.
- Confusion-matrix values are computed by aggregating these scored tables at plot time (no pre-aggregated confusion CSV is stored).

These helper folders are kept for reproducibility, but the **per-figure Excel workbooks** listed above are the intended “Source Data” deliverables for journal submission.
