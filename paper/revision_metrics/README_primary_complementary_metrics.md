# Primary benchmark complementary metrics

`make_l4_l5_letter_tables.py` generates Table L4 (complementary metrics) and
Table L5 (weight sensitivity) from the saved predictions in `L2_per_cluster_audit`.
`make_l4_l5_review_figures.py` generates Figure L1a–d from those tables.
These scripts evaluate the original 52-cluster benchmark, separately from the
manual o3, local Ollama, native GPTCelltype, and HLCA analyses.

The table script parses the saved labels with the public CD8, CD4, CAF, and mouse
B-cell configurations. It checks every available default Sanno against the
recorded score before writing results. It retains all 52 predefined clusters,
including B_Other; `UsedInConfusion` does not control quantitative inclusion.
An unavailable comparator score is not counted as a failed prediction.

Exact state agreement compares the parsed state categories, independent of major
lineage. Exact-match accuracy requires both lineage and state to agree. The
previous script inferred these components from Sanno, which loses information
when a forbidden-lineage penalty sets the score to zero. For example,
"Cytotoxic NK cell" in CD8.c09.Tk.KIR2DL4 maps to NK/EffMem against T/EffMem:
the state agrees, the lineage does not, and Sanno remains zero.

Weight sensitivity changes only the lineage/state weights in the public scorer;
aliases, near-lineage rules, and penalties are retained. Therefore state-component-only
scores can differ from exact state agreement, and lineage-component-only scores
can differ from strict major-lineage accuracy. Existing CSV column names and
weighting-scheme identifiers are retained for compatibility; figure labels use
"Hierarchy-consistent accuracy" and "component only".

## Run from the repository root

Use the workbook containing the current Table L2 audit. No LLM calls are made.
The scripts require numpy, pandas, scipy, openpyxl, and matplotlib.

```bash
python paper/revision_metrics/make_l4_l5_letter_tables.py \
  --in-xlsx /path/to/LetterTables.xlsx \
  --csv-outdir /path/to/primary_metrics_public

python paper/revision_metrics/make_l4_l5_review_figures.py \
  --csv-dir /path/to/primary_metrics_public \
  --outdir /path/to/primary_metrics_public/figures

python -m unittest discover -s tests -p 'test_primary_complementary_metrics.py'
```

The first command writes full-precision L4/L5 CSVs, a prediction-level evaluation
audit, and metadata with input/code hashes. The second writes four PDF/PNG panels
named `ReviewerFig_L1a` through `ReviewerFig_L1d`.

Optionally add `--out-xlsx /path/to/LetterTables_updated.xlsx` to the first command
to copy the source workbook and replace its L4/L5 sheets. The output workbook
must differ from the input. `--round` controls workbook precision (default: 4
decimal places); CSV files retain full precision and are preferred for plotting.

Verification with `LetterTables_R1_HLCA.xlsx` reproduced all 250 available default
Sanno values, including the 104 Standard/Full pipeline scores. Standard exact
state agreement changed from 34/52 (65.4%) to 35/52 (67.3%); Full pipeline remained
32/52 (61.5%). Existing Table L5 values were unchanged within numerical tolerance.

## Figure 3a–c

Use `paper/scripts/make_figure3.py` with the full-precision
`L4_complementary_metrics.csv` and `L4_prediction_audit.csv` from the same run.
See [the paper README](../README.md#current-figure-3abc) for the command and outputs.
`06_Fig3_minimal.ipynb` is a wrapper for the script, not a separate evaluator.
Figure 3a uses 20,000 bootstrap resamples (seed 42); Table L5 retains its
separately documented 10,000 resamples. Missing comparators are NA.
