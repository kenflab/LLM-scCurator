# CD8 complementary-metric correction

The two L4/L5 analysis scripts previously computed Sanno with the public CD8
scorer but inferred major lineage separately from the predicted state. For
example, "Proliferating doublet (T+B)" became T_cell because its state was
Cycling. The public lineage parser instead returns Other.

Both scripts now use the public major-lineage parser and the public reference
mapping, with the same text normalization used by Sanno. The public aliases,
weights, penalties, marker lists, prompts, and saved predictions are unchanged.
The preparation script is included unchanged for reproducibility.

## Files

- make_l4_external_marker_noise_manual_o3_top10.py: corrected L4 evaluation.
- make_gptcelltype_shared_annotator_check.py: corrected L5 evaluation.
- prepare_l4_festem_markermap.py: original preparation script, unchanged.
- rescore_l4_l5_saved_predictions.py: offline replay from existing scored CSVs.
- tests/test_cd8_complementary_metrics.py: six regression tests for both scorers.

## Install and test

From the local LLM-scCurator repository, copy the four Python files into
paper/revision_metrics, this README into the same directory, and the regression
test into tests. Keep a backup of any existing local versions.

Use the analysis environment with numpy, pandas, scipy, and matplotlib.
No AnnData, MarkerMap, R, or LLM API access is needed for offline replay.

    python -m unittest discover -s tests -p test_cd8_complementary_metrics.py -v

## Recompute from the saved run directories

    tables_dir="/Users/kfurudate/Library/CloudStorage/OneDrive-InsideMDAnderson/LLM-scCurator/Submission_CM/R1/paper/revision_tables"
    python paper/revision_metrics/rescore_l4_l5_saved_predictions.py \
      --l4-dir "$tables_dir/l4_manual_o3_top10_final" \
      --l5-dir "$tables_dir/l5_reviewer1_low_cost" \
      --outdir "$tables_dir/cd8_complementary_metrics_public_20260928"

The output directory must be new. Source files are read only.
The default bootstrap settings are the original script defaults:
20,000 L4 resamples, 10,000 L5 resamples, seed 42.
If the original run used other bootstrap settings, pass the corresponding
--l4-bootstrap, --l5-bootstrap, and --seed values.

Required saved files:

- l4_manual_o3_top10_final/L4_shared_annotator_predictions_scored.csv
- l4_manual_o3_top10_final/L4_external_marker_input_diagnostics.csv
- l5_reviewer1_low_cost/L5_all_predictions_scored.csv

Use --skip-figures if the saved L4 diagnostics are unavailable.
Either analysis can be processed independently by omitting the other directory.
The replay checks the complete 17-cluster design, requires the original repeat
counts, and stops if Sanno, exact-state agreement, or hierarchy consistency
changes. It does not create marker lists, prompts, or new predictions.

## Outputs and manuscript updates

- L4/L4_shared_annotator_summary.csv: source for Table L15.
- L5/L5_summary.csv: source for Table L19 (also retains the auxiliary Gemini
  Standard condition, which the existing Figure L5 omits).
- L4/ReviewerFig_L4.pdf and .png: updated Figure L4 / Figure S4.
- L5/ReviewerFig_L5.pdf and .png: updated Figure L5 / Figure S6.
- L5/L22_native_GPTCelltype_predictions.csv: saved native GPTCelltype records
  with evaluation fields, to add alongside the marker inputs in Table L22.
- Each analysis also exports per-cluster metrics, rescored prediction records,
  score-based contrasts, and major_lineage_changes.csv.
- rescoring_metadata.json records source/code hashes and confirms zero new
  inference calls. It records the rescoring date, not a new inference date.

Update the affected table values and figure panels from these outputs.
LetterTables.xlsx and manuscript documents are not edited automatically.
Tables L18/L23 can reference the rescoring metadata without changing the
original inference dates or original input-file hashes.

## Verification using the supplied records

| Condition | Old major-lineage accuracy | Public parser |
| --- | --- | --- |
| COSG (L4) | 100.0% | 98.8% (84/85) |
| Standard DEG + CoT (L5) | 95.3% | 94.1% (80/85) |
| regex_mask + CoT (L5) | 98.8% | 96.5% (82/85) |
| Reused Gemini Standard (auxiliary L5 condition) | 94.1% | 82.4% (14/17) |

All 629 supplied records (425 L4, 170 new L5 manual, and 34 reused Gemini)
retained identical Sanno, exact-state agreement, hierarchy consistency, and raw
predictions. Six major-lineage match flags changed. The reused Gemini Full
pipeline retained 94.1% major-lineage accuracy.

Six regression tests passed. The offline L4 replay was also executed
successfully using the supplied predictions.

The native GPTCelltype prediction records were not supplied. They were not
fabricated or re-inferred, and the full L5 replay/figure was not verified here.
The replay uses the existing local L5_all_predictions_scored.csv to evaluate
that condition when run on the original machine.

## MarkerMap provenance

The supplied preparation script selects a global MarkerMap panel and allocates
its genes to clusters using the Festem AllocateMarker A/B rule. Therefore,
describe those inputs as "MarkerMap-selected genes allocated to clusters using
Festem::AllocateMarker", not native cluster-specific MarkerMap outputs.
This correction does not rerun or alter that preparation.

## Suggested pull request

Title: Use public CD8 parsing for complementary metrics

Description:

Use the public CD8 lineage parser and reference mapping in the L4/L5 evaluation
scripts. Add offline rescoring of saved predictions, regression tests, and the
marker-preparation script. Saved model predictions and Sanno are unchanged.
