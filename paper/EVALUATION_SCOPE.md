# What the saved evaluations measure

This document describes the evaluation artifacts in this repository snapshot. It does not change saved predictions, method keys, scores, or historical figures. An integrated workflow comparison and a marker-input comparison answer different questions; neither alone identifies the contribution of every component.

The audit on 2 November 2026 used source and saved results from commit `125777c19995850ed63ddd3a814ef74e08c1d196`.

## Evaluation map

| Evaluation | Inputs and context | What it supports | What it does not establish |
| --- | --- | --- | --- |
| Main Figure 3, CD8/CD4/MSC | Standard top DEGs versus curated markers with automatic expression-derived lineage context enabled in the accompanying benchmark notebooks | Descriptive performance of the implemented annotation workflows on the retained clusters | An isolated causal effect of marker curation; uniform superiority over reference methods; generalization to all cells/tissues |
| Main Figure 3, mouse B | Standard and curated markers both receive explicit `cell_type="Mouse B cell"` in the notebook | Marker-input comparison conditional on a supplied broad label | Independent recovery of B-cell lineage or validation of automatic lineage inference |
| Figure 2e–f, CD8 stepwise | `standard`, `filter_only`, `regex_mask` have `Use_Auto_Context=False`; all `full_core` rows have `Use_Auto_Context=True` | Descriptive comparison of the four saved conditions | A feature-only ablation, despite the historical `full_core` key; a 2×2 factorial experiment |
| HLCA external validation | Standard top 50 versus curated up-to-50 markers; both use the same healthy-human-lung prompt context, without inferred parent-lineage injection | External test of curated marker inputs under fixed tissue context with Gemini 2.5 Pro | External validation of the full automatic-lineage-context workflow; proof of curation benefit on HLCA |
| Revision component-ablation code | Production marker selection with one component disabled at a time; no LLM inference | A runnable design for examining marker-list properties | Completed annotation-accuracy evidence merely because the script exists |
| Revision GPTCelltype comparison code | Native and reused workflows differ in backend, prompt, context, and inputs | Descriptive workflow comparisons; separately defined matched prompt controls | Matched-backend superiority of the full workflow |

The scripts for revision analyses are present, but their external `LetterTables.xlsx`, expression matrices, and many generated revision result tables are not part of this snapshot. Do not present those outputs as newly verified from this repository.

## Main Figure 3: historical values under the original mapping

Source: [`L4_complementary_metrics.csv`](source_data/figure_data/Fig3/inputs/L4_complementary_metrics.csv), backed by [`L4_prediction_audit.csv`](source_data/figure_data/Fig3/inputs/L4_prediction_audit.csv). There are 52 clusters and 250 available method–cluster predictions. Mouse B CellTypist and Azimuth results are unassessed, not zero.

All values below are fractions. `Sanno` is a task-specific weighted hierarchical score, not exact-label accuracy. `Exact_Match_Accuracy` requires the parsed major lineage and state to match. `Exact_State_Agreement` is a separate metric and may match even when lineage does not. These historical mappings have limitations described below; exact agreement under this parser should not be equated with independently adjudicated biological identity.

| Dataset | Clusters | Standard Sanno | Curated Sanno | Standard exact match | Curated exact match | Standard major lineage | Curated major lineage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CD8 T | 17 | 0.752941 | 0.817647 | 0.588235 | 0.529412 | 0.823529 | 0.941176 |
| CD4 T | 22 | 0.750000 | 0.850000 | 0.545455 | 0.500000 | 0.818182 | 1.000000 |
| MSC | 8 | 0.806250 | 0.893750 | 0.625000 | 0.750000 | 0.875000 | 0.875000 |
| Mouse B | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The T-cell results illustrate why lineage and finer identity/state outcomes must be reported separately: higher mean Sanno and lineage agreement coexist with lower exact-match agreement. These are observations under the original mapping, not evidence of universally improved or independently validated cell-identity resolution.

The full workflow is not best on every comparison. For example, CD8 mean Sanno is 0.858824 for SingleR versus 0.817647 for Curated; CD4 mean Sanno is 0.863636 for both CellTypist and Azimuth versus 0.850000 for Curated. Reference compatibility and the resolution of reference labels must be considered when interpreting these comparisons.

The accompanying notebooks select `models/gemini-2.5-pro`, and their stored initialization output reports that model. Unlike the HLCA archive, the Fig3 audit CSV does not itself retain exact prompts, returned model versions, and timestamps for every prediction. The notebook settings and saved prediction table should therefore be described separately from a fully linked per-call provenance record.

### Current and historical Figure 3 sources

Use `source_data/figure_data/Fig3/Fig3abc_data.csv`, its panel-specific CSVs, and the two `source_data/figure_data/Fig3/inputs/L4_*.csv` files above for the current Figure 3. `source_data/SourceData_Fig3.xlsx` and `source_data/figure_data/Fig3/Fig3a_c_data.csv` are historical artifacts with some different complementary metrics. For example, the historical CD8 Standard major-lineage value is 0.941176, whereas the current strict-label audit gives 0.823529. Preserve the historical files for provenance; do not mix them into the current manuscript tables. Any replacement workbook should be regenerated explicitly from the same current prediction audit.

### Label-mapping validity requires a separate review

The original parser groups CD4 and CD8 predictions into the same broad T lineage. It therefore cannot assess CD4-versus-CD8 identity errors. In the CD4 task, six Standard predictions and six Curated predictions explicitly name CD8 cells; two predictions in each condition nevertheless receive a score of 1 under the original broad-lineage/state mapping. This affects both conditions and must be addressed consistently across all methods.

The parser and reference mappings also require review for synonym coverage and resolution. For example, `CD8.c10.Trm.ZNF683` is harmonized to `EffMem`, while its `Tissue-Resident Memory T cell` predictions map to `Resident`. The Curated prediction `T follicular regulatory (Tfr) cell` for `CD8.c14.Tex.TCF7` does not match the CD8 parser's T-lineage aliases. Substring matching creates additional risks, such as interpreting a gene symbol containing `dc` as dendritic evidence. Resolve these issues using a frozen reviewed taxonomy, rather than changing individual rules after inspecting which condition benefits.

Keep three questions separate: (1) does the implementation reproduce its saved score, (2) does the mapping represent the intended biological resolution, and (3) does the method improve independent outcomes? The existing numerical audit addresses the first question only. A condition-blinded prediction-mapping preparation workflow is provided in [revision_scoring](revision_scoring/README.md). Its sheets are unadjudicated inputs to review, not corrected results. Freeze reference-label decisions separately and apply the same reviewed mapping to all 250 method records, including the 104 Standard/Curated records.

Within the 104 Standard/Curated records, the original pooled mean scores are 0.783654 and 0.860577; 11 paired clusters improve, 32 tie and 9 worsen. Seven T-cell clusters change from a non-T Standard label to a T label under Curated and account for a summed score difference of +5.8, while the other 45 clusters sum to -1.8. This pattern motivates matched-context controls; it does not establish that context causally explains the whole difference.

For local-model experiments, also align the scoring target with the prompt's output contract. A prompt requesting only a cell-state label under supplied CD8 context should not be treated as an unconstrained full-identity prediction without an explicit, prespecified rule. Historical and revised scores must be reported together when rescoring changes conclusions.

## Historical Figure 2 condition name

Sources: [`Fig2e_data.csv`](source_data/figure_data/Fig2/Fig2e_data.csv), [`Fig2f_data.csv`](source_data/figure_data/Fig2/Fig2f_data.csv), and [`05_Fig2e_f_StepwiseAnalyses.ipynb`](notebooks/05_Fig2e_f_StepwiseAnalyses.ipynb).

| Saved key | Clusters | Auto context in saved CSV | Mean Sanno |
| --- | ---: | --- | ---: |
| `standard` | 17 | False | 0.752941 |
| `filter_only` | 17 | False | 0.729412 |
| `regex_mask` | 17 | False | 0.729412 |
| `full_core` | 17 | True | 0.811765 |

Retain these keys for reproducibility, but describe this historical `full_core` condition as **curated markers plus automatic lineage context**. It is distinct from HLCA's `full_core` condition, which does not inject an inferred parent lineage. Figure 2's 0.811765 and Figure 3's 0.817647 are separate saved results and must not be silently interchanged.

## HLCA: fixed-context marker-input comparison

Sources: [`HLCA design`](hlca_external_validation/README.md), [`paired_summary.csv`](hlca_external_validation/source_data/paired_summary.csv), [`call_level_scores.csv`](hlca_external_validation/source_data/call_level_scores.csv), and the 546 archived JSON responses under `hlca_external_validation/source_data/raw_responses/`.

The design uses 91 author-provided clusters, two conditions, and three successful calls per cluster and condition. Calls are averaged within cluster before paired analysis. Every saved response reports requested and returned model `gemini-2.5-pro`. The ground-truth path is the majority official level-3 label and its parent path; it is not supplied to the LLM. Prediction mapping is documented as a condition/cluster/ground-truth-blinded human review step, with codebook and reviewed mapping retained. Presence of that protocol and files does not independently prove that a reviewer never saw outcomes.

| Metric | Standard | Full core | Paired difference | Saved 95% bootstrap CI | Saved two-sided Wilcoxon P |
| --- | ---: | ---: | ---: | --- | ---: |
| Sanno HLCA | 0.957143 | 0.955861 | −0.001282 | [−0.020147, 0.015934] | 0.633080 |
| Exact level-3 agreement | 0.743590 | 0.776557 | 0.032967 | [−0.043956, 0.113553] | 0.386105 |
| Major-lineage agreement | 1.000000 | 0.992674 | −0.007326 | [−0.018315, 0.000000] | 0.157299 |

`Sanno_HLCA = 0.7 × I(level-1 match) + 0.3 × state agreement`, where state agreement is 1 for level-3 match, 0.5 for a level-2-only match, and 0 otherwise. These results do not demonstrate a reproducible improvement from curation on HLCA. They also do not test the added value of automatic parent-lineage context. UMAP is visualization-only in this pipeline and does not enter annotation or scoring.

## Context provenance and component attribution

Automatic context in the current implementation comes from canonical-marker expression in the target cluster, stored after feature curation and optionally added to the prompt. It is not an atlas-neighbor composition, reference-label transfer, or validated uncertainty probability. Explicit `cell_type` or tissue context is a different source and should be recorded as such. The curation routine can also retain/rescue lineage markers; a prompt-context toggle alone does not remove every lineage-aware operation from marker selection.

No complete four-arm, matched 2×2 result set was found among the saved benchmark tables in this snapshot. A future experiment should define and freeze these arms before inference:

| Arm | Marker input | Parent-lineage prompt context |
| --- | --- | --- |
| A | Standard top DEGs | Absent |
| B | Curated list | Absent |
| C | Standard top DEGs | Present |
| D | Curated list | Present |

Derive the same context for C and D from the same expression data, independently of reference outcomes. Fix model, prompt template, tissue context, gene budget, cluster set, and inference-repeat policy. Preserve exact prompts, marker lists, context evidence/provenance, requested/returned model identifiers, raw responses, and scorer version. Specify that B–A is the effect of the **defined curation bundle**, including any marker rescue retained in that bundle; D–B is the effect of adding prompt context; D–C is curation conditional on context. The interaction is `(D−C)−(B−A)`. A simple no-LLM lineage-evidence baseline and an equally informed LLM comparator are needed to assess whether the LLM/integration contributes beyond the supplied evidence.

Use clusters as the paired unit for within-dataset comparisons, average repeated calls within cluster, and report donor/dataset-level generalization separately. Report lineage agreement, supported-resolution identity agreement, state agreement, and error/abstention behavior separately; do not retrospectively redefine the historical scores. Evaluate false or conflicting context as well as informative context, because added context can propagate errors.

## Verification performed on this snapshot

A read-only audit recomputed all 250 main-Figure-3 Sanno values from saved raw label strings with the repository's public hierarchical scorer; every value matched `Public_Sanno` exactly. The 546 HLCA calls contain no duplicate cluster/condition/repeat keys and exactly three calls in each of 182 cluster–condition groups. Re-aggregation of call-level means reproduced all three committed HLCA condition means. No new LLM calls or expression-based marker selection were performed for this audit.

Run the saved-score and aggregation checks from the repository root:

```bash
python paper/scripts/audit_evaluation_scope.py .
```

This script checks saved tables and counts archived responses. It does not rerun expression processing, validate the biological reference labels, or independently reproduce the confidence intervals.
