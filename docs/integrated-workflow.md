# Integrated workflow

LLM-scCurator combines marker-list curation with expression-derived lineage
context to support LLM-assisted annotation. The intended use is to prepare
interpretable inputs and proposed labels within a broader annotation workflow.

## Why two inputs?

Differential expression identifies genes that distinguish one cluster from the
chosen comparison group. A short top-DEG list can therefore omit identity genes
shared across related clusters. Marker curation changes which ranked candidates
are shown to the LLM. Lineage-context supplementation separately examines
predefined markers in the expression matrix and passes a broad inferred label.

These components address different limitations. An improvement from their
combination cannot automatically be attributed to marker curation alone. Their
value depends on the dataset, label resolution and LLM; inspect both successes
and failures.

## What is implemented?

| Input or operation | Current implementation | Interpretation |
| --- | --- | --- |
| Curated marker list | DE ranking, abundance/effect filters, global statistics, module masks and selected marker protection | Up to `n_top` gene symbols; the LLM does not receive the underlying expression table |
| Automatic lineage context | Mean expression of predefined lineage panels across target cells; highest mean wins | A heuristic broad label from observed expression, including genes outside the final DE list |
| Explicit parent label | `annotate(..., cell_type=...)` | User-supplied context overrides automatic inference; record its source |
| Additional notes | `annotate(..., context={...})` | Free-form user-supplied context, not a typed evidence-integration interface |
| Cross-lineage filtering | Optional `coarse_col` compares candidate expression across labeled groups | Requires appropriate label provenance; missing metadata skips this step |
| Coarse-to-fine mode | Coarse clustering/annotation followed by within-group reclustering/annotation | Predicted coarse labels provide context for fine calls; early errors can propagate |
| Output | `cell_type`, `confidence`, `reasoning`; exported cluster table and optional per-cell propagation | Proposed labels and LLM self-reported confidence for review |

The automatic lineage heuristic does not quantify marker coexpression within the
same cells, evaluate negative marker evidence, or return calibrated lineage
probabilities. A high cluster-level mean can arise from a subset of cells. Review
cell-level distributions when mixed populations, contamination or doublets could
explain the signal.

The current prompt calls an explicit `cell_type` value a **Known Parent Lineage**.
In coarse-to-fine mode that value is a model prediction, not independent ground
truth. Preserve this distinction in analysis logs and interpretation.

## A practical annotation workflow

1. Define clusters and inspect data quality, known-marker expression and top DEGs.
   Use the full available expression matrix when checking identity markers.
2. Generate curated marker lists and lineage context. Keep each cluster's
   `curate_features()` and `annotate()` calls together: automatic context is held
   in `last_inferred_context` and changes with the next curation call.
3. Cross-check proposed labels with a suitable pretrained classifier or reference
   atlas where available. Assess reference coverage and whether its label
   granularity matches the question. The package does not perform these checks
   or automatically combine their outputs.
4. Review marker combinations, detection fractions and expression distributions
   with dotplots and cell-level views. UMAP can guide inspection, but visual
   proximity alone does not establish identity.
5. Resolve discordant calls manually and document the evidence supporting the
   final label and its resolution. A broad supported identity is preferable to
   an unsupported specific label.

A T-cell label, for example, can be assessed using multiple CD3/TCR identity
markers together with the distribution of cytotoxic or activation genes. Cluster
averages alone do not show that all markers occur in the same cells. Low observed
expression is also affected by sampling and assay coverage, so absent transcripts
should not automatically be treated as definitive negative evidence.

## Describe identity and state separately

A useful interpretation scheme is shown below. It is a reporting convention for
human review, not the package's current output schema or a universal cell ontology.

| Reporting dimension | Question |
| --- | --- |
| Compartment | Which broad tissue compartment is supported? |
| Lineage | Which lineage is supported by the observed evidence? |
| Identity at supported resolution | How specifically can the cells be named without exceeding the evidence? |
| State | Which activation, cycling, stress or other state programs are supported? |
| Continuous coordinates / uncertainty | Is there evidence for a continuum, competing identity or unresolved boundary? |

State programs and continuous variation can cross identity categories. Do not
force them into a universal sequence of discrete types. The current output uses
one `cell_type` string, with state requested in parentheses, and a categorical
confidence self-report. It does not infer a trajectory or calibrated uncertainty.

## Keep evidence provenance explicit

Record how each context field was obtained: observed expression, a predicted
coarse label, independent reference mapping, manual review, or known study
metadata. In a benchmark, do not supply held-out reference labels as model inputs.
If a coarse label or lineage-restricted dataset is available to one comparator,
document that information advantage or match it across relevant conditions.

Keep the marker lists, actual prompts, model identifier, settings, raw responses,
cluster definitions and code release/commit with the results. Model-provided
reasoning is a proposed explanation, not proof that the expression data support
the claimed markers.

## Separate integrated and component evaluation

A matched design can separate the two inputs:

| Condition | Marker list | Lineage context |
| --- | --- | --- |
| Top-DEG baseline | Standard top DEGs | Absent |
| Marker-curation component | Curated markers | Absent |
| Lineage-context component | Standard top DEGs | Present |
| Integrated workflow | Curated markers | Present |

Use the same clusters, model/version, settings and scoring definitions across
these conditions. Report lineage errors and identity/state agreement separately
at declared resolutions. Repeat stochastic calls and summarize paired results at
the cluster level rather than treating repeated calls as independent clusters.
A component may help one task or model while contributing little to another.

The repository contains different evaluation designs. In
[`run_benchmarks.py`](https://github.com/kenflab/LLM-scCurator/blob/main/paper/scripts/run_benchmarks.py), Standard disables
automatic lineage context and Curated enables it, so the comparison changes both
marker preparation and context. In the
[HLCA validation](https://github.com/kenflab/LLM-scCurator/blob/main/paper/hlca_external_validation/README.md), `full_core` denotes
feature distillation under fixed tissue/dataset context, with `coarse_col=None`
and without automatic lineage-label injection. These interventions should not be
reported as interchangeable.

See [Evaluation scope](https://github.com/kenflab/LLM-scCurator/blob/main/paper/EVALUATION_SCOPE.md) for the exact saved conditions, values and current source files.

## Proposed extensions, not current functionality

A structured evidence interface could add marker-level expression summaries,
same-cell combinations, conflicting evidence, reference-neighbor composition,
reference-transfer uncertainty and separate identity/state fields. These inputs
would need clear provenance, definitions and matched validation. The current
package does not calculate or integrate them.

Neighbor graphs are currently used for clustering. Reference-neighbor class
proportions and transfer uncertainty would require a defined reference mapping
procedure; they cannot be inferred from UMAP coordinates alone. Any future
continuum or uncertainty output also needs its own validated definition.
