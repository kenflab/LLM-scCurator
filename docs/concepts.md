# Concepts

LLM-scCurator prepares marker-list and lineage-context inputs for LLM-assisted
annotation. The workflow combines two sources of information from the same
expression data; their performance contributions must be evaluated separately.

## Marker curation

`curate_features()` ranks genes by differential expression and filters candidates
using expression abundance, detection fractions, effect sizes, global Gini
statistics, regex patterns and curated gene lists. These rules target programs
that can dominate an identity-oriented prompt, including ribosomal, mitochondrial,
stress and TCR/Ig signals. Such programs can also carry useful biological state
information; masking is a task-specific choice, not a declaration that a gene is
biologically irrelevant.

When HVG selection is enabled, predefined lineage genes and selected high-Gini
genes expand the candidate universe. Built-in whitelist protection and
proliferation sentinels preserve selected genes through masking. Default top-1
module rescue applies to LINC and hemoglobin modules and uses global mean
expression. It does not generally rescue one TCR/Ig gene per module.

Rescued candidates still pass through DE ranking and selection; rescue does not
guarantee inclusion in the final marker list. If expression filters or masking
remove every candidate, the method falls back to an earlier DE ranking and logs
a warning.

## Lineage-context supplementation

Separately from the final DE list, the workflow examines predefined lineage
markers in the full target expression matrix. It averages expression over target
cells and available markers within each panel, then selects the highest-scoring
lineage. A maximum score below 0.1 produces `Unknown/Mixed`.

`annotate(..., use_auto_context=True)` adds this inferred label to the prompt.
This allows shared identity genes outside the top-DEG list to influence context.
The current prompt receives the lineage label, not marker-level expression
statistics, negative evidence or same-cell marker coexpression. Panel coverage,
mixed clusters and expression scale can affect the heuristic.

An explicit `cell_type` argument overrides automatic context; optional `context`
entries provide user-specified notes. Record the source of each supplied label.
The inferred label is stored in `last_inferred_context`, so curate and annotate
each cluster together before moving to the next cluster.

## Global expression context and cross-lineage filtering

`set_global_context()` establishes expression distributions from the supplied
AnnData, optionally with balanced subsampling. This can be the dataset under
study; it need not be an external reference atlas.

When `coarse_col` is supplied, the cross-lineage filter compares target-cluster
expression percentiles with other labeled groups and removes candidates with
disproportionately high expression elsewhere. It requires appropriate labels in
both the global and local data. This is an expression-specificity heuristic, not
a guarantee against evaluation-label leakage. The provenance of those labels
must be documented, and held-out evaluation labels must not be used as inputs.

## Coarse-to-fine annotation

The optional discovery workflow performs coarse Leiden clustering and annotation,
then reclusters each coarse group and uses its predicted label as parent context.
The outputs are `major_type` and `fine_type`. These are two computational passes,
not a validated fixed ontology of compartment, lineage, subtype and state.

For interpretation, separate lineage/identity from state and retain the level of
resolution justified by the data. The current LLM prompt requests a single label
with state in parentheses; it does not produce separate fields for a five-level
hierarchy or continuous coordinates. `High`/`Medium`/`Low` confidence is the LLM's
self-report, and the exported numeric mapping is an ordinal code, not a probability.

See the [integrated workflow guide](integrated-workflow.md) for practical review
and the distinction between implemented and proposed evidence inputs.

## Stable outputs (table contract)

LLM-scCurator’s cluster-level annotation is often consumed outside Python notebooks
(e.g., by collaborators using spreadsheets). To keep downstream analyses stable across versions,
we treat the exported cluster table as a **public contract**.

### Core principles

- The cluster ID column is preserved as-is: `{cluster_col}` (e.g., `seurat_clusters`).
- `n_cells` is a reserved meta column (int).
- LLM-derived fields are namespaced by a prefix (default: `Curated`) using:
- `{prefix}_{FieldName}` where `FieldName` is **UpperCamelCase** (e.g., `CellType`).

### Minimum stable fields (v0.1.x)

These columns are guaranteed when exporting cluster annotations:

- `{prefix}_CellType` (string)
- `{prefix}_Confidence` (string; `High` / `Medium` / `Low`)
- `{prefix}_ConfidenceScore` (float; `High=2`, `Medium=1`, `Low=0`)
- `{prefix}_Reasoning` (string)
- `{prefix}_Genes` (string; `;`-separated)

### Forward-compatible extension fields

Depending on the backend and configuration, additional fields may be returned.
To remain forward-compatible, any extra keys are preserved and exported as
namespaced columns:

- `{prefix}_<UpperCamelCaseKey>`

Downstream code should rely on the minimum stable fields above and treat any
additional columns as optional.

## Optional Gini operations

Low-Gini suppression and high-Gini rescue are optional operations. The existing
`curate_features()` argument `disabled_components` can disable either one or both:
`disabled_components=("low_gini_suppression", "high_gini_rescue")`. Other
settings remain unchanged. This switch supports a component comparison; its
availability alone does not establish an annotation benefit.
