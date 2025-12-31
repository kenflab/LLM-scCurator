# Concepts

LLM-scCurator improves zero-shot annotation robustness by standardizing **what you feed into the LLM**:
a compact, noise-aware marker summary with explicit safeguards.

## Mask (noise modules)

Suppress programs that frequently dominate naive marker rankings but do not define identity:

- **Clonotypes:** TCR/Ig variable/constant regions
- **Ubiquitous programs:** ribosomal/mitochondrial, translation factors
- **Generic responses:** stress / housekeeping signatures

## Rescue (high-specificity recovery)

HVGs alone can miss low-to-moderate, lineage-restricted markers—especially in rare populations.
LLM-scCurator can **rescue informative identity genes** using global statistics (e.g., high specificity / high-Gini)
while avoiding globally high-mean ubiquitous genes.

## Leakage filter (cross-lineage specificity)

Markers should be specific to the target lineage. The leakage filter removes genes that are
widely expressed across unrelated lineages, reducing semantic drift and off-lineage calls.

## Hierarchy (coarse → fine)

LLM-scCurator separates annotation into two explicit steps:

1. **Coarse lineage call** (major cell class)
2. **Fine-grained subtype/state** within that lineage

This structure reduces hallucinations under ambiguous inputs and makes failure modes easier to diagnose.

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
