# Concepts

LLM-scCurator improves robustness by standardizing **what you feed into the LLM**.

## Mask
Remove confounders that dominate marker lists but do not define identity:
- clonotype genes (TCR/Ig)
- ribosomal/mitochondrial programs
- generic stress / housekeeping signals

## Rescue
HVGs alone are not enough. LLM-scCurator can recover **rare, lineage-restricted markers**
using global statistics (e.g., high specificity) when they would otherwise be missed.

## Leakage (cross-lineage specificity)
Markers must be specific to the target lineage.
Leakage filters remove genes that are shared across unrelated lineages and cause semantic drift.

## Hierarchy
Separate the problem into two steps:
1) broad lineage call (coarse)
2) fine-grained subtype/state (fine)

This reduces hallucinations under ambiguous inputs and makes failure modes explicit.
