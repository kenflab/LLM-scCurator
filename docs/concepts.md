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
