# LLM-scCurator

LLM-scCurator is a data-centric framework that stabilizes zero-shot cell-type annotation by suppressing biological confounders in marker lists (clonotype, housekeeping, stress, ribosomal/mitochondrial, cell-cycle) before LLM inference.

## Quick links
- Getting started → install + minimal example
- User guide → Scanpy / Seurat / Spatial workflows + paper reproducibility
- Concepts → masking, rescue, leakage filter, hierarchical inference
- API reference → public API

## Privacy (one paragraph)
By default, LLM-scCurator operates on locally computed marker summaries. If you use an external LLM API, only compact, cluster-level inputs are sent (no raw matrices).
