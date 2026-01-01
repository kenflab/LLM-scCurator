# API reference

## Core
::: llm_sc_curator.core.LLMscCurator
    options:
      members:
        - __init__
        - set_global_context
        - set_global_context_spatial
        - curate_features
        - annotate
        - run_hierarchical_discovery
      inherited_members: false
      show_if_no_docstring: false

## Backends
::: llm_sc_curator.backends
    options:
      members:
        - BaseLLMBackend
        - GeminiBackend
        - OpenAIBackend
        - OllamaBackend
        - LocalLLMBackend
      show_source: false
      show_if_no_docstring: false

## Masking
::: llm_sc_curator.masking.FeatureDistiller
    options:
      members:
        - __init__
        - calculate_gene_stats
        - detect_biological_noise
        - calculate_lineage_specificity
      show_source: false
      show_if_no_docstring: false
      
## Utils
Helpers for converting per-cluster LLM outputs into tidy tables (CSV/DataFrame) and per-cell labels.

### Output table contract
`export_cluster_annotation_table()` produces a cluster summary table intended to be stable across versions.

Required columns:
- `{cluster_col}` (e.g., `seurat_clusters`)
- `n_cells`
- `{prefix}_CellType`, `{prefix}_Confidence`, `{prefix}_ConfidenceScore`, `{prefix}_Reasoning`, `{prefix}_Genes`

Extra keys returned by LLM backends may be exported as `{prefix}_<UpperCamelCaseKey>` columns.

::: llm_sc_curator.utils
    options:
      members:
        - ensure_json_result
        - export_cluster_annotation_table
        - apply_cluster_map_to_cells
        - harmonize_labels
      show_source: false
      show_if_no_docstring: false

## Noise modules
::: llm_sc_curator.noise_lists
    options:
      show_source: false
