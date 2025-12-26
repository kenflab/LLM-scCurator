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

## Noise modules
::: llm_sc_curator.noise_lists
    options:
      show_source: false
