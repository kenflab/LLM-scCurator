# Record prompts and raw model responses

For new annotation runs, wrap an existing backend in `RecordingBackend` to preserve
its exact prompts and returned response text in an append-only JSONL file. The
annotation dictionary and inference settings are unchanged. Recording is opt-in.

```python
from pathlib import Path
from llm_sc_curator import LLMscCurator, RecordingBackend
from llm_sc_curator.backends import GeminiBackend

# Set api_key and model_name using the configuration for your experiment.
backend = GeminiBackend(api_key=api_key, model_name=model_name, temperature=0.0)
Path("results").mkdir(exist_ok=True)
recorded = RecordingBackend(
    backend,
    "results/annotation_calls.jsonl",
    metadata={"run_id": "run_01", "dataset_id": "cd8", "code_ref": "YOUR_COMMIT_SHA"},
)
curator = LLMscCurator(backend=recorded)

# Set these identifiers for each cluster/condition before the annotation call.
recorded.metadata.update(cluster_id="cluster_01", condition="standard", replicate=1)
result = curator.annotate(
    gene_list,
    context={"Tissue": "lung"},
    use_auto_context=False,
)
```

Each call writes a `call_start` event before inference and a `call_response` or
`call_error` event with the same `call_id`. The start event stores the exact prompt,
its SHA-256 digest, JSON-mode flag, configured model/settings, UTC time, and supplied
run identifiers. `raw_response_text` preserves the unmodified backend string,
including malformed JSON that triggers another annotation attempt. An unmatched
start event identifies an interrupted or incompletely recorded call. Parent
lineage and other context can be checked in the actual prompt for each condition.

The parent directory must exist. Use one log per process. If recording fails,
`ResponseAuditError` stops the `annotate()` call; it does not trigger another
model call within that call. An outer batch or hierarchical runner may handle
exceptions separately; inspect its failure records before restarting work.
Credentials are not read from the backend configuration, and exception messages
are not copied. Prompts and model text are recorded verbatim, so keep logs in a
location appropriate for their content.

These records capture **raw model-response text at the backend interface**, not the
complete provider SDK/HTTP payload. They do not capture retries hidden inside a
backend, provider-resolved model versions, token usage, or whether a provider
accepted a configured seed. A configured model alias is not a frozen version.
Existing historical responses cannot be recovered by enabling this wrapper, and
adding recording does not constitute a new benchmark or a correction of published
scores.
