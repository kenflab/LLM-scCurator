# HLCA core prospective external validation

This directory contains the prospective, separately analyzed HLCA core validation used to
address dataset-scope concerns. It compares only two marker inputs under an otherwise identical
LLM setting:

- `standard`: uncurated Wilcoxon DEG top 50
- `full_core`: `LLMscCurator.curate_features(..., use_statistics=True, use_hvg=True)` up to 50 genes

The ground truth is never sent to Gemini. All evaluable author-provided `leiden_3` clusters are
retained; there is no selection by label purity, donor count, source-dataset count, or result.

## Fixed design

- Dataset: HLCA core v1.0, 584,944 cells, CELLxGENE dataset `066943a2-fdac-4b29-b348-40cede398e4e`
- Pinned CELLxGENE Census LTS release: `2025-11-08`
- Evaluation unit: author-provided `leiden_3`
- Ground truth: majority `ann_level_3`; its majority parent path supplies levels 1 and 2
- Sampling: fixed seed 42, up to 300 cells per evaluable cluster
- Expression: `adata.raw.X` raw counts, normalize-total to 10,000, then `log1p`
- LLM: `gemini-2.5-pro`, temperature 0, identical healthy-human-lung context for both inputs
- Repeats: three successful inference calls per cluster and condition
- Statistical unit: cluster; calls are averaged within cluster before paired analysis
- Wilcoxon input: paired differences rounded to 12 decimal places to preserve theoretical ties
  across CSV round trips

The HLCA hierarchical score retains the manuscript's lineage/state structure while using the
official three-level HLCA path:

`Sanno_HLCA = 0.7 × I(level 1 match) + 0.3 × state agreement`, where state agreement is 1 for
a level-3 match, 0.5 for a level-2-only match, and 0 otherwise. Exact level-3 agreement and major
lineage accuracy are also reported separately.

## Why prediction mapping is a separate step

Gemini returns free-text cell-type names, whereas HLCA uses fixed expert labels. The pipeline does
not ask another LLM to judge those outputs. Instead it creates a blinded mapping sheet containing
only each unique prediction and the official hierarchy codebook—no condition, cluster, or ground
truth. Normalized exact label matches are filled automatically; biological synonyms must be mapped
manually before scoring. This prevents a hidden, outcome-aware evaluation step.

## New-computer setup

Run from the repository root with Python 3.11 or 3.12:

```bash
python -m venv .venv-hlca
source .venv-hlca/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[hlca]"
```

The download is approximately 5.87 GB. Allow additional space for the sampled intermediate and
results. The code opens the source H5AD in backed mode and loads only the fixed per-cluster sample
into memory.

## Run in order

```bash
python paper/hlca_external_validation/run_hlca_validation.py download
python paper/hlca_external_validation/run_hlca_validation.py prepare
```

Inspect prompts without making an API call:

```bash
python paper/hlca_external_validation/run_hlca_validation.py infer --dry-run --limit 2
```

Set the API key only in the environment. It is never saved:

```bash
export GEMINI_API_KEY="YOUR_KEY"
python paper/hlca_external_validation/run_hlca_validation.py infer
```

Each call is saved immediately as a self-contained JSON file under
`work/results/raw_responses/`, including the exact prompt, genes, unparsed response, parsed fields,
requested and returned model identifiers, timestamps, and usage metadata. Re-running `infer`
resumes incomplete work without replacing successful calls.

Create the blinded mapping table:

```bash
python paper/hlca_external_validation/run_hlca_validation.py mapping
```

Review `work/results/prediction_mapping.csv` without viewing condition, cluster, or ground truth.
Resolve every row using one of the following rules:

- Exact or defensible level-3 synonym: enter one `path_id` from
  `work/intermediate/hlca_label_hierarchy.csv` in `mapped_path_id`. The three mapped-label cells may
  remain blank because the scorer retrieves the canonical labels from that path.
- Prediction supports only a major compartment: enter `LEVEL_1` in `mapped_path_id`, enter the
  official compartment in `mapped_ann_level_1`, and leave levels 2 and 3 blank.
- Prediction supports level 2 but not a unique level 3: enter `LEVEL_2` in `mapped_path_id`, enter
  the official labels in `mapped_ann_level_1` and `mapped_ann_level_2`, and leave level 3 blank.
- No defensible atlas mapping (for example, an uninterpretable label or doublet): enter `UNMAPPED`
  and leave all three mapped-label cells blank. It receives zero for all three metrics.

Rows sharing a `prediction_id` are capitalization or punctuation variants and must receive the same
mapping. Record the rationale for partial or unmapped decisions in `mapping_note`. Do not force a
broad prediction to the nearest level-3 label, do not add condition or ground-truth columns to the
mapping sheet, and do not use another LLM to adjudicate the predictions. Preserve the human-reviewed
file as Source Data.

After all rows are resolved:

```bash
python paper/hlca_external_validation/run_hlca_validation.py score
python paper/hlca_external_validation/run_hlca_validation.py status
```

Regenerate the publication figures later without making any Gemini calls or changing the reviewed
mapping and score tables:

```bash
python paper/hlca_external_validation/run_hlca_validation.py figures
```

This also computes a visualization-only UMAP of the fixed sampled cells and caches its coordinates.
Use `figures --skip-umap` when only the two score panels need to be redrawn. The UMAP does not enter
marker selection, prediction, mapping, or scoring.

## Outputs

- `cluster_manifest.csv`: every cluster, GT path, purity, cell/donor/dataset counts, inclusion audit
- `marker_lists.csv`: exact Standard and `full_core` gene inputs
- `raw_responses/*.json`: prompt and raw/parsed Gemini response for every call
- `prediction_mapping.csv`: reviewed blinded prediction mapping
- `call_level_scores.csv`: complete call-level audit table
- `cluster_level_scores.csv`: three calls averaged within cluster
- `paired_summary.csv`: means, paired difference, cluster-bootstrap 95% CI, Wilcoxon P, and counts
- `Fig3d_HLCA_external_validation.{pdf,png}`: paired mean differences with 95% bootstrap CIs
- `FigS6_HLCA_metrics.{pdf,png}`: absolute exact, lineage, and Sanno summaries
- `FigS7_HLCA_scope.{pdf,png}`: sampled-cell UMAPs by evaluation compartment and level-3 ground truth
- `hlca_umap_coordinates.csv`: cached visualization-only coordinates used for Figure S7
- `run_metadata.json`: config, package versions, platform, and returned model version(s)

The `work/` directory is intentionally git-ignored. Add only reviewed manuscript Source Data to the
repository after the run is frozen.

## Data provenance

- [HCA HLCA v1.0 atlas page](https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0)
- [HLCA official repository and metadata dictionary](https://github.com/LungCellAtlas/HLCA)
- [CELLxGENE source-H5AD download documentation](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_datasets.html)
- [Sikkema et al., Nature Medicine (2023)](https://www.nature.com/articles/s41591-023-02327-2)

The fixed S3 URL, expected byte count, ETag, cell count, all analysis parameters, and model name are
recorded in `config.toml`. The source H5AD was checked directly to contain `raw/X`,
`raw/var/feature_name`, `leiden_3`, and `ann_level_1–3`. As of the date this workflow was added,
Google listed `gemini-2.5-pro` as available; provider availability can change, so the prospective
run should be completed and archived promptly.
