# Prepare independent label review

The historical scorer can be executed reproducibly, but that does not establish
that its aliases and category definitions are biologically valid. This utility
prepares a review packet without changing the scorer, predictions, source data,
or historical results. It does not infer new labels, call an LLM, or generate
corrected performance estimates.

## Run

Use Python 3.11 or later, with no additional packages required:

```bash
python paper/revision_scoring/prepare_label_review.py \
  --outdir /absolute/path/outside-the-repository/label_review_v1
```

The default input is the committed
`paper/source_data/figure_data/Fig3/inputs/L4_prediction_audit.csv`.
The script verifies 250 available method–cluster records, 104 Standard/Curated
records, and 52 complete primary cluster pairs. It refuses an existing output
directory and requires output outside the repository. Without `--outdir`, it
creates `LLM_scCurator_label_review` beside the repository checkout. Use a new
directory for each review version; do not overwrite a reviewer’s completed work.

## Keep the three views separate

| Folder | Recipient and contents |
| --- | --- |
| `reviewer/` | Prediction-text reviewer: exact unique labels, stable hashed IDs, blank mapping fields and instructions. No method, condition, dataset, cluster, ground truth, score, occurrence count or codebook is joined to these rows. |
| `reference/` | Reference reviewer: the 52 cluster reference labels, blank adjudication fields, the complete historical scorer/configuration source, and its used reference categories. No method predictions or scores. |
| `private/audit/` | Analysis coordinator only: all 250 join keys, the separate 104-record Standard/Curated key, and the exact original audit CSV. |

Give only `reviewer/` to the prediction-text reviewer. `manifest.json` and the
private keys are coordinator records. Do not commit generated packets or share
the whole packet as a blinded review sheet. The original labels are public and
some may be familiar, so metadata removal is not a claim of perfect reviewer
blinding. Record prior exposure and the actual review procedure.

IDs depend on exact UTF-8 prediction text, not on method or row order. The same
text receives the same ID across methods and reruns. Unicode, capitalization and
punctuation variants retain separate rows for explicit synonym review; biological
punctuation is not silently discarded. In this frozen input, 250 records yield
104 unique exact raw strings. Do not confuse this number with the 104 primary
Standard/Curated records. The latter contain 67 unique exact strings.

## Freeze semantics before comparing methods

All new biological mapping fields are blank. Reviewers record compartment,
broad lineage, explicit T-cell identity, identity at supported resolution, state,
and unresolved interpretations with a rationale. Broad T identity and explicit
CD4/CD8 identity are separate. A generic T-cell label must not be silently filled
in as CD4 or CD8 from the evaluation dataset. An explicitly contradictory subtype
and a broad underspecified label should remain distinguishable. State agreement
is recorded separately from identity; multiple state programs need not be forced
into one biologically exclusive class.

Reference labels and the mapping of their aliases also require review. The
historical source copies document all old aliases and priorities; their presence
does not validate them. Resolve resident-memory treatment consistently on both
reference and prediction sides, review synonym disagreements, and identify
ambiguous reference labels before scoring. Do not choose mappings to favor a
method or change the reference labels after inspecting method performance.

Freeze and version the adjudicated prediction mapping, reference mapping, final
evaluation hierarchy and missing/ambiguous-label policy before joining the
private keys. Then apply the same rules to all available methods, retain the
historical scores, and report changes with reasons. Unassessed comparator entries
remain unavailable rather than being counted as errors. This preparation utility
deliberately does not decide or implement the new scoring rules.
