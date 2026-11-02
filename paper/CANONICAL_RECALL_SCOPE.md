# CD8 canonical marker recall

Figure 2c and the component-removal analysis underlying Table S5 use the same
eight-state CD8 reference dictionary and the same per-cluster recall definition:

`recall = |selected genes ∩ state reference ∩ input feature space| / |state reference ∩ input feature space|`.

The denominator is the number of eligible state-reference genes, not the marker-list
length. A cluster with no eligible reference genes has undefined recall (`NaN`);
the summary uses the unweighted mean over clusters with defined recall. The
reported percentage is that mean multiplied by 100. This measures retention of
the specified state-marker evidence, not annotation accuracy or comprehensive
preservation of cell identity.

The analyses ask different questions. Figure 2c compares the `standard`,
`filter_only`, `regex_mask`, and `full_core` configurations at list lengths of
10, 20, 30, and 50 genes. Table S5 compares removal of individual components
from a separate production `full_core` run with a top-50 budget. Reference
categories enter the component analysis after marker lists have been selected.

The sentinel-retention operation can protect lineage genes outside this
state-reference dictionary. In particular, CD3D, CD3E, CD3G, CD8A, and CD8B are
absent from all eight sets (45 distinct genes in total). Removing these genes
can change a selected list without changing its state-marker recall. An
unchanged recall therefore does not establish that the removed genes were
biologically uninformative or that a component has no annotation effect.

The committed Figure 2c `full_core` value at 50 genes is approximately 53.68%,
whereas Table S5 reports 52.9%. The same metric definition does not make these
separate runs interchangeable. Reconciliation requires the original selected
lists and run provenance for Table S5; this source audit neither reconstructs
those missing records nor changes either reported result.

## Sources and reproducible check

- [Figure 2 notebook](notebooks/04_Fig2a_d_FigS1.ipynb): zero-based code-cell indices
  7 (`CD8_MARKER_DB`), 8 (`marker_recall_at_n`), 11 (per-cluster evaluation), and
  12 (unweighted averaging and export).
- [Component analysis](revision_metrics/make_l9_l10_component_ablation.py):
  `CD8_MARKER_DB`, `evaluate_lists`, and `summarize`.
- [Committed Figure 2 summary](source_data/figure_data/Fig2/Fig2a_c_data.csv).

From the repository root, run:

```bash
python paper/scripts/audit_canonical_recall_scope.py
```

The standard-library-only audit reads these sources without executing notebook
cells or importing analysis libraries. It checks the shared dictionary and
per-cluster formula, reports reference-set counts and the cited lineage-gene
exclusions, and prints committed Figure 2 values and source hashes. It writes
no files, makes no model calls, and does not rescore annotations.
