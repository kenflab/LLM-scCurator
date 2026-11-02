#!/usr/bin/env python3
"""Audit the saved CD8 state-marker recall definition without executing notebooks.

Reads source and a committed summary CSV only. Does not select genes, score
annotations, import the analysis environment, or call an LLM. JSON goes to stdout.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys


NOTEBOOK = Path("paper/notebooks/04_Fig2a_d_FigS1.ipynb")
ABLATION = Path("paper/revision_metrics/make_l9_l10_component_ablation.py")
SUMMARY = Path("paper/source_data/figure_data/Fig2/Fig2a_c_data.csv")
LINEAGE_EXAMPLES = {"CD3D", "CD3E", "CD3G", "CD8A", "CD8B"}
EXPECTED_NOTEBOOK_FUNCTION = """
def marker_recall_at_n(genes, gt_label, adata_var_names):
    truth = CD8_MARKER_DB.get(gt_label, set())
    truth = [g for g in truth if g in set(map(str, adata_var_names))]
    if len(truth) == 0:
        return np.nan
    hit = sum(g in set(genes) for g in truth)
    return hit / len(truth)
"""


def same_ast(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def literal_dictionary(tree):
    matches = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "CD8_MARKER_DB" for t in node.targets)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one literal CD8_MARKER_DB assignment per source.")
    result = ast.literal_eval(matches[0])
    if not isinstance(result, dict) or not all(isinstance(k, str) and isinstance(v, set)
                                             and all(isinstance(g, str) for g in v)
                                             for k, v in result.items()):
        raise ValueError("CD8_MARKER_DB must map state names to literal gene sets.")
    return result


def require_expression(tree, expression, message):
    expected = ast.parse(expression, mode="eval").body
    if not any(same_ast(node, expected) for node in ast.walk(tree)):
        raise ValueError(message)


def audit(repo):
    notebook = json.loads((repo / NOTEBOOK).read_text())
    cells = [(i, "".join(cell.get("source", []))) for i, cell in enumerate(notebook["cells"])
             if cell.get("cell_type") == "code"]
    db_cells = [(i, source) for i, source in cells if "CD8_MARKER_DB =" in source]
    if len(db_cells) != 1:
        raise ValueError("Expected one notebook cell defining CD8_MARKER_DB.")
    notebook_db = literal_dictionary(ast.parse(db_cells[0][1]))
    ablation_source = (repo / ABLATION).read_text()
    ablation_tree = ast.parse(ablation_source)
    ablation_db = literal_dictionary(ablation_tree)
    if notebook_db != ablation_db:
        raise ValueError("Notebook and ablation reference dictionaries differ; review their definitions.")

    functions = [(i, node) for i, source in cells if "def marker_recall_at_n" in source
                 for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "marker_recall_at_n"]
    expected = ast.parse(EXPECTED_NOTEBOOK_FUNCTION).body[0]
    if len(functions) != 1 or not same_ast(functions[0][1], expected):
        raise ValueError("Notebook recall function changed; re-review the stated metric before using this audit.")
    evaluate = next(node for node in ablation_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_lists")
    require_expression(evaluate, "CD8_MARKER_DB[gt] & set(var_names)",
                       "Ablation eligible-reference definition changed.")
    require_expression(evaluate, "set(genes)", "Ablation selected-gene set definition changed.")
    require_expression(evaluate, "len(gene_set & canonical) / len(canonical) if canonical else np.nan",
                       "Ablation recall formula changed.")
    genes = set().union(*notebook_db.values())
    if LINEAGE_EXAMPLES & genes:
        raise ValueError("At least one cited lineage-sentinel example is now in the recall dictionary.")

    with (repo / SUMMARY).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    at_50 = {row["variant"]: float(row["marker_recall_pct"])
             for row in rows if int(row["n_top_eval"]) == 50}
    return {
        "scope": "Source-definition audit; no expression analysis, annotation scoring, or inference",
        "dictionaries_identical": True,
        "state_categories": len(notebook_db),
        "distinct_reference_genes": len(genes),
        "reference_genes_per_state": {k: len(v) for k, v in sorted(notebook_db.items())},
        "lineage_examples_outside_all_reference_sets": sorted(LINEAGE_EXAMPLES),
        "per_cluster_formula": "|selected_genes intersect state_reference intersect feature_space| / |state_reference intersect feature_space|",
        "zero_eligible_reference_behavior": "NaN; excluded from pandas mean",
        "figure2_committed_recall_percent_at_50": at_50,
        "notebook_reference_cell_zero_based": db_cells[0][0],
        "notebook_formula_cell_zero_based": functions[0][0],
        "source_sha256": {str(p): hashlib.sha256((repo / p).read_bytes()).hexdigest()
                          for p in (NOTEBOOK, ABLATION, SUMMARY)},
        "limits": [
            "Shared metric definitions do not establish identical selected lists or identical runs.",
            "This audit does not recover Table S5 gene lists or reconcile its full_core baseline with Figure 2c.",
            "State-marker recall does not measure all lineage evidence or annotation accuracy.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = audit(args.repo.resolve())
    except (OSError, ValueError, KeyError, SyntaxError, StopIteration) as exc:
        print(f"Canonical-recall audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
