#!/usr/bin/env python3
"""Recompute L4/L5 complementary metrics and figures from saved scored records.

No marker selection or LLM inference is performed. Source files are read only.
Run from the repository root; outputs must go to a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_script(name):
    path = Path(__file__).parent / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescore(module, source, expected_repeats):
    old, _ = module.read_csv_with_encoding_fallback(source)
    required = {"Condition", "Cluster_ID", "Repeat", "Raw_Prediction",
                "Ground_Truth", "Score_Sanno", "Exact_State",
                "Major_Lineage_Match", "Hierarchy_Consistent"}
    missing = required - set(old.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if old[list(required)].isna().any().any():
        raise ValueError(f"{source}: missing evaluation fields")
    keys = ["Condition", "Cluster_ID", "Repeat"]
    if old.duplicated(keys).any():
        raise ValueError(f"{source}: duplicate prediction records")
    if set(old.Condition) != set(expected_repeats):
        raise ValueError(f"{source}: unexpected or missing conditions")
    audit = old[["Cluster_ID", "Ground_Truth"]].drop_duplicates()
    if len(audit) != 17 or audit.Cluster_ID.duplicated().any():
        raise ValueError(f"{source}: expected 17 uniquely mapped clusters")
    for condition, repeats in expected_repeats.items():
        group = old[old.Condition.eq(condition)]
        if set(group.Cluster_ID) != set(audit.Cluster_ID):
            raise ValueError(f"{condition}: incomplete cluster coverage")
        repeat_ids = pd.to_numeric(group.Repeat, errors="raise")
        if set(repeat_ids) != set(range(1, repeats + 1)):
            raise ValueError(f"{condition}: unexpected repeat identifiers")
        if not group.groupby("Cluster_ID").size().eq(repeats).all():
            raise ValueError(f"{condition}: incomplete repeats")

    derived = {"Ground_Truth", "Pred_State", "GT_State", "Pred_Major", "GT_Major",
               "Score_Sanno", "Exact_State", "Major_Lineage_Match",
               "Hierarchy_Consistent"}
    payload = old.drop(columns=list(derived & set(old.columns)))
    new = module.score_predictions(payload, audit)
    # Refuse to turn a complementary-metric correction into a score revision.
    if not np.allclose(old.Score_Sanno.astype(float), new.Score_Sanno,
                       rtol=0, atol=1e-12):
        raise ValueError(f"{source}: Sanno changed; inspect the scorer/version first")
    for column in ["Exact_State", "Hierarchy_Consistent"]:
        original = old[column].astype(str).str.lower()
        if not original.equals(new[column].astype(str).str.lower()):
            raise ValueError(f"{source}: {column} changed; inspect before proceeding")
    pd.testing.assert_frame_equal(payload, new[payload.columns], check_dtype=False)
    previous = old.Major_Lineage_Match.astype(str).str.lower().eq("true")
    changes = new.loc[previous.ne(new.Major_Lineage_Match),
                      keys + ["Raw_Prediction", "Ground_Truth",
                              "Pred_Major", "GT_Major", "Major_Lineage_Match"]].copy()
    changes["Previous_Major_Lineage_Match"] = previous.loc[changes.index]
    return new, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l4-dir", type=Path)
    parser.add_argument("--l5-dir", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--l4-bootstrap", type=int, default=20000)
    parser.add_argument("--l5-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    if not args.l4_dir and not args.l5_dir:
        parser.error("Supply --l4-dir and/or --l5-dir")
    if min(args.l4_bootstrap, args.l5_bootstrap) < 1:
        parser.error("Bootstrap counts must be positive")
    # Resolve the checked-out public scorer, not another installed copy.
    repo = Path(__file__).resolve().parents[2]
    if not (repo / "benchmarks/cd8_config.py").is_file():
        parser.error("Install this script under paper/revision_metrics in the repository")
    sys.path.insert(0, str(repo))
    if args.outdir.exists():
        parser.error("--outdir must be a new directory")

    results = []
    for label, directory, name in [
        ("L4", args.l4_dir, "make_l4_external_marker_noise_manual_o3_top10"),
        ("L5", args.l5_dir, "make_gptcelltype_shared_annotator_check"),
    ]:
        if directory is None:
            continue
        module = load_script(name)
        filename = ("L4_shared_annotator_predictions_scored.csv" if label == "L4"
                    else "L5_all_predictions_scored.csv")
        source = directory / filename
        repeats = {c: (1 if label == "L5" and c in module.WORKFLOW_CONDITIONS else 5)
                   for c in module.CONDITION_ORDER}
        scored, changes = rescore(module, source, repeats)
        boot = args.l4_bootstrap if label == "L4" else args.l5_bootstrap
        values = module.summarize_predictions(scored, n_bootstrap=boot, seed=args.seed)
        diagnostics = None
        if label == "L4" and not args.skip_figures:
            diagnostics = pd.read_csv(directory / "L4_external_marker_input_diagnostics.csv")
        results.append((label, module, source, scored, changes, values, diagnostics))

    args.outdir.mkdir(parents=True)
    provenance = {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "new_inference_calls": 0, "marker_selection_recomputed": False,
        "scorer_sha256": {str(p.relative_to(repo)): sha256(p) for p in
                         [repo / "benchmarks/cd8_config.py",
                          repo / "benchmarks/hierarchical_scoring.py"]},
        "seed": args.seed, "l4_bootstrap": args.l4_bootstrap,
        "l5_bootstrap": args.l5_bootstrap, "analyses": [],
    }
    for label, module, source, scored, changes, values, diagnostics in results:
        out = args.outdir / label
        out.mkdir()
        per_cluster, summary = values[:2]
        scored.to_csv(out / source.name, index=False, encoding="utf-8-sig")
        changes.to_csv(out / "major_lineage_changes.csv", index=False)
        if label == "L4":
            per_cluster.to_csv(out / "L4_shared_annotator_per_cluster_metrics.csv", index=False)
            summary.to_csv(out / "L4_shared_annotator_summary.csv", index=False)
            values[2].to_csv(out / "L4_shared_annotator_paired_comparisons.csv", index=False)
            if not args.skip_figures:
                module.make_figure(diagnostics, per_cluster, summary,
                    out_pdf=out / "ReviewerFig_L4.pdf", out_png=out / "ReviewerFig_L4.png",
                    noise_available=diagnostics.Biological_Noise_Fraction.notna().all(),
                    annotator_label="manual o3")
        else:
            per_cluster.to_csv(out / "L5_per_cluster_metrics.csv", index=False)
            summary.to_csv(out / "L5_summary.csv", index=False)
            manual, workflow = module.build_contrasts(
                per_cluster, n_bootstrap=args.l5_bootstrap, seed=args.seed)
            manual.to_csv(out / "L5_manual_matched_contrasts.csv", index=False)
            workflow.to_csv(out / "L5_workflow_descriptive_contrast.csv", index=False)
            native = scored[scored.Condition.eq(module.API_CONDITION)]
            native.to_csv(out / "L22_native_GPTCelltype_predictions.csv", index=False)
            if not args.skip_figures:
                module.make_figure(per_cluster, summary, workflow,
                    out / "ReviewerFig_L5.pdf", out / "ReviewerFig_L5.png")
        provenance["analyses"].append({
            "analysis": label, "source": str(source.resolve()), "sha256": sha256(source),
            "script_sha256": sha256(Path(module.__file__)), "n_records": len(scored),
            "n_lineage_match_changes": len(changes), "sanno_unchanged": True,
            "exact_state_unchanged": True, "hierarchy_consistency_unchanged": True,
        })
        print(label)
        print(summary[["Condition", "Mean_Sanno", "Major_Lineage_Accuracy"]].to_string(index=False))
    (args.outdir / "rescoring_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved: {args.outdir}")


if __name__ == "__main__":
    main()
