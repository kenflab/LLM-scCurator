#!/usr/bin/env python3
"""Prepare separate prediction, reference, and private audit packets; no scoring or APIs."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile


REVIEW_FIELDS = [
    "compartment", "broad_lineage", "t_cell_identity",
    "identity_at_supported_resolution", "state", "unresolved", "review_note",
]
CODEBOOK_FILES = [
    "benchmarks/hierarchical_scoring.py", "benchmarks/gt_mappings.py",
    "benchmarks/cd8_config.py", "benchmarks/cd4_config.py",
    "benchmarks/caf_config.py", "benchmarks/mouse_b_config.py",
]
SOURCE_RELATIVE = "paper/source_data/figure_data/Fig3/inputs/L4_prediction_audit.csv"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prediction_id(raw_text):
    """Use the exact saved text; do not silently collapse case or biological punctuation."""
    return "pred_" + digest(raw_text.encode("utf-8"))[:24]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_packet(repo, source, outdir):
    if outdir == repo or outdir.is_relative_to(repo):
        raise ValueError("Choose an output directory outside the repository; review keys must not be committed.")
    if outdir.exists():
        raise ValueError("Output directory already exists. Use a new directory; existing reviews are never overwritten.")
    raw_bytes = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    required = {"Dataset", "Cluster_ID", "Method", "Ground_Truth", "Raw_Prediction", "GT_Major", "GT_State"}
    if not required.issubset(fields):
        raise ValueError("Missing source fields: " + ", ".join(sorted(required - set(fields))))
    rows = list(reader)
    primary = [r for r in rows if r["Method"] in {"Standard", "Curated"}]
    if len(rows) != 250 or len(primary) != 104:
        raise ValueError(f"Expected the frozen 250-row audit with 104 primary records; found {len(rows)} and {len(primary)}.")
    keys = [(r["Dataset"], r["Cluster_ID"], r["Method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset/cluster/method keys in source audit.")
    primary_pairs = {}
    references, codebook = {}, {}
    for row in rows:
        if not row["Raw_Prediction"].strip():
            raise ValueError("Empty saved prediction; resolve source records before review preparation.")
        key = (row["Dataset"], row["Cluster_ID"])
        value = (row["Ground_Truth"], row["GT_Major"], row["GT_State"])
        if key in references and references[key] != value:
            raise ValueError(f"Inconsistent reference labels across methods: {key}")
        references[key] = value
        code_key = (row["Dataset"], row["Ground_Truth"])
        if code_key in codebook and codebook[code_key] != value[1:]:
            raise ValueError(f"Inconsistent legacy reference parsing: {code_key}")
        codebook[code_key] = value[1:]
    for row in primary:
        primary_pairs.setdefault((row["Dataset"], row["Cluster_ID"]), set()).add(row["Method"])
    if len(references) != 52 or len(primary_pairs) != 52 or any(v != {"Standard", "Curated"} for v in primary_pairs.values()):
        raise ValueError("Expected 52 complete Standard/Curated cluster pairs.")
    texts = sorted({r["Raw_Prediction"] for r in rows}, key=prediction_id)
    if len({prediction_id(t) for t in texts}) != len(texts):
        raise ValueError("Prediction identifier collision.")
    for path in CODEBOOK_FILES:
        if not (repo / path).is_file():
            raise ValueError(f"Missing legacy codebook source: {path}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".label_review_", dir=outdir.parent) as temporary:
        stage = Path(temporary) / "packet"
        stage.mkdir()
        blinded = [{"prediction_id": prediction_id(t), "raw_prediction": t, **{k: "" for k in REVIEW_FIELDS}} for t in texts]
        write_csv(stage / "reviewer" / "prediction_mapping.csv", ["prediction_id", "raw_prediction"] + REVIEW_FIELDS, blinded)
        review_instructions = """Prediction label review

Review only the wording of each prediction. The sheet deliberately omits model,
method, condition, dataset, cluster, reference label, scores, and occurrence counts.
Do not inspect the private audit keys or search for the labels in the public data.
This is masked presentation, not proof that a reviewer has never seen the results.

All mapping fields are blank. Record only what the text supports. Use a broad
lineage and a separate T-cell identity (for example CD4, CD8, T_unspecified,
gamma_delta_T, MAIT, non_T, or unresolved) where applicable. These examples are
field guidance, not an approved ontology or an instruction to force a category.
Do not infer CD4 or CD8 from the unknown benchmark's composition. Keep a generic
T-cell label broad; an explicit CD8 call and an unspecified T call are different.
Record state independently from identity; multiple stated programs may be entered
with semicolons. Retain ambiguity instead of inferring an unstated subtype.
Mark unresolved yes or no and explain uncertainty or an ambiguous synonym in
review_note. The reference ontology and final scoring rules require a separate
review and must be frozen before joining outcomes to conditions.

Repeated exact text has one row and must receive one mapping. Case, punctuation,
or spelling variants retain separate IDs so they can be compared explicitly.
Semantically equivalent variants should be mapped consistently; disagreements
require adjudication with a rationale. Do not use an LLM as an automatic judge.
Do not edit prediction_id or raw_prediction. Save a separately versioned reviewed
copy. This packet does not ask you to decide whether the label matches a cell.
"""
        (stage / "reviewer" / "README.txt").write_text(review_instructions, encoding="utf-8")
        ref_rows = []
        for (dataset, cluster), (ground_truth, major, state) in sorted(references.items()):
            ref_rows.append({
                "dataset": dataset, "cluster_id": cluster, "legacy_ground_truth": ground_truth,
                "legacy_broad_lineage": major, "legacy_state": state,
                **{k: "" for k in REVIEW_FIELDS}, "reference_provenance": "", "reference_evidence": "",
            })
        ref_fields = ["dataset", "cluster_id", "legacy_ground_truth", "legacy_broad_lineage", "legacy_state"] + REVIEW_FIELDS + ["reference_provenance", "reference_evidence"]
        write_csv(stage / "reference" / "reference_label_review.csv", ref_fields, ref_rows)
        code_rows = [{"dataset": d, "legacy_ground_truth": gt, "legacy_broad_lineage": v[0], "legacy_state": v[1]} for (d, gt), v in sorted(codebook.items())]
        write_csv(stage / "reference" / "legacy_reference_codebook.csv", ["dataset", "legacy_ground_truth", "legacy_broad_lineage", "legacy_state"], code_rows)
        code_hashes = {}
        for path in CODEBOOK_FILES:
            data = (repo / path).read_bytes()
            target = stage / "reference" / "legacy_codebook_source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            code_hashes[path] = digest(data)
        (stage / "reference" / "README.txt").write_text(
            "Reference review is separate from prediction-text mapping. This folder has no model predictions or scores.\n"
            "The legacy codebook CSV lists all reference categories used by this snapshot; exact source code freezes\n"
            "the complete historical aliases, precedence and scoring rules. These are provenance, not endorsement.\n"
            "Review original reference labels and source evidence independently of method outcomes. Record uncertain\n"
            "or disputed reference labels explicitly; do not force a finer identity than the source supports.\n"
            "Resolve CD4/CD8 specificity, resident-memory treatment, mixed labels and lineage/state separation in\n"
            "one evaluation specification. Do not silently fill reference fields from current parser outputs.\n"
            "New mapping fields are blank. Freeze adjudicated references and rules before the condition join.\n",
            encoding="utf-8",
        )
        for filename, selected in [("all_predictions_key.csv", rows), ("primary_104_key.csv", primary)]:
            key_rows = [{"prediction_id": prediction_id(r["Raw_Prediction"]), **r} for r in selected]
            write_csv(stage / "private" / "audit" / filename, ["prediction_id"] + fields, key_rows)
        shutil.copyfile(source, stage / "private" / "audit" / "source_prediction_audit.csv")
        manifest = {
            "schema_version": 1, "operation": "review_preparation_only", "new_llm_calls": 0,
            "source_file_name": source.name, "source_sha256": digest(raw_bytes),
            "source_records": len(rows), "primary_records": len(primary), "reference_clusters": len(references),
            "unique_raw_predictions": len(texts), "prediction_id_rule": "pred_ plus first 24 hex digits of SHA256 of exact UTF-8 raw text",
            "legacy_codebook_sha256": code_hashes,
            "historical_predictions_or_scores_modified": False, "biological_mappings_prefilled": False,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "This packet prepares review; it does not produce corrected scores.\n"
            "Give only reviewer/ to the prediction-text reviewer. Do not give them private/ or manifest.json.\n"
            "Give reference/ to the reference-label reviewer separately. Keep private/audit/ with the analysis\n"
            "coordinator. It contains all method, condition, cluster, reference and historical score information.\n"
            "Do not publish the join keys with a blinded reviewer packet. All 250 available method predictions\n"
            "are retained; primary_104_key.csv is the Standard/Curated subset, not a different mapping exercise.\n"
            "The missing mouse B CellTypist/Azimuth comparisons remain absent, not incorrect predictions.\n"
            "Before scoring, freeze the reviewed mapping, reference labels, ontology, missing-label policy,\n"
            "treatment of generic versus explicitly contradictory subtype calls, and state metrics.\n"
            "Apply the same mappings to every method. Report disagreements and old/new results together.\n"
            "The original raw labels and scoring code are unchanged. This folder must remain outside git.\n",
            encoding="utf-8",
        )
        stage.rename(outdir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit-csv", type=Path, help="Default: frozen Figure 3 L4 prediction audit")
    parser.add_argument("--outdir", type=Path, help="New directory outside the repository; defaults beside the checkout")
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = args.audit_csv.resolve() if args.audit_csv else repo / SOURCE_RELATIVE
    outdir = args.outdir.resolve() if args.outdir else repo.parent / "LLM_scCurator_label_review"
    try:
        result = build_packet(repo, source, outdir)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_directory": str(outdir), **result}, indent=2))


if __name__ == "__main__":
    main()
