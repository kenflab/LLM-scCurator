from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from paper.hlca_external_validation.hlca_pipeline import (
    Config,
    bootstrap_mean_ci,
    build_cluster_manifest,
    build_label_hierarchy,
    create_mapping_template,
    make_scope_figure,
    normalized_label,
    paired_wilcoxon,
    parse_llm_json,
    prepare_markers,
    score_results,
    score_path,
)


def minimal_config(root: Path) -> Config:
    return Config(
        source_path=root / "config.toml",
        values={
            "columns": {
                "cluster": "leiden_3",
                "level_1": "ann_level_1",
                "level_2": "ann_level_2",
                "level_3": "ann_level_3",
                "donor": "donor_id",
                "source_dataset": "dataset",
            }
        },
    )


class TestHLCAHelpers(unittest.TestCase):
    def test_manifest_uses_majority_state_and_its_parent_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = minimal_config(Path(tmp))
            obs = pd.DataFrame(
                {
                    "leiden_3": ["1.0", "1.0", "1.0", "2.0"],
                    "ann_level_1": ["Immune", "Immune", "Epithelial", "Stroma"],
                    "ann_level_2": ["Lymphoid", "Lymphoid", "Airway", "Mesenchymal"],
                    "ann_level_3": ["T cell lineage", "T cell lineage", "Basal", None],
                    "donor_id": ["d1", "d2", "d3", "d4"],
                    "dataset": ["a", "a", "b", "b"],
                }
            )
            manifest = build_cluster_manifest(obs, cfg).set_index("cluster_id")
            self.assertEqual(manifest.at["1.0", "gt_ann_level_3"], "T cell lineage")
            self.assertEqual(manifest.at["1.0", "gt_ann_level_1"], "Immune")
            self.assertAlmostEqual(manifest.at["1.0", "gt_ann_level_3_purity"], 2 / 3)
            self.assertTrue(bool(manifest.at["1.0", "evaluable"]))
            self.assertFalse(bool(manifest.at["2.0", "evaluable"]))

    def test_hierarchy_counts_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = minimal_config(Path(tmp))
            obs = pd.DataFrame(
                {
                    "ann_level_1": ["Immune", "Immune", "Stroma"],
                    "ann_level_2": ["Lymphoid", "Lymphoid", "Mesenchymal"],
                    "ann_level_3": ["T cell lineage", "T cell lineage", "Fibroblasts"],
                }
            )
            hierarchy = build_label_hierarchy(obs, cfg)
            self.assertEqual(len(hierarchy), 2)
            self.assertEqual(int(hierarchy.loc[hierarchy["ann_level_3"] == "T cell lineage", "n_cells"].iloc[0]), 2)

    def test_sanno_levels(self) -> None:
        truth = ("Immune", "Lymphoid", "T cell lineage")
        exact = score_path(truth, truth, lineage_weight=0.7, state_weight=0.3, level_2_partial_credit=0.5)
        level2 = score_path(
            truth,
            ("Immune", "Lymphoid", "B cell lineage"),
            lineage_weight=0.7,
            state_weight=0.3,
            level_2_partial_credit=0.5,
        )
        lineage = score_path(
            truth,
            ("Immune", "Myeloid", "Macrophages"),
            lineage_weight=0.7,
            state_weight=0.3,
            level_2_partial_credit=0.5,
        )
        wrong = score_path(
            truth,
            ("Stroma", "Mesenchymal", "Fibroblasts"),
            lineage_weight=0.7,
            state_weight=0.3,
            level_2_partial_credit=0.5,
        )
        explicit_level2 = score_path(
            truth,
            ("Immune", "Lymphoid", "__NO_LEVEL_3__"),
            lineage_weight=0.7,
            state_weight=0.3,
            level_2_partial_credit=0.5,
        )
        explicit_level1 = score_path(
            truth,
            ("Immune", "__NO_LEVEL_2__", "__NO_LEVEL_3__"),
            lineage_weight=0.7,
            state_weight=0.3,
            level_2_partial_credit=0.5,
        )
        self.assertAlmostEqual(exact["sanno_hlca"], 1.0)
        self.assertAlmostEqual(level2["sanno_hlca"], 0.85)
        self.assertAlmostEqual(lineage["sanno_hlca"], 0.7)
        self.assertAlmostEqual(wrong["sanno_hlca"], 0.0)
        self.assertAlmostEqual(explicit_level2["sanno_hlca"], 0.85)
        self.assertEqual(explicit_level2["exact_level_3_accuracy"], 0.0)
        self.assertAlmostEqual(explicit_level1["sanno_hlca"], 0.7)
        self.assertEqual(explicit_level1["level_2_accuracy"], 0.0)

    def test_json_and_normalization(self) -> None:
        value = parse_llm_json(
            '```json\n{"cell_type":"AT2 cell","confidence":"High","reasoning":"SFTPC"}\n```'
        )
        self.assertEqual(value["cell_type"], "AT2 cell")
        self.assertEqual(normalized_label("AT2-cell"), "at2 cell")

    def test_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_mean_ci([0.1, 0.2, -0.1], iterations=2000, seed=42)
        second = bootstrap_mean_ci([0.1, 0.2, -0.1], iterations=2000, seed=42)
        self.assertEqual(first, second)

    def test_wilcoxon_is_stable_to_machine_level_tie_noise(self) -> None:
        values = np.asarray([0.0, 0.05, 0.05, -0.10, -0.10, 0.15, -0.15])
        noise = np.asarray([0.0, 1e-16, -1e-16, 1e-16, -1e-16, 1e-16, -1e-16])
        self.assertEqual(paired_wilcoxon(values), paired_wilcoxon(values + noise))

    def test_prepare_markers_from_backed_raw_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            clusters = np.repeat(["1.0", "2.0", "3.0"], 30)
            counts = rng.poisson(0.2, size=(len(clusters), 60)).astype(np.float32)
            for cluster_index in range(3):
                mask = clusters == f"{cluster_index + 1}.0"
                counts[mask, cluster_index * 5 : (cluster_index + 1) * 5] += rng.poisson(
                    5.0, size=(int(mask.sum()), 5)
                )
            obs = pd.DataFrame(
                {
                    "leiden_3": clusters,
                    "ann_level_1": np.repeat(["Immune", "Epithelial", "Stroma"], 30),
                    "ann_level_2": np.repeat(["Lymphoid", "Airway", "Mesenchymal"], 30),
                    "ann_level_3": np.repeat(["T cell lineage", "AT2", "Fibroblasts"], 30),
                    "donor_id": np.tile(["d1", "d2", "d3"], 30),
                    "dataset": np.tile(["a", "b"], 45),
                },
                index=[f"cell_{i:03d}" for i in range(len(clusters))],
            )
            var = pd.DataFrame(
                {"feature_name": [f"GENE{i:03d}" for i in range(counts.shape[1])]},
                index=[f"ENSG{i:06d}" for i in range(counts.shape[1])],
            )
            source = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
            source.raw = source.copy()
            sc.pp.normalize_total(source, target_sum=10000)
            sc.pp.log1p(source)
            source_path = root / "source.h5ad"
            source.write_h5ad(source_path)

            values = {
                "columns": minimal_config(root).values["columns"]
                | {"gene_symbol_candidates": ["feature_name"]},
                "analysis": {
                    "seed": 42,
                    "max_cells_per_cluster": 20,
                    "min_cells_per_gene": 1,
                    "normalization_target_sum": 10000.0,
                    "n_markers": 5,
                },
                "dataset": {"expected_n_obs": len(clusters)},
                "paths": {
                    "source_h5ad": "source.h5ad",
                    "processed_h5ad": "processed.h5ad",
                    "cluster_manifest": "manifest.csv",
                    "label_hierarchy": "hierarchy.csv",
                    "marker_lists": "markers.csv",
                },
            }
            cfg = Config(source_path=root / "config.toml", values=values)
            manifest_path, marker_path = prepare_markers(cfg)
            manifest = pd.read_csv(manifest_path)
            markers = pd.read_csv(marker_path)
            self.assertEqual(int(manifest["evaluable"].sum()), 3)
            self.assertTrue((manifest["n_cells_sampled"] == 20).all())
            self.assertEqual(len(markers), 6)
            self.assertEqual(set(markers["condition"]), {"standard", "full_core"})
            self.assertTrue((markers["n_genes"] > 0).all())

    def test_mapping_scoring_and_figures_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text("# synthetic test config\n", encoding="utf-8")
            paths = {
                "cluster_manifest": "manifest.csv",
                "label_hierarchy": "hierarchy.csv",
                "raw_responses_dir": "responses",
                "mapping": "mapping.csv",
                "call_scores": "call_scores.csv",
                "cluster_scores": "cluster_scores.csv",
                "summary": "summary.csv",
                "figure_main": "fig3d",
                "figure_supp": "figs6",
                "run_metadata": "run_metadata.json",
            }
            cfg = Config(
                source_path=config_path,
                values={
                    "paths": paths,
                    "analysis": {
                        "n_repeats": 2,
                        "bootstrap_iterations": 500,
                        "seed": 42,
                        "sanno_lineage_weight": 0.7,
                        "sanno_state_weight": 0.3,
                        "sanno_level_2_partial_credit": 0.5,
                    },
                    "dataset": {"name": "synthetic"},
                    "columns": minimal_config(root).values["columns"],
                    "llm": {"model": "mock", "temperature": 0.0},
                },
            )
            manifest = pd.DataFrame(
                [
                    {
                        "cluster_id": "1.0",
                        "evaluable": True,
                        "gt_ann_level_1": "Immune",
                        "gt_ann_level_2": "Lymphoid",
                        "gt_ann_level_3": "T cell lineage",
                        "n_cells_full": 100,
                        "n_cells_sampled": 20,
                        "n_donors": 3,
                        "n_source_datasets": 2,
                        "gt_ann_level_3_purity": 0.9,
                    },
                    {
                        "cluster_id": "2.0",
                        "evaluable": True,
                        "gt_ann_level_1": "Epithelial",
                        "gt_ann_level_2": "Alveolar",
                        "gt_ann_level_3": "AT2",
                        "n_cells_full": 120,
                        "n_cells_sampled": 20,
                        "n_donors": 4,
                        "n_source_datasets": 2,
                        "gt_ann_level_3_purity": 0.95,
                    },
                ]
            )
            manifest.to_csv(cfg.path("cluster_manifest"), index=False)
            hierarchy = pd.DataFrame(
                [
                    {
                        "path_id": "HLCA_PATH_001",
                        "ann_level_1": "Immune",
                        "ann_level_2": "Lymphoid",
                        "ann_level_3": "T cell lineage",
                        "n_cells": 100,
                    },
                    {
                        "path_id": "HLCA_PATH_002",
                        "ann_level_1": "Epithelial",
                        "ann_level_2": "Alveolar",
                        "ann_level_3": "AT2",
                        "n_cells": 120,
                    },
                ]
            )
            hierarchy.to_csv(cfg.path("label_hierarchy"), index=False)
            response_dir = cfg.path("raw_responses_dir")
            response_dir.mkdir(parents=True)
            for cluster_id, truth in (("1.0", "T cell lineage"), ("2.0", "AT2")):
                for condition in ("standard", "full_core"):
                    for repeat in (1, 2):
                        prediction = truth
                        if cluster_id == "1.0" and condition == "standard":
                            prediction = "outside atlas"
                        record = {
                            "status": "success",
                            "cluster_id": cluster_id,
                            "condition": condition,
                            "repeat": repeat,
                            "requested_model": "mock",
                            "returned_model_version": "mock-v1",
                            "parsed": {
                                "cell_type": prediction,
                                "confidence": "High",
                                "reasoning": "synthetic",
                            },
                        }
                        name = f"{cluster_id}__{condition}__{repeat}.json"
                        (response_dir / name).write_text(json.dumps(record), encoding="utf-8")

            mapping_path = create_mapping_template(cfg)
            mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
            mapping.loc[
                mapping["prediction_text"] == "outside atlas", "mapped_path_id"
            ] = "UNMAPPED"
            mapping.to_csv(mapping_path, index=False)
            call_path, cluster_path, summary_path = score_results(cfg)
            summary = pd.read_csv(summary_path).set_index("metric")
            self.assertTrue(call_path.exists())
            self.assertTrue(cluster_path.exists())
            self.assertAlmostEqual(summary.at["sanno_hlca", "mean_paired_difference"], 0.5)
            self.assertTrue(cfg.path("figure_main").with_suffix(".pdf").exists())
            self.assertTrue(cfg.path("figure_supp").with_suffix(".png").exists())

    def test_scope_figure_from_cached_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(
                source_path=root / "config.toml",
                values={
                    "paths": {
                        "figure_scope": "figs7",
                        "umap_coordinates": "umap_coordinates.csv",
                    },
                    "analysis": {"seed": 42},
                },
            )
            labels = [
                ("Epithelial", "AT2"),
                ("Endothelial", "EC capillary"),
                ("Stroma", "Fibroblasts"),
                ("Immune", "Macrophages"),
            ]
            coordinates = pd.DataFrame(
                [
                    {
                        "cell_id": f"cell_{i}",
                        "cluster_id": str(i // 4),
                        "ann_level_1": labels[i % 4][0],
                        "ann_level_3": labels[i % 4][1],
                        "umap_1": float(np.cos(i)),
                        "umap_2": float(np.sin(i)),
                    }
                    for i in range(24)
                ]
            )
            coordinates.to_csv(cfg.path("umap_coordinates"), index=False)
            figure_path, coordinate_path = make_scope_figure(cfg)
            self.assertTrue(figure_path.exists())
            self.assertTrue(figure_path.with_suffix(".png").exists())
            self.assertEqual(coordinate_path, cfg.path("umap_coordinates"))


if __name__ == "__main__":
    unittest.main()
