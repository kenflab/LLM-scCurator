"""Regression tests for the two reviewer-analysis scoring functions."""
import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name):
    path = ROOT / "paper/revision_metrics" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [load(name) for name in [
            "make_l4_external_marker_noise_manual_o3_top10",
            "make_gptcelltype_shared_annotator_check",
        ]]

    def evaluate(self, prediction, truth, major, gt_major, state, score):
        for module in self.modules:
            with self.subTest(script=module.__name__, prediction=prediction):
                predictions = pd.DataFrame([{"Cluster_ID": "test",
                                             "Raw_Prediction": prediction}])
                audit = pd.DataFrame([{"Cluster_ID": "test", "Ground_Truth": truth}])
                row = module.score_predictions(predictions, audit).iloc[0]
                self.assertEqual(row.Raw_Prediction, prediction)
                self.assertEqual(row.Pred_Major, major)
                self.assertEqual(row.GT_Major, gt_major)
                self.assertEqual(row.Pred_State, state)
                self.assertEqual(bool(row.Major_Lineage_Match), major == gt_major)
                self.assertAlmostEqual(row.Score_Sanno, score)

    def test_state_does_not_assign_t_lineage(self):
        self.evaluate("Proliferating doublet (T+B)", "CD8_Exhausted",
                      "Other", "T", "Cycling", 0.0)

    def test_mixed_label_preserves_public_alias_priority(self):
        self.evaluate("T cell-monocyte doublet", "CD8_EffectorMemory",
                      "Myeloid", "T", "Other", 0.0)

    def test_state_only_label_gets_only_state_credit(self):
        self.evaluate("Effector", "CD8_EffectorMemory",
                      "Other", "T", "EffMem", 0.3)

    def test_reference_lineage_comes_from_public_mapping(self):
        self.evaluate("NK cell effector", "NK_Killer",
                      "NK", "NK", "EffMem", 1.0)

    def test_normalization_matches_public_scorer(self):
        self.evaluate("N*K cell\neffector", "CD8_EffectorMemory",
                      "NK", "T", "EffMem", 0.0)

    def test_failure_penalty_does_not_erase_parsed_lineage(self):
        self.evaluate("Ribosomal-high CD8 T cell", "CD8_EffectorMemory",
                      "T", "T", "Other", 0.0)


if __name__ == "__main__":
    unittest.main()
