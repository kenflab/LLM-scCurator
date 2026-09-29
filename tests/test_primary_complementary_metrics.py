"""Regression tests for label-based primary benchmark evaluation."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "primary_metrics", ROOT / "paper/revision_metrics/make_l4_l5_letter_tables.py"
)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def cd8_row(prediction="Cytotoxic NK cell", score=0.0):
    return {
        "Dataset": "CD8 T", "Cluster_ID": "CD8.c09.Tk.KIR2DL4",
        "Ground_Truth": "CD8_EffectorMemory",
        "Standard_Answer": prediction, "Score_Standard": score,
    }


class PrimaryComplementaryMetricsTest(unittest.TestCase):
    def test_forbidden_lineage_does_not_erase_state_agreement(self):
        record = metrics.parse_prediction_records(pd.DataFrame([cd8_row()])).iloc[0]
        self.assertEqual((record.GT_Major, record.Pred_Major), ("T", "NK"))
        self.assertEqual((record.GT_State, record.Pred_State), ("EffMem", "EffMem"))
        self.assertTrue(record.Exact_State_Agreement)
        self.assertFalse(record.Major_Lineage_Accuracy)
        self.assertFalse(record.Exact_Match_Accuracy)
        self.assertEqual(record.Public_Sanno, 0.0)

    def test_missing_or_conflicting_labels_cannot_be_reconstructed_from_score(self):
        for row in [
            {k: v for k, v in cd8_row().items() if k != "Standard_Answer"},
            dict(cd8_row(), Standard_CellType="Effector T cell"),
        ]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                metrics.parse_prediction_records(pd.DataFrame([row]))

    def test_changed_default_score_stops_evaluation(self):
        with self.assertRaisesRegex(ValueError, "Sanno mismatch"):
            metrics.parse_prediction_records(pd.DataFrame([cd8_row(score=0.3)]))

    def test_unavailable_comparator_is_not_counted_as_an_error(self):
        row = dict(cd8_row(), CellTypist_Answer="Unavailable", Score_CellTypist=float("nan"))
        records = metrics.parse_prediction_records(pd.DataFrame([row]))
        self.assertEqual(records.Method.tolist(), ["Standard"])

    def test_alternative_weights_retain_forbidden_lineage_penalties(self):
        row = dict(cd8_row(), Full_Pipeline_Answer="Effector T cell", Score_Curated=1.0)
        records = metrics.parse_prediction_records(pd.DataFrame([row]))
        # This checks scoring, not the unchanged bootstrap implementation.
        with patch.object(metrics, "bootstrap_ci_paired_diff", return_value=(1.0, 1.0)):
            sensitivity = metrics.build_l5_weight_sensitivity(records)
        self.assertEqual(len(sensitivity), 6)
        self.assertTrue(sensitivity.Mean_Standard.eq(0.0).all())
        self.assertTrue(sensitivity.Mean_Curated.eq(1.0).all())

    def test_quantitative_scope_does_not_follow_confusion_plot_flag(self):
        rows = [
            {"Dataset": dataset, "Cluster_ID": f"{dataset}.{i}",
             "Score_Standard": 0.0, "Score_Curated": 0.0, "UsedInConfusion": False}
            for dataset, count in metrics.EXPECTED_CLUSTERS.items()
            for i in range(count)
        ]
        audit = metrics.clean_audit_table(pd.DataFrame(rows))
        metrics.validate_primary_design(audit)
        self.assertEqual(len(audit), 52)
        with self.assertRaisesRegex(ValueError, "all 52"):
            metrics.validate_primary_design(audit.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
