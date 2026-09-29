import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("figure3", ROOT / "paper/scripts/make_figure3.py")
FIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE)
INPUTS = ROOT / "paper/source_data/figure_data/Fig3/inputs"
L4 = INPUTS / "L4_complementary_metrics.csv"
AUDIT = INPUTS / "L4_prediction_audit.csv"


class Figure3Tests(unittest.TestCase):
    def test_corrected_values_and_missing_comparators(self):
        data = FIGURE.build_source_data(L4, AUDIT).set_index(["Dataset", "Method"])
        for pair, value in [(('CD8 T', 'Standard'), 14/17),
                            (('CD4 T', 'SingleR'), 16/22),
                            (('MSC', 'Curated'), 7/8),
                            (('Mouse B', 'SingleR'), 4/5)]:
            self.assertAlmostEqual(data.loc[pair, 'Major_Lineage_Accuracy'], value)
        for method in ['CellTypist', 'Azimuth']:
            self.assertFalse(data.loc[('Mouse B', method), 'Evaluated'])
            self.assertTrue(pd.isna(data.loc[('Mouse B', method), 'Mean_S_anno']))
        self.assertEqual(data.loc[('MSC', 'SingleR'), FIGURE.METRICS[4]], 0)
        self.assertEqual(data.N.sum(), 250)

    def test_mismatched_l4_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = pd.read_csv(L4)
            changed.loc[0, 'Major_Lineage_Accuracy'] = 1
            path = Path(tmp) / 'changed.csv'
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'L4/audit mismatch'):
                FIGURE.build_source_data(path, AUDIT, n_boot=10)

    def test_incomplete_clusters_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'incomplete.csv'
            pd.read_csv(AUDIT).iloc[1:].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Incomplete cluster coverage'):
                FIGURE.build_source_data(L4, path, n_boot=10)

    def test_bootstrap_matches_original_sampling_convention(self):
        values = np.array([0., .3, .7, 1.])
        rng = np.random.default_rng(42)
        means = [rng.choice(values, len(values), replace=True).mean() for _ in range(20000)]
        np.testing.assert_array_equal(FIGURE.bootstrap_ci(values), np.quantile(means, [.025, .975]))


if __name__ == '__main__':
    unittest.main()
