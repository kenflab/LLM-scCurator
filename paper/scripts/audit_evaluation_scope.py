#!/usr/bin/env python3
"""Verify saved LLM-scCurator evidence; read-only, no expression or LLM calls."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('repository', type=Path)
a = p.parse_args()
repo = a.repository.resolve()
sys.path.insert(0, str(repo))
from benchmarks.hierarchical_scoring import score_hierarchical
configs = {dataset: getattr(importlib.import_module(f'benchmarks.{module}'), name)
           for dataset,module,name in [('CD8 T','cd8_config','CD8_HIER_CFG'),
                                       ('CD4 T','cd4_config','CD4_HIER_CFG'),
                                       ('MSC','caf_config','CAF_HIER_CFG'),
                                       ('Mouse B','mouse_b_config','MOUSE_B_CFG')]}
main = pd.read_csv(repo/'paper/source_data/figure_data/Fig3/inputs/L4_prediction_audit.csv')
recomputed = main.apply(lambda row: score_hierarchical(row, 'Raw_Prediction', configs[row.Dataset]), axis=1)
assert np.allclose(recomputed, main.Public_Sanno, rtol=0, atol=1e-12)
step = pd.read_csv(repo/'paper/source_data/figure_data/Fig2/Fig2f_data.csv')
hlca = repo/'paper/hlca_external_validation/source_data'
calls = pd.read_csv(hlca/'call_level_scores.csv')
assert not calls.duplicated(['cluster_id','condition','repeat']).any()
counts = calls.groupby(['cluster_id','condition']).size()
assert counts.eq(3).all()
summary = pd.read_csv(hlca/'paired_summary.csv')
means = calls.groupby(['cluster_id','condition']).mean(numeric_only=True)
checks = []
for r in summary.itertuples():
    values = means[r.metric].unstack()
    assert np.isclose(values.standard.mean(), r.mean_standard, rtol=0, atol=1e-12)
    assert np.isclose(values.full_core.mean(), r.mean_full_core, rtol=0, atol=1e-12)
    checks.append({'metric':r.metric, 'standard':float(values.standard.mean()),
                   'full_core':float(values.full_core.mean())})
print(json.dumps({'new_llm_calls':0, 'marker_selection_recomputed':False,
                  'main_prediction_records':len(main),
                  'main_score_max_absolute_difference':float(np.max(np.abs(recomputed-main.Public_Sanno))),
                  'stepwise_context_flags':step.groupby(['Variant','Use_Auto_Context']).size().rename('n').reset_index().to_dict('records'),
                  'hlca_calls':len(calls), 'hlca_raw_json_files':len(list((hlca/'raw_responses').glob('*.json'))),
                  'hlca_clusters':int(calls.cluster_id.nunique()),
                  'hlca_means_reproduced':checks},indent=2))
