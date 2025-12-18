#!/usr/bin/env python3
# paper/scripts/example_subsampled_ids_with_gt.py

"""
example_subsampled_ids_with_gt.py

Example (optional): export fixed subsampled cell/spot IDs and add deterministic ground-truth labels.

This script demonstrates how to:
1) write `paper/config/datasets.tsv` from local `.h5ad` paths,
2) export `paper/source_data/subsampled_ids/*_cells.csv` via `export_subsampled_ids.py`,
3) add `Ground_Truth` columns via deterministic YAML label maps using `apply_label_map.py`.

Inputs:
- Local `.h5ad` files (not distributed in this repo)
- YAML label maps in `paper/config/label_maps/`

Outputs (written under versioned Source Data paths):
- `paper/config/datasets.tsv`
- `paper/source_data/subsampled_ids/*_cells.csv`
- `paper/source_data/subsampled_ids/*_cells.with_gt.csv`

Notes:
- This is provided as a convenience example. Most review/verification can be done directly from `paper/source_data/`,
  indexed by `paper/FIGURE_MAP.csv`.
"""

from pathlib import Path
import os

DAY = "20251201"
Version = "v1"

PROJECT_ROOT = Path(os.getenv("LLMSC_ROOT", ".")).resolve()

DATA_DIR = Path(os.getenv("LLMSC_DATA_DIR", PROJECT_ROOT / "input")).resolve()
OUT_DIR  = Path(os.getenv("LLMSC_OUT_DIR",  PROJECT_ROOT / "runs" / f"{DAY}.{Version}")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
import random, os
import numpy as np
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"🔒 Random seed set to {RANDOM_SEED} for reproducibility.")

import scanpy as sc
import scipy

import os
import sys
import numpy as np
import pandas as pd

import json
import gc
import time

import warnings
warnings.filterwarnings("ignore")

DATASETS = {
    "cd8": {
        "adata_path": str(OUT_DIR / "cd8_benchmark_data.h5ad"),
        "cluster_col": "meta.cluster",
    },
    "cd4": {
        "adata_path":  str(OUT_DIR / "cd4_benchmark_data.h5ad"),
        "cluster_col": "meta.cluster",
    },
    "msc": {
        "adata_path":  str(OUT_DIR / "brca_msc_benchmark_data.h5ad"),
        "cluster_col": "meta.cluster",
    },
    "mouse_b": {
        "adata_path":  str(OUT_DIR / "mouse_b_benchmark_data.h5ad"),
        "cluster_col": "meta.cluster",
    },
}

IDMAP = {
    "cd8": "cd8",
    "cd4": "cd4",
    "msc": "brca_msc",
    "mouse_b": "mouse_b",
}

COLS = {
    "cd8": "Cancer_Type,meta.cluster",
    "cd4": "Cancer_Type,meta.cluster",
    "msc": "meta.cluster",
    "mouse_b": "meta.cluster",
}

rows = []
for key, spec in DATASETS.items():
    rows.append({
        "dataset_id": IDMAP.get(key, key),
        "adata_path": spec["adata_path"],
        "out_csv": f"paper/source_data/subsampled_ids/{IDMAP.get(key, key)}_cells.csv",
        "cols": COLS.get(key, "meta.cluster"),
    })

df = pd.DataFrame(rows)
os.makedirs("paper/config", exist_ok=True)
df.to_csv("paper/config/datasets.tsv", sep="\t", index=False)
df

import subprocess, sys

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

run([sys.executable, "paper/scripts/export_subsampled_ids.py",
     "--datasets-tsv", "paper/config/datasets.tsv", "--dry-run"])

run([sys.executable, "paper/scripts/export_subsampled_ids.py",
     "--datasets-tsv", "paper/config/datasets.tsv"])

run([sys.executable, "paper/scripts/apply_label_map.py",
     "--label-map", "paper/config/label_maps/cd8_gt_map.yaml",
     "--text", "CD8.c02.Tm.IL7R"])

run([sys.executable, "paper/scripts/apply_label_map.py",
     "--label-map", "paper/config/label_maps/cd8_gt_map.yaml",
     "--text", "CD8.c02.Tm.IL7R"])

run([sys.executable, "paper/scripts/apply_label_map.py",
     "--label-map", "paper/config/label_maps/cd8_gt_map.yaml",
     "--csv", "paper/source_data/subsampled_ids/cd8_cells.csv",
     "--col", "meta.cluster",
     "--out", "paper/source_data/subsampled_ids/cd8_cells.with_gt.csv",
     "--out-col", "Ground_Truth"])

run([sys.executable, "paper/scripts/apply_label_map.py",
     "--label-map", "paper/config/label_maps/cd4_gt_map.yaml",
     "--csv", "paper/source_data/subsampled_ids/cd4_cells.csv",
     "--col", "meta.cluster",
     "--out", "paper/source_data/subsampled_ids/cd4_cells.with_gt.csv",
     "--out-col", "Ground_Truth"])

run([sys.executable, "paper/scripts/apply_label_map.py",
     "--label-map", "paper/config/label_maps/brca_msc_gt_map.yaml",
     "--csv", "paper/source_data/subsampled_ids/brca_msc_cells.csv",
     "--col", "meta.cluster",
     "--out", "paper/source_data/subsampled_ids/brca_msc_cells.with_gt.csv",
     "--out-col", "Ground_Truth"])

run([sys.executable, "paper/scripts/apply_label_map.py",
     "--label-map", "paper/config/label_maps/mouse_b_gt_map.yaml",
     "--csv", "paper/source_data/subsampled_ids/mouse_b_cells.csv",
     "--col", "meta.cluster",
     "--out", "paper/source_data/subsampled_ids/mouse_b_cells.with_gt.csv",
     "--out-col", "Ground_Truth"])

df = pd.read_csv("paper/source_data/subsampled_ids/cd8_cells.with_gt.csv")
print(df["Ground_Truth"].value_counts(dropna=False).head(20))