#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export subsampled cell IDs (and minimal metadata) from .h5ad into Source Data CSVs.

Two modes:
  A) Single dataset mode (explicit args):
     python export_subsampled_ids.py --h5ad path/to/file.h5ad --dataset-id pancancer_cd8 \
       --out paper/source_data/subsampled_ids/pancancer_cd8_cells.csv \
       --cols Cancer_Type,meta.cluster

  B) Batch mode (datasets.tsv):
     python export_subsampled_ids.py --datasets-tsv paper/config/datasets.tsv

The CSV is designed to be safe to host on GitHub (no expression matrices),
while enabling exact reconstruction of the analyzed subset by ID.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


def _base_id_guess(s: str) -> str:
    """
    Heuristic for scanpy/anndata obs_names_make_unique suffixes.
    Example: AAAC...-1-1 -> AAAC...-1 (drops trailing -<digit>)
    """
    parts = str(s).split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    return str(s)


def _split_cols(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # accept comma-separated or repeated flags in future
    return [c.strip() for c in s.split(",") if c.strip()]


def _safe_makedirs(path: str) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)


@dataclass
class DatasetRow:
    dataset_id: str
    h5ad_path: str
    out_csv: str
    cols: List[str]


def _normalize_header(h: str) -> str:
    return h.strip().lower().replace(" ", "_").replace("-", "_")


def _read_datasets_tsv(tsv_path: str) -> List[DatasetRow]:
    """
    Expected columns (case/space insensitive):
      - dataset_id
      - h5ad_path
      - out_csv
      - cols   (comma-separated; optional)
    Additional columns are ignored.

    Example datasets.tsv (minimum):
      dataset_id  h5ad_path  out_csv  cols
      pancancer_cd8  /abs/path/cd8.h5ad  paper/source_data/subsampled_ids/pancancer_cd8_cells.csv  Cancer_Type,meta.cluster
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"datasets.tsv not found: {tsv_path}")

    with open(tsv_path, "r", encoding="utf-8") as f:
        sniff = f.read(4096)
        f.seek(0)
        # allow either TSV or CSV; default TSV
        dialect = csv.excel_tab if "\t" in sniff else csv.excel
        reader = csv.DictReader(f, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError(f"No header found in {tsv_path}")

        # map flexible headers to canonical
        header_map = {_normalize_header(h): h for h in reader.fieldnames}

        def get(row: dict, key: str, default: str = "") -> str:
            hk = _normalize_header(key)
            if hk in header_map:
                return str(row.get(header_map[hk], "")).strip()
            return default

        rows: List[DatasetRow] = []
        for i, r in enumerate(reader, start=2):
            dataset_id = get(r, "dataset_id")
            h5ad_path = get(r, "h5ad_path") or get(r, "h5ad") or get(r, "adata_path")
            out_csv = get(r, "out_csv") or get(r, "out") or ""

            cols_raw = get(r, "cols", "")
            cols = _split_cols(cols_raw)

            if not dataset_id or not h5ad_path or not out_csv:
                raise ValueError(
                    f"datasets.tsv row {i} missing required fields. "
                    f"Need dataset_id, h5ad_path, out_csv. Got: dataset_id={dataset_id!r}, "
                    f"h5ad_path={h5ad_path!r}, out_csv={out_csv!r}"
                )

            rows.append(DatasetRow(dataset_id=dataset_id, h5ad_path=h5ad_path, out_csv=out_csv, cols=cols))

    return rows


def export_subsampled_ids(
    h5ad_path: str,
    out_csv: str,
    dataset_id: str,
    keep_cols: List[str],
    id_col_out: str = "cell_id_in_h5ad",
    base_guess_col_out: str = "cell_id_base_guess",
    add_counts_summary: bool = True,
) -> pd.DataFrame:
    """
    Reads h5ad, exports a small CSV with:
      dataset_id, cell_id_in_h5ad, cell_id_base_guess, [keep_cols...]

    Returns the exported DataFrame (also written to disk).
    """
    import scanpy as sc  # local import to keep CLI snappy if not needed

    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"h5ad not found: {h5ad_path}")

    adata = sc.read_h5ad(h5ad_path)

    obs = adata.obs.copy()
    obs[id_col_out] = obs.index.astype(str)
    obs[base_guess_col_out] = obs[id_col_out].map(_base_id_guess)

    cols = [id_col_out, base_guess_col_out]
    for c in keep_cols:
        if c in obs.columns and c not in cols:
            cols.append(c)

    out = obs[cols].copy()
    out.insert(0, "dataset_id", dataset_id)

    _safe_makedirs(out_csv)
    out.to_csv(out_csv, index=False)

    print(f"[OK] {dataset_id}: {adata.n_obs} rows -> {out_csv}")
    if add_counts_summary and "meta.cluster" in out.columns:
        vc = out["meta.cluster"].value_counts().head(10)
        print("  Top meta.cluster counts:")
        for k, v in vc.items():
            print(f"    {k}: {v}")

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export subsampled cell IDs from .h5ad into Source Data CSV(s)."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--datasets-tsv", help="Batch mode: TSV/CSV describing datasets to export.")
    mode.add_argument("--h5ad", help="Single mode: path to a .h5ad file to export.")

    # single mode args
    p.add_argument("--dataset-id", help="Single mode: dataset identifier (e.g., pancancer_cd8).")
    p.add_argument("--out", dest="out_csv", help="Single mode: output CSV path.")
    p.add_argument(
        "--cols",
        default="",
        help="Single mode: comma-separated obs columns to keep (e.g., Cancer_Type,meta.cluster).",
    )

    # shared options
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse datasets and print planned outputs without writing CSV.",
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    jobs: List[DatasetRow] = []

    if args.datasets_tsv:
        jobs = _read_datasets_tsv(args.datasets_tsv)

    else:
        # single mode validation
        if not args.dataset_id or not args.out_csv:
            print("ERROR: --dataset-id and --out are required in single mode.", file=sys.stderr)
            return 2
        jobs = [
            DatasetRow(
                dataset_id=args.dataset_id,
                h5ad_path=args.h5ad,
                out_csv=args.out_csv,
                cols=_split_cols(args.cols),
            )
        ]

    if args.dry_run:
        print("[DRY RUN] Planned exports:")
        for j in jobs:
            print(f"  - dataset_id={j.dataset_id}")
            print(f"    h5ad_path={j.h5ad_path}")
            print(f"    out_csv={j.out_csv}")
            print(f"    cols={','.join(j.cols) if j.cols else '(none)'}")
        return 0

    for j in jobs:
        export_subsampled_ids(
            h5ad_path=j.h5ad_path,
            out_csv=j.out_csv,
            dataset_id=j.dataset_id,
            keep_cols=j.cols,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

