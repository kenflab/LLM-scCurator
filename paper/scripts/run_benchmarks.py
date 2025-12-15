#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper/scripts/run_benchmarks.py

Optional (development / re-run) script to regenerate LLM-based outputs:
  - Standard (Top DE genes)
  - Curated  (LLM-scCurator feature distillation)

IMPORTANT:
- Paper reproduction should NOT depend on re-running LLMs or external references.
- The canonical figure pipeline is:
    source_data/benchmark_tables/*_SCORED.csv
      -> source_data/figure_data/*.csv
      -> paper/figures/*.pdf

This script writes "integrated-like" tables so downstream tooling can stay consistent.
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # pyyaml
except Exception as e:
    raise RuntimeError("pyyaml is required: pip install pyyaml") from e

try:
    import scanpy as sc
except Exception as e:
    raise RuntimeError("scanpy is required") from e

LOG = logging.getLogger("paper.run_benchmarks")


# -------------------------
# Helpers
# -------------------------
def set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        x = yaml.safe_load(f)
    if not isinstance(x, dict):
        raise ValueError(f"YAML must be a dict: {path}")
    return x


def read_ids_csv(path: Path) -> List[str]:
    # allow single-column csv with header or without
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()
    # common patterns
    for col in ["Barcode", "barcodes", "cell_id", "Cell_ID", "obs_names"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    # fallback: first column
    return df.iloc[:, 0].astype(str).tolist()


def ensure_llm_import(repo_root: Path) -> None:
    """Try import llm_sc_curator; if not installed, add repo root to sys.path."""
    try:
        import llm_sc_curator  # noqa: F401
        return
    except Exception:
        pass
    sys.path.insert(0, str(repo_root.resolve()))
    try:
        import llm_sc_curator  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import llm_sc_curator. "
            "Install your package (pip install -e .) or run from repo root."
        ) from e


def ensure_json_result(x: Any) -> Dict[str, str]:
    """Normalize LLM response into a dict with {cell_type, confidence, reasoning}."""
    if isinstance(x, dict):
        return {
            "cell_type": str(x.get("cell_type", "Unknown")),
            "confidence": str(x.get("confidence", "Low")),
            "reasoning": str(x.get("reasoning", "")),
        }
    if isinstance(x, str):
        return {"cell_type": x, "confidence": "Low", "reasoning": ""}
    return {"cell_type": "Error", "confidence": "Low", "reasoning": repr(x)}


# -------------------------
# GT mapping (robust)
# -------------------------
def load_gt_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Accept mapping YAML values as:
      - "CD8_Naive" (string)
      - {label: "...", keywords: "...", major: "...", state: "...", used_in_confusion: true}
    Returns:
      cluster_id -> dict with keys:
        label, keywords, major, state, used_in_confusion
    """
    raw = load_yaml(path)
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        cid = str(k)
        if isinstance(v, dict):
            out[cid] = {
                "label": str(v.get("label", "Unknown")),
                "keywords": str(v.get("keywords", "")),
                "major": str(v.get("major", "")),
                "state": str(v.get("state", "")),
                "used_in_confusion": bool(v.get("used_in_confusion", True)),
            }
        else:
            out[cid] = {
                "label": str(v),
                "keywords": "",
                "major": "",
                "state": "",
                "used_in_confusion": True,
            }
    return out


# -------------------------
# DE cache (once per dataset)
# -------------------------
def compute_rank_cache(
    adata: "sc.AnnData",
    cluster_col: str,
    key_added: str,
    method: str = "wilcoxon",
    use_raw: bool = False,
) -> None:
    if key_added in adata.uns:
        LOG.info("DE cache already exists: adata.uns[%s]", key_added)
        return
    LOG.info("Computing DE ranks once: groupby=%s key=%s", cluster_col, key_added)
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_col,
        reference="rest",
        method=method,
        use_raw=use_raw,
        key_added=key_added,
    )


def top_genes_for_cluster(adata: "sc.AnnData", cluster_name: str, key_added: str, n: int) -> List[str]:
    df = sc.get.rank_genes_groups_df(adata, group=cluster_name, key=key_added)
    if "names" not in df.columns:
        return []
    return df["names"].dropna().astype(str).tolist()[:n]


# -------------------------
# Config model
# -------------------------
@dataclass(frozen=True)
class DatasetCfg:
    tag: str
    adata_path: Path
    cluster_col: str
    gt_map_path: Path
    subsampled_ids_csv: Optional[Path]
    n_genes: int
    de_key: str
    model_name: str
    sleep_sec: float
    use_raw: bool
    resume: bool


def parse_dataset_cfg(d: Dict[str, Any], base_dir: Path, resume_default: bool) -> DatasetCfg:
    def p(x: Optional[str]) -> Optional[Path]:
        if x is None:
            return None
        return (base_dir / x).resolve()

    return DatasetCfg(
        tag=str(d["tag"]),
        adata_path=(base_dir / str(d["adata_path"])).resolve(),
        cluster_col=str(d.get("cluster_col", "meta.cluster")),
        gt_map_path=(base_dir / str(d["gt_map_path"])).resolve(),
        subsampled_ids_csv=p(d.get("subsampled_ids_csv")),
        n_genes=int(d.get("n_genes", 50)),
        de_key=str(d.get("de_key", "de_cache")),
        model_name=str(d.get("model_name", "models/gemini-2.5-pro")),
        sleep_sec=float(d.get("sleep_sec", 0.0)),
        use_raw=bool(d.get("use_raw", False)),
        resume=bool(d.get("resume", resume_default)),
    )


# -------------------------
# Core
# -------------------------
def run_dataset(
    cfg: DatasetCfg,
    repo_root: Path,
    out_results_dir: Path,
    api_key: str,
    seed: int,
    cache_dir: Optional[Path] = None,
) -> Path:
    ensure_llm_import(repo_root)
    from llm_sc_curator import LLMscCurator  # type: ignore

    safe_mkdir(out_results_dir)
    if cache_dir is not None:
        safe_mkdir(cache_dir)

    out_csv = out_results_dir / f"{cfg.tag}_benchmark_results_integrated.csv"
    progress_csv = out_results_dir / f"{cfg.tag}_benchmark_results_integrated.progress.csv"

    LOG.info("=== %s ===", cfg.tag)
    LOG.info("Loading AnnData: %s", cfg.adata_path)
    adata = sc.read_h5ad(str(cfg.adata_path))

    if cfg.subsampled_ids_csv is not None:
        ids = set(read_ids_csv(cfg.subsampled_ids_csv))
        before = adata.n_obs
        adata = adata[adata.obs_names.astype(str).isin(ids)].copy()
        LOG.info("Subsample applied: %d -> %d cells (ids=%s)", before, adata.n_obs, cfg.subsampled_ids_csv.name)

    if cfg.cluster_col not in adata.obs.columns:
        raise ValueError(f"cluster_col '{cfg.cluster_col}' not found in adata.obs")

    gt_map = load_gt_map(cfg.gt_map_path)

    compute_rank_cache(adata, cluster_col=cfg.cluster_col, key_added=cfg.de_key, use_raw=cfg.use_raw)

    curator = LLMscCurator(api_key=api_key, model_name=cfg.model_name)
    curator.set_global_context(adata)

    clusters = sorted(pd.unique(adata.obs[cfg.cluster_col].astype(str)))
    LOG.info("Clusters: %d", len(clusters))

    # Resume
    done = set()
    rows: List[Dict[str, Any]] = []
    if cfg.resume and progress_csv.exists():
        prev = pd.read_csv(progress_csv)
        if "Cluster_ID" in prev.columns:
            done = set(prev["Cluster_ID"].astype(str))
            rows = prev.to_dict(orient="records")
        LOG.info("Resume: loaded %d completed clusters", len(done))

    def cached_call(mode: str, genes: List[str], cluster_id: str) -> Dict[str, str]:
        if cache_dir is None:
            # no cache
            if mode == "standard":
                raw = curator.annotate(genes, use_auto_context=False) if genes else {"cell_type": "NoGenes"}
            else:
                raw = curator.annotate(genes, use_auto_context=True) if genes else {"cell_type": "NoGenes"}
            return ensure_json_result(raw)

        key = sha1_text("|".join([cfg.tag, cluster_id, mode, cfg.model_name, ";".join(genes)]))[:16]
        f = cache_dir / f"{cfg.tag}.{cluster_id}.{mode}.{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        if mode == "standard":
            raw = curator.annotate(genes, use_auto_context=False) if genes else {"cell_type": "NoGenes"}
        else:
            raw = curator.annotate(genes, use_auto_context=True) if genes else {"cell_type": "NoGenes"}
        res = ensure_json_result(raw)
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res

    for idx, cluster_id in enumerate(clusters, start=1):
        if cluster_id in done:
            continue

        gt = gt_map.get(cluster_id, {"label": "Unknown", "keywords": "", "major": "", "state": "", "used_in_confusion": True})
        gt_label = str(gt.get("label", "Unknown"))

        LOG.info("[%d/%d] %s (GT=%s)", idx, len(clusters), cluster_id, gt_label)

        genes_std = top_genes_for_cluster(adata, cluster_id, key_added=cfg.de_key, n=cfg.n_genes)
        try:
            res_std = cached_call("standard", genes_std, cluster_id)
        except Exception as e:
            res_std = ensure_json_result({"cell_type": "Error", "confidence": "Low", "reasoning": str(e)})

        if cfg.sleep_sec:
            time.sleep(cfg.sleep_sec)

        try:
            genes_cur = curator.curate_features(
                adata,
                group_col=cfg.cluster_col,
                target_group=cluster_id,
                n_top=cfg.n_genes,
                use_statistics=True,
            )
        except Exception as e:
            LOG.warning("curate_features failed for %s: %s", cluster_id, e)
            genes_cur = []

        try:
            res_cur = cached_call("curated", list(map(str, genes_cur)), cluster_id)
        except Exception as e:
            res_cur = ensure_json_result({"cell_type": "Error", "confidence": "Low", "reasoning": str(e)})

        row = {
            # identifiers
            "Cluster_ID": cluster_id,
            "Ground_Truth": gt_label,
            "Ground_Truth_Label": gt_label,
            "Ground_Truth_Keywords": str(gt.get("keywords", "")),

            # Standard
            "Standard_Genes": ";".join(map(str, genes_std)),
            "Standard_Answer": res_std["cell_type"],
            "Standard_CellType": res_std["cell_type"],
            "Standard_Confidence": res_std["confidence"],
            "Standard_Reasoning": res_std["reasoning"],

            # Curated
            "Curated_Genes": ";".join(map(str, genes_cur)),
            "Curated_Answer": res_cur["cell_type"],
            "Curated_CellType": res_cur["cell_type"],
            "Curated_Confidence": res_cur["confidence"],
            "Curated_Reasoning": res_cur["reasoning"],

            # Baselines (optional / left blank here)
            "SingleR_Answer": "",
            "Azimuth_Answer": "",
            "CellTypist_Answer": "",

            # For downstream scoring/plots (optional in GT map)
            "GT_Major": str(gt.get("major", "")),
            "GT_State": str(gt.get("state", "")),
            "UsedInConfusion": bool(gt.get("used_in_confusion", True)),
        }
        rows.append(row)

        # progress save (every cluster; small tables)
        pd.DataFrame(rows).to_csv(progress_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if progress_csv.exists():
        progress_csv.unlink()

    LOG.info("Wrote: %s", out_csv)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="YAML config (datasets list)")
    ap.add_argument("--repo-root", default=".", type=str, help="Repo root (for importing llm_sc_curator)")
    ap.add_argument("--out-results", default="paper/results", type=str, help="Output dir for results/*.csv")
    ap.add_argument("--datasets", default="all", type=str, help="Comma-separated dataset tags (default: all)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY", type=str)
    ap.add_argument("--cache-dir", default="", type=str, help="Optional cache dir for LLM calls")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    set_global_seeds(args.seed)

    repo_root = Path(args.repo_root).resolve()
    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent

    out_results = Path(args.out_results).resolve()
    safe_mkdir(out_results)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"API key missing. Set env var: {args.api_key_env}")

    cfg = load_yaml(cfg_path)
    ds_list = cfg.get("datasets", [])
    if not ds_list:
        raise ValueError("Config must contain: datasets: [...]")

    selected: Optional[set] = None
    if args.datasets != "all":
        selected = set([x.strip() for x in args.datasets.split(",") if x.strip()])

    for d in ds_list:
        ds = parse_dataset_cfg(d, base_dir=base_dir, resume_default=args.resume)
        if selected is not None and ds.tag not in selected:
            continue
        run_dataset(ds, repo_root=repo_root, out_results_dir=out_results, api_key=api_key, seed=args.seed, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
