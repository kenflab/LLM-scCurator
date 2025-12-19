#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper/scripts/run_benchmarks.py

Optional (advanced) script to regenerate benchmark intermediates from large public inputs.

What it does
- Runs the Standard baseline and LLM-scCurator on per-dataset inputs defined in a YAML config.
- Writes per-cluster outputs to `paper/results/` for inspection/debugging.

What it does NOT do
- It is not required to verify the manuscript figures.
  The canonical numeric values used for plotting are already exported under `paper/source_data/`
  and indexed by `paper/FIGURE_MAP.csv`.

Notes on determinism
- Local preprocessing can be made deterministic by fixing random seeds.
- LLM API outputs may still vary across runs even with temperature=0; treat regenerated runs as
  best used for auditing and sensitivity checks, not as the manuscript ground truth.
"""

from __future__ import annotations

import os
import csv
import sys
import time
import json
import hashlib
import argparse
import logging
from dataclasses import dataclass, field
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
# GT mapping 
# -------------------------
def load_gt_map(path: Path) -> Any:
    """
    Supports TWO YAML formats:

    (A) Mapping YAML (old format)
        <cluster_id>: "CD8_Naive"
        <cluster_id>:
          label: CD8_Naive
          keywords: "..."
          major: "..."
          state: "..."
          used_in_confusion: true

    (B) Rules YAML (new format)
        version: 1
        dataset_id: pancancer_cd8
        input_key: meta.cluster
        matching:
          mode: substring_ci
          first_match_wins: true
        rules:
          - label: CD8_MAIT
            include_any: ["mait"]
          - label: CD8_Naive
            include_any: ["tn.", "naive"]
        default_label: CD8_Other

    Returns an object that supports:
        gt_map.get(cluster_id, default_dict) -> dict(label, keywords, major, state, used_in_confusion)

    Notes:
      - Only matching.mode=substring_ci is supported (case-insensitive substring matching).
      - For rules:
          include:      AND (all patterns must match)
          include_any:  OR  (any pattern matches)
          exclude:      AND (rarely needed; treated as OR here would be surprising)
          exclude_any:  OR  (any pattern excludes)
    """
    raw = load_yaml(path)

    # -----------------
    # (B) Rules YAML
    # -----------------
    if isinstance(raw, dict) and isinstance(raw.get("rules", None), list):
        matching = raw.get("matching", {}) or {}
        mode = str(matching.get("mode", "substring_ci"))
        first_match_wins = bool(matching.get("first_match_wins", True))
        default_label = str(raw.get("default_label", "Unknown"))
        input_key = str(raw.get("input_key", "")).strip()
        dataset_id = str(raw.get("dataset_id", "")).strip()

        if mode != "substring_ci":
            raise ValueError(
                f"Unsupported matching.mode={mode} in {path} (only substring_ci is supported)."
            )

        def _as_list(x: Any) -> List[str]:
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return [str(i) for i in x]
            return [str(x)]

        def _match_any(s: str, pats: List[str]) -> bool:
            s = s.lower()
            return any(str(p).lower() in s for p in pats)

        def _match_all(s: str, pats: List[str]) -> bool:
            s = s.lower()
            return all(str(p).lower() in s for p in pats)

        def _label_to_major_state(label: str) -> Tuple[str, str]:
            parts = str(label).split("_", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            return "", ""

        parsed_rules: List[Dict[str, Any]] = []
        for rr in raw.get("rules", []):
            if not isinstance(rr, dict) or "label" not in rr:
                continue

            label = str(rr["label"])
            inc = _as_list(rr.get("include"))
            inc_any = _as_list(rr.get("include_any"))
            exc = _as_list(rr.get("exclude"))
            exc_any = _as_list(rr.get("exclude_any"))
            used = bool(rr.get("used_in_confusion", True))

            keywords = ";".join([*inc, *inc_any])
            major, state = _label_to_major_state(label)

            parsed_rules.append(
                dict(
                    label=label,
                    include=inc,
                    include_any=inc_any,
                    exclude=exc,
                    exclude_any=exc_any,
                    record={
                        "label": label,
                        "keywords": keywords,
                        "major": major,
                        "state": state,
                        "used_in_confusion": used,
                    },
                )
            )

        class _RuleBasedGTMap:
            def __init__(self) -> None:
                self.input_key = input_key
                self.dataset_id = dataset_id

            def get(self, cluster_id: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                s = str(cluster_id).lower()

                chosen: Optional[Dict[str, Any]] = None
                for rr in parsed_rules:
                    # exclusions (OR)
                    if rr["exclude"] and _match_any(s, rr["exclude"]):
                        continue
                    if rr["exclude_any"] and _match_any(s, rr["exclude_any"]):
                        continue

                    # inclusions
                    if rr["include"] and not _match_all(s, rr["include"]):
                        continue
                    if rr["include_any"] and not _match_any(s, rr["include_any"]):
                        continue

                    chosen = rr
                    if first_match_wins:
                        break

                if chosen is None:
                    major, state = _label_to_major_state(default_label)
                    return {
                        "label": default_label,
                        "keywords": "",
                        "major": major,
                        "state": state,
                        "used_in_confusion": True,
                    }

                return dict(chosen["record"])

        return _RuleBasedGTMap()

    # -----------------
    # (A) Mapping YAML (backward compatible)
    # -----------------
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        raise ValueError(f"GT map YAML must be a dict at top-level: {path}")

    meta_keys = {"version", "dataset_id", "input_key", "matching", "labels", "rules", "default_label"}
    for k, v in raw.items():
        if str(k) in meta_keys:
            continue

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

    # Optional reproducibility / filtering knobs
    exclude_gt_labels: Tuple[str, ...] = ()
    hvg_kwargs: Optional[Dict[str, Any]] = None


def parse_dataset_cfg(d: Dict[str, Any], base_dir: Path, resume_default: bool) -> DatasetCfg:
    def p(x: Optional[str]) -> Optional[Path]:
        if x is None:
            return None
        return (base_dir / x).resolve()

    exclude_gt_labels_raw = d.get("exclude_gt_labels", []) or []
    if isinstance(exclude_gt_labels_raw, (str, int, float)):
        exclude_gt_labels_raw = [exclude_gt_labels_raw]
    if not isinstance(exclude_gt_labels_raw, list):
        raise ValueError("exclude_gt_labels must be a list (or a single string).")

    hvg_kwargs = d.get("hvg_kwargs", None)
    if hvg_kwargs is not None and not isinstance(hvg_kwargs, dict):
        raise ValueError("hvg_kwargs must be a dict compatible with scanpy.pp.highly_variable_genes")

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
        exclude_gt_labels=tuple(map(str, exclude_gt_labels_raw)),
        hvg_kwargs=hvg_kwargs,
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
    dropped_csv = out_results_dir / f"{cfg.tag}_benchmark_dropped.csv"
    meta_json = out_results_dir / f"{cfg.tag}_benchmark_run_metadata.json"

    # If a previous run finished cleanly, keep it (avoid accidental recompute).
    if cfg.resume and out_csv.exists() and not progress_csv.exists():
        LOG.info("Output exists and no progress file; assuming complete: %s", out_csv)
        return out_csv

    LOG.info("=== %s ===", cfg.tag)
    LOG.info("Loading AnnData: %s", cfg.adata_path)
    if not cfg.adata_path.exists():
        raise FileNotFoundError(cfg.adata_path)
    adata = sc.read_h5ad(str(cfg.adata_path))

    if cfg.cluster_col not in adata.obs.columns:
        raise ValueError(f"cluster_col '{cfg.cluster_col}' not found in adata.obs")

    # Optional balanced subsampling (for reproducible cost/time)
    if cfg.subsampled_ids_csv is not None:
        if not cfg.subsampled_ids_csv.exists():
            raise FileNotFoundError(cfg.subsampled_ids_csv)
        keep_ids = set(read_ids_csv(cfg.subsampled_ids_csv))
        adata = adata[adata.obs_names.isin(keep_ids)].copy()
        LOG.info("Subsampled to %d cells using: %s", adata.n_obs, cfg.subsampled_ids_csv)

    # Optional: compute HVGs with a paper-matching recipe if requested.
    # (Not required for correctness; curator can compute on-the-fly.)
    if cfg.hvg_kwargs is not None and "highly_variable" not in adata.var.columns:
        hvg_kwargs = dict(cfg.hvg_kwargs)
        batch_key = hvg_kwargs.get("batch_key", None)
        if batch_key is not None and str(batch_key) not in adata.obs.columns:
            LOG.warning("HVG batch_key=%s not found in adata.obs; computing HVGs without batch_key.", batch_key)
            hvg_kwargs.pop("batch_key", None)
        layer = hvg_kwargs.get("layer", None)
        if layer is not None and layer not in adata.layers:
            LOG.warning("HVG layer=%s not found in adata.layers; computing HVGs without layer.", layer)
            hvg_kwargs.pop("layer", None)
        LOG.info("Computing HVGs (as requested): %s", hvg_kwargs)
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)

    gt_map = load_gt_map(cfg.gt_map_path)
    input_key = getattr(gt_map, "input_key", "")
    if input_key and input_key != cfg.cluster_col:
        LOG.warning("GT map input_key=%s but dataset cluster_col=%s (continuing).", input_key, cfg.cluster_col)

    # Rank markers once per dataset
    compute_rank_cache(adata, cluster_col=cfg.cluster_col, key_added=cfg.de_key, method="wilcoxon", use_raw=cfg.use_raw)

    curator = LLMscCurator(api_key=api_key, model_name=cfg.model_name)
    curator.set_global_context(adata)

    clusters = list(pd.unique(adata.obs[cfg.cluster_col]))
    clusters = sorted(clusters, key=lambda x: str(x))
    LOG.info("Clusters: %d", len(clusters))

    # Resume from progress file if present
    done: set[str] = set()
    if cfg.resume and progress_csv.exists():
        prev = pd.read_csv(progress_csv, usecols=["Cluster_ID"])
        done = set(prev["Cluster_ID"].astype(str))
        LOG.info("Resume: loaded %d completed clusters", len(done))

    # Stream results to progress CSV (fast + robust)
    fieldnames = [
        "Cluster_ID",
        "Ground_Truth", "Ground_Truth_Label", "Ground_Truth_Keywords",
        "Standard_Genes", "Standard_Answer", "Standard_CellType", "Standard_Confidence", "Standard_Reasoning",
        "Curated_Genes", "Curated_Answer", "Curated_CellType", "Curated_Confidence", "Curated_Reasoning",
        "SingleR_Answer", "Azimuth_Answer", "CellTypist_Answer",
        "GT_Major", "GT_State", "UsedInConfusion",
    ]
    write_header = (not progress_csv.exists()) or (progress_csv.stat().st_size == 0)

    def cached_call(mode: str, genes: List[str], cluster_id: str) -> Dict[str, str]:
        if cache_dir is None:
            raw = curator.annotate(genes, use_auto_context=(mode != "standard")) if genes else {"cell_type": "NoGenes"}
            return ensure_json_result(raw)

        key = sha1_text("|".join([cfg.tag, cluster_id, mode, cfg.model_name, ";".join(genes)]))[:16]
        f = cache_dir / f"{cfg.tag}.{cluster_id}.{mode}.{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass

        raw = curator.annotate(genes, use_auto_context=(mode != "standard")) if genes else {"cell_type": "NoGenes"}
        res = ensure_json_result(raw)
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res

    dropped: List[Dict[str, Any]] = []

    with progress_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            fh.flush()

        for idx, cluster_id in enumerate(clusters, start=1):
            cluster_id_str = str(cluster_id)
            if cluster_id_str in done:
                continue

            gt = gt_map.get(
                cluster_id_str,
                {"label": "Unknown", "keywords": "", "major": "", "state": "", "used_in_confusion": True},
            )
            gt_label = str(gt.get("label", "Unknown"))

            # Optional: skip clusters from evaluation (still auditable via dropped_csv)
            if cfg.exclude_gt_labels and gt_label in cfg.exclude_gt_labels:
                LOG.info("[%d/%d] %s (GT=%s) -> excluded", idx, len(clusters), cluster_id_str, gt_label)
                dropped.append({"Cluster_ID": cluster_id_str, "Ground_Truth_Label": gt_label, "Reason": "exclude_gt_labels"})
                continue

            LOG.info("[%d/%d] %s (GT=%s)", idx, len(clusters), cluster_id_str, gt_label)

            genes_std = top_genes_for_cluster(adata, cluster_id, key_added=cfg.de_key, n=cfg.n_genes)
            try:
                res_std = cached_call("standard", genes_std, cluster_id_str)
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
                LOG.warning("curate_features failed for %s: %s", cluster_id_str, e)
                genes_cur = []

            try:
                res_cur = cached_call("curated", list(map(str, genes_cur)), cluster_id_str)
            except Exception as e:
                res_cur = ensure_json_result({"cell_type": "Error", "confidence": "Low", "reasoning": str(e)})

            row = {
                "Cluster_ID": cluster_id_str,
                "Ground_Truth": gt_label,
                "Ground_Truth_Label": gt_label,
                "Ground_Truth_Keywords": str(gt.get("keywords", "")),

                "Standard_Genes": ";".join(map(str, genes_std)),
                "Standard_Answer": res_std["cell_type"],
                "Standard_CellType": res_std["cell_type"],
                "Standard_Confidence": res_std["confidence"],
                "Standard_Reasoning": res_std["reasoning"],

                "Curated_Genes": ";".join(map(str, genes_cur)),
                "Curated_Answer": res_cur["cell_type"],
                "Curated_CellType": res_cur["cell_type"],
                "Curated_Confidence": res_cur["confidence"],
                "Curated_Reasoning": res_cur["reasoning"],

                "SingleR_Answer": "",
                "Azimuth_Answer": "",
                "CellTypist_Answer": "",

                "GT_Major": str(gt.get("major", "")),
                "GT_State": str(gt.get("state", "")),
                "UsedInConfusion": bool(gt.get("used_in_confusion", True)),
            }

            writer.writerow(row)
            fh.flush()

    # Materialize final CSV
    if progress_csv.exists() and progress_csv.stat().st_size > 0:
        os.replace(progress_csv, out_csv)

    if dropped:
        pd.DataFrame(dropped).to_csv(dropped_csv, index=False)

    # Minimal run metadata (helps auditing)
    try:
        script_text = Path(__file__).read_text(encoding="utf-8")
        script_sha1 = sha1_text(script_text)
    except Exception:
        script_sha1 = ""

    meta = {
        "dataset_tag": cfg.tag,
        "adata_path": str(cfg.adata_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "cluster_col": cfg.cluster_col,
        "n_clusters": int(len(clusters)),
        "gt_map_path": str(cfg.gt_map_path),
        "gt_dataset_id": getattr(gt_map, "dataset_id", ""),
        "gt_input_key": getattr(gt_map, "input_key", ""),
        "model_name": cfg.model_name,
        "seed": int(seed),
        "exclude_gt_labels": list(cfg.exclude_gt_labels),
        "hvg_kwargs": cfg.hvg_kwargs,
        "script_sha1": script_sha1,
        "python": sys.version,
        "scanpy": getattr(sc, "__version__", ""),
        "pandas": getattr(pd, "__version__", ""),
        "numpy": getattr(np, "__version__", ""),
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

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
