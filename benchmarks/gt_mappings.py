# LLM-scCurator/benchmarks/gt_mappings.py
"""
Ground-truth mapping for benchmarks.

Source of truth:
  paper/config/label_maps/*.yaml

These functions remain as thin wrappers so older evaluation code keeps working.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Reuse the YAML matcher (single source of implementation)
# (repo layout assumed)
#   paper/scripts/apply_label_map.py
#   paper/config/label_maps/*.yaml
try:
    from paper.scripts.apply_label_map import load_label_map, apply_label_map_to_text
except Exception:
    # Fallback for when "paper" isn't importable as a package (common in scripts).
    # We load by relative path from this file's location.
    import importlib.util

    _HERE = Path(__file__).resolve()
    _REPO_ROOT = _HERE.parents[2]  # LLM-scCurator/benchmarks/ -> repo root
    _APPLY = _REPO_ROOT / "paper" / "scripts" / "apply_label_map.py"

    if not _APPLY.exists():
        raise ImportError(
            f"Could not import apply_label_map.py. Expected at: {_APPLY}"
        )

    spec = importlib.util.spec_from_file_location("apply_label_map", str(_APPLY))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore

    load_label_map = mod.load_label_map
    apply_label_map_to_text = mod.apply_label_map_to_text


@lru_cache(maxsize=32)
def _label_map_path(name: str) -> str:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../LLM-scCurator/benchmarks -> repo root
    p = repo_root / "paper" / "config" / "label_maps" / name
    return str(p)


@lru_cache(maxsize=32)
def _lm(name: str):
    return load_label_map(_label_map_path(name))


def get_cd8_ground_truth(cluster_name: str) -> str:
    lm = _lm("pancancer_cd8_gt_map.yaml")
    return apply_label_map_to_text(cluster_name, lm)


def get_cd4_ground_truth(cluster_name: str) -> str:
    lm = _lm("pancancer_cd4_gt_map.yaml")
    return apply_label_map_to_text(cluster_name, lm)


def get_msc_ground_truth(cluster_name: str) -> str:
    lm = _lm("brca_stromal_gt_map.yaml")
    return apply_label_map_to_text(cluster_name, lm)


def get_bcell_ground_truth(cluster_name: str) -> str:
    lm = _lm("tabula_muris_b_gt_map.yaml")
    return apply_label_map_to_text(cluster_name, lm)
