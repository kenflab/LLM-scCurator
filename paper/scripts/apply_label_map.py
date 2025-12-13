#!/usr/bin/env python3
# paper/scripts/apply_label_map.py
"""
Apply a simple YAML label-map (case-insensitive substring rules) to strings or CSV columns.

YAML schema (expected):
  version: 1
  dataset_id: <str>
  input_key: <str>         # e.g., "meta.cluster"
  matching:
    mode: substring_ci
    first_match_wins: true
  rules:
    - label: <str>
      include: [<substr>, ...]          # ALL must be present
      include_any: [<substr>, ...]      # ANY must be present
      exclude: [<substr>, ...]          # ANY present => fail (treated as exclude_any)
      exclude_any: [<substr>, ...]      # ANY present => fail
  default_label: <str>

Rule matching:
  - A rule matches if:
      (include ALL satisfied OR no include) AND
      (include_any ANY satisfied OR no include_any) AND
      (no exclude/exclude_any hits)
  - Rules are applied in order; first match wins (if configured).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


def _norm(s: str) -> str:
    return str(s).lower()


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return [str(x)]


@dataclass(frozen=True)
class LabelRule:
    label: str
    include: List[str]
    include_any: List[str]
    exclude_any: List[str]


@dataclass(frozen=True)
class LabelMap:
    dataset_id: str
    input_key: str
    mode: str
    first_match_wins: bool
    rules: List[LabelRule]
    default_label: str


def _compile_label_map(d: Dict[str, Any]) -> LabelMap:
    dataset_id = str(d.get("dataset_id", "")).strip()
    input_key = str(d.get("input_key", "")).strip()

    matching = d.get("matching", {}) or {}
    mode = str(matching.get("mode", "substring_ci")).strip()
    first_match_wins = bool(matching.get("first_match_wins", True))

    default_label = str(d.get("default_label", "")).strip()
    if not default_label:
        raise ValueError("label map is missing 'default_label'.")

    raw_rules = d.get("rules", [])
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("label map must contain a non-empty 'rules' list.")

    rules: List[LabelRule] = []
    for i, rr in enumerate(raw_rules):
        if not isinstance(rr, dict) or "label" not in rr:
            raise ValueError(f"rule #{i} must be a dict with a 'label' field.")
        label = str(rr["label"]).strip()

        include = _as_list(rr.get("include"))
        include_any = _as_list(rr.get("include_any"))

        # treat 'exclude' as exclude_any for simplicity
        exclude_any = _as_list(rr.get("exclude_any")) + _as_list(rr.get("exclude"))

        rules.append(
            LabelRule(
                label=label,
                include=[_norm(x) for x in include if str(x).strip()],
                include_any=[_norm(x) for x in include_any if str(x).strip()],
                exclude_any=[_norm(x) for x in exclude_any if str(x).strip()],
            )
        )

    return LabelMap(
        dataset_id=dataset_id,
        input_key=input_key,
        mode=mode,
        first_match_wins=first_match_wins,
        rules=rules,
        default_label=default_label,
    )


@lru_cache(maxsize=128)
def load_label_map(label_map_path: str) -> LabelMap:
    p = Path(label_map_path)
    if not p.exists():
        raise FileNotFoundError(f"Label map not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"Invalid YAML (expected a dict) in: {p}")
    return _compile_label_map(d)


def apply_label_map_to_text(text: Any, lm: LabelMap) -> str:
    if text is None:
        return lm.default_label

    if lm.mode != "substring_ci":
        raise ValueError(f"Unsupported matching mode: {lm.mode}")

    t = _norm(str(text))

    for rule in lm.rules:
        # Exclusions (ANY hit => fail)
        if any(x in t for x in rule.exclude_any):
            continue

        # include (ALL must be present) if provided
        if rule.include and not all(x in t for x in rule.include):
            continue

        # include_any (ANY must be present) if provided
        if rule.include_any and not any(x in t for x in rule.include_any):
            continue

        return rule.label

    return lm.default_label


def apply_label_map_to_series(series: pd.Series, lm: LabelMap) -> pd.Series:
    return series.apply(lambda x: apply_label_map_to_text(x, lm))


def apply_label_map_to_csv(
    csv_path: str,
    out_path: str,
    label_map_path: str,
    input_col: str,
    output_col: str = "Ground_Truth",
) -> None:
    lm = load_label_map(label_map_path)
    df = pd.read_csv(csv_path)
    if input_col not in df.columns:
        raise ValueError(f"Column '{input_col}' not found in {csv_path}. Available: {list(df.columns)}")
    df[output_col] = apply_label_map_to_series(df[input_col], lm)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a YAML label_map to a string or CSV column.")
    ap.add_argument("--label-map", required=True, help="Path to YAML label map.")
    ap.add_argument("--text", default=None, help="Single text value to map (optional).")
    ap.add_argument("--csv", default=None, help="Input CSV path (optional).")
    ap.add_argument("--col", default=None, help="Input column name (required if --csv).")
    ap.add_argument("--out", default=None, help="Output CSV path (required if --csv).")
    ap.add_argument("--out-col", default="Ground_Truth", help="Output column name for CSV mode.")
    args = ap.parse_args()

    lm = load_label_map(args.label_map)

    if args.text is not None:
        print(apply_label_map_to_text(args.text, lm))
        return

    if args.csv is not None:
        if not args.col or not args.out:
            raise ValueError("--csv mode requires --col and --out")
        apply_label_map_to_csv(args.csv, args.out, args.label_map, args.col, args.out_col)
        print(f"[OK] wrote: {args.out}")
        return

    raise ValueError("Provide either --text or --csv/--col/--out.")


if __name__ == "__main__":
    main()
