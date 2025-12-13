
# hierarchical_scoring.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import pandas as pd


@dataclass
class HierarchyConfig:
    """
    Generic config for ontology / hierarchy-aware scoring.

    You can reuse this for CD8, CD4, B cells, myeloid, spatial tasks, etc.
    All task-specific logic is expressed as dictionaries here.
    """
    # 1) How to parse major lineage from prediction text
    #    e.g. {"T": ["t cell", "cd8", ...], "B": ["b cell", ...], ...}
    lineage_aliases: Dict[str, List[str]]

    # 2) How to parse state from prediction text
    #    e.g. {"Naive": ["naive", "tcm"], "EffMem": ["effector", "tem"], ...}
    state_aliases: Dict[str, List[str]]

    # 3) How to map GT label to expected (major, state)
    #    For simplicity: substring pattern → (major, state)
    #    e.g. [("naive", ("T", "Naive")), ("nk", ("NK", "EffMem")), ...]
    gt_rules: List[Tuple[str, Tuple[str, str]]] = field(default_factory=list)

    # 4) Hard cross-lineage penalties: exp_major → {forbidden_pred_majors}
    #    e.g. {"T": {"B", "Myeloid"}, "NK": {"B", "Myeloid"}}
    major_penalties: Dict[str, Set[str]] = field(default_factory=dict)

    # 5) Near-lineage pairs giving partial credit (e.g. T ↔ NK)
    #    e.g. {frozenset({"T", "NK"})}
    near_lineage_pairs: Set[frozenset] = field(default_factory=set)

    # 6) Failure keywords in prediction text → immediate 0
    failure_keywords: List[str] = field(default_factory=lambda: ["error", "unknown"])

    # 7) Weights for lineage / state contribution
    w_lineage: float = 0.7
    w_state: float = 0.3

    # 8) Defaults if GT does not match any rule
    default_major: str = "Other"
    default_state: str = "Other"


def _parse_major_lineage_generic(ans: str, cfg: HierarchyConfig) -> str:
    """Parse major lineage (T / B / NK / Myeloid / Other) from prediction text."""
    a = ans.lower()
    for major, keywords in cfg.lineage_aliases.items():
        if any(k in a for k in keywords):
            return major
    return cfg.default_major


def _parse_state_generic(ans: str, cfg: HierarchyConfig) -> str:
    """Parse state (Naive / EffMem / Resident / Exhausted / ISG / etc.) from prediction text."""
    a = ans.lower()
    for state, keywords in cfg.state_aliases.items():
        if any(k in a for k in keywords):
            return state
    return cfg.default_state


def _expected_major_state_generic(gt: str, cfg: HierarchyConfig) -> Tuple[str, str]:
    """
    Map Ground_Truth label to expected (major, state) according to gt_rules.
    Rules are applied in order; first match wins (substring match).
    """
    g = str(gt).lower()
    for pattern, (major, state) in cfg.gt_rules:
        if pattern.lower() in g:
            return major, state
    # Fallback if no rule matches
    return cfg.default_major, cfg.default_state


def score_hierarchical(
    row: pd.Series,
    col_name: str,
    cfg: HierarchyConfig,
) -> float:
    """
    Generic ontology-aware hierarchical scorer.

    - First evaluates major lineage (heavy weight).
    - Then evaluates state (lighter weight).
    - Strong penalties for clearly wrong lineages (e.g. T→B, T→Myeloid).
    - Partial credit between near lineages (e.g. T↔NK).
    """
    raw_ans = str(row[col_name])
    ans = raw_ans.lower().replace("*", "").replace("\n", " ").strip()

    # 0) Kill switch
    if any(k in ans for k in cfg.failure_keywords):
        return 0.0

    exp_major, exp_state = _expected_major_state_generic(row["Ground_Truth"], cfg)
    pred_major = _parse_major_lineage_generic(ans, cfg)
    pred_state = _parse_state_generic(ans, cfg)

    # 1) Major lineage scoring
    #    Hard cross penalties
    forbidden = cfg.major_penalties.get(exp_major, set())
    if pred_major in forbidden:
        return 0.0

    if pred_major == exp_major:
        lineage_score = 1.0
    elif frozenset({pred_major, exp_major}) in cfg.near_lineage_pairs:
        # e.g. T ↔ NK
        lineage_score = 0.5
    else:
        lineage_score = 0.0

    # 2) State scoring
    state_score = 1.0 if pred_state == exp_state else 0.0

    # 3) Weighted combination (range [0, 1])
    total = cfg.w_lineage * lineage_score + cfg.w_state * state_score
    total = float(max(0.0, min(1.0, total)))
    return total
