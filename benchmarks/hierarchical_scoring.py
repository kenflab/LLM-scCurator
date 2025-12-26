"""
Ontology-aware hierarchical scoring utilities.

This module provides a lightweight, backend-agnostic scoring layer for evaluating
LLM-derived (or otherwise free-text) cell-type annotations against a harmonized
ground truth. The scorer maps both ground-truth labels and prediction text to a
(common) ontology consisting of:

- a major lineage (e.g., T, B, NK, Myeloid), and
- a within-lineage state (e.g., Naive, EffMem, Exhausted, ISG).

Scoring is hierarchical:
1) Major lineage receives a higher weight (default 0.7).
2) State receives a lower weight (default 0.3).
3) Optional hard penalties enforce zero score for biologically incompatible
   lineage confusions (e.g., expected T predicted as B or Myeloid).
4) Optional near-lineage pairs allow partial credit (e.g., T ↔ NK).

The design is intentionally simple and deterministic to support reproducible
benchmarking and figure generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import pandas as pd


@dataclass
class HierarchyConfig:
    """
    Configuration for ontology-aware hierarchical scoring.

    All task-specific behavior is expressed through dictionaries in this config
    (aliases, ground-truth mapping rules, and penalty sets). This allows the same
    scoring function to be reused across immune, stromal, and spatial benchmarks.

    Attributes
    ----------
    lineage_aliases : dict[str, list[str]]
        Mapping from major lineage name to a list of substrings used to detect that
        lineage in free-text predictions (case-insensitive).
        Example: {"T": ["t cell", "cd8", "cd4"], "B": ["b cell"], ...}
    state_aliases : dict[str, list[str]]
        Mapping from state name to a list of substrings used to detect that state
        in free-text predictions (case-insensitive).
        Example: {"Naive": ["naive", "tcm"], "Exhausted": ["exhaust"], ...}
    gt_rules : list[tuple[str, tuple[str, str]]]
        Ordered rules mapping ground-truth label strings to an expected (major, state).
        Each rule is a substring match; the first match wins.
        Example: [("naive", ("T", "Naive")), ("nk", ("NK", "EffMem")), ...]
    major_penalties : dict[str, set[str]]
        Hard cross-lineage penalties. Keys are expected majors; values are predicted
        majors that receive an immediate score of 0.0 if matched.
    near_lineage_pairs : set[frozenset]
        Pairs of majors that receive partial credit if confused (e.g., T ↔ NK).
    failure_keywords : list[str]
        If any keyword appears in the prediction text (after normalization),
        the score is forced to 0.0 (e.g., "error", "unknown").
    w_lineage : float
        Weight assigned to major lineage correctness (default 0.7).
    w_state : float
        Weight assigned to state correctness (default 0.3).
    default_major : str
        Fallback major if ground truth does not match any rule or prediction cannot
        be parsed (default "Other").
    default_state : str
        Fallback state if ground truth does not match any rule or prediction cannot
        be parsed (default "Other").

    Notes
    -----
    This config is intentionally task-local and does not depend on external ontologies.
    For rigorous comparisons, keep the mapping rules stable across runs.
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
    """
    Parse the predicted major lineage from free-text output.

    Parameters
    ----------
    ans : str
        Raw prediction text (free-form).
    cfg : HierarchyConfig
        Task-specific ontology configuration.

    Returns
    -------
    str
        Predicted major lineage label. If no alias matches, returns `cfg.default_major`.
    """

    a = ans.lower()
    for major, keywords in cfg.lineage_aliases.items():
        if any(k in a for k in keywords):
            return major
    return cfg.default_major


def _parse_state_generic(ans: str, cfg: HierarchyConfig) -> str:
    """
    Parse the predicted within-lineage state from free-text output.

    Parameters
    ----------
    ans : str
        Raw prediction text (free-form).
    cfg : HierarchyConfig
        Task-specific ontology configuration.

    Returns
    -------
    str
        Predicted state label. If no alias matches, returns `cfg.default_state`.
    """

    a = ans.lower()
    for state, keywords in cfg.state_aliases.items():
        if any(k in a for k in keywords):
            return state
    return cfg.default_state


def _expected_major_state_generic(gt: str, cfg: HierarchyConfig) -> Tuple[str, str]:
    """
    Map a ground-truth label to the expected (major, state) ontology tuple.

    Rules are applied in order; the first substring match wins. If no rule matches,
    returns (cfg.default_major, cfg.default_state).

    Parameters
    ----------
    gt : str
        Ground-truth label (typically a harmonized category string).
    cfg : HierarchyConfig
        Task-specific ontology configuration.

    Returns
    -------
    tuple[str, str]
        (expected_major, expected_state)
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
    Compute an ontology-aware hierarchical score for one prediction.

    The score is computed in three steps:

    1) Fail-fast: if the normalized prediction text contains any `failure_keywords`,
       return 0.0.
    2) Major lineage scoring (weighted by `cfg.w_lineage`):
       - If the predicted major is in `major_penalties[expected_major]`, return 0.0.
       - If predicted major equals expected major: lineage_score = 1.0.
       - If (pred_major, exp_major) is in `near_lineage_pairs`: lineage_score = 0.5.
       - Otherwise: lineage_score = 0.0.
    3) State scoring (weighted by `cfg.w_state`):
       - state_score = 1.0 if predicted state equals expected state else 0.0.

    The final score is:
        total = w_lineage * lineage_score + w_state * state_score,
    clipped to [0.0, 1.0].

    Parameters
    ----------
    row : pandas.Series
        A row from the evaluation table. Must include:
        - `row[col_name]`: prediction text
        - `row["Ground_Truth"]`: ground-truth label
    col_name : str
        Column name containing the prediction text.
    cfg : HierarchyConfig
        Task-specific ontology configuration.

    Returns
    -------
    float
        Hierarchical score in the range [0.0, 1.0].

    Notes
    -----
    This scorer is deterministic and intentionally conservative:
    it assigns no partial credit for state if the state string does not match the
    expected state alias mapping.
    """
    
    raw_ans = str(row[col_name])
    ans = raw_ans.lower().replace("*", "").replace("\n", " ").strip()

    # 0) Fail-fast: explicit failure tokens
    if any(k in ans for k in cfg.failure_keywords):
        return 0.0

    exp_major, exp_state = _expected_major_state_generic(row["Ground_Truth"], cfg)
    pred_major = _parse_major_lineage_generic(ans, cfg)
    pred_state = _parse_state_generic(ans, cfg)

    # 1) Major lineage scoring (with hard penalties / near-lineage partial credit)
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

    # 2) State scoring (exact match on parsed state)
    state_score = 1.0 if pred_state == exp_state else 0.0

    # 3) Weighted combination (clipped to [0, 1])
    total = cfg.w_lineage * lineage_score + cfg.w_state * state_score
    total = float(max(0.0, min(1.0, total)))
    return total
