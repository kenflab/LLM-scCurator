"""
CD4 benchmark hierarchy configuration.

This module defines the task-specific ontology configuration used to score CD4⁺ T cell
annotations in an ontology-aware, hierarchy-aware manner. The configuration specifies:

- prediction-side aliases for major lineage parsing (T, NK, B, Myeloid, Other),
- prediction-side aliases for within-lineage state parsing (Treg, Tfh, Th17, ISG, Cycling,
  Naive, Exhausted, EffMem),
- ground-truth mapping rules (Ground_Truth → expected major/state), and
- strict scoring constraints (hard cross-lineage penalties, failure keywords, weights).

Notes
-----
For CD4 benchmarking we intentionally avoid granting implicit T-lineage credit:
predictions must contain explicit T-cell terminology to be parsed as "T". If no alias
matches, the prediction is assigned to `default_major="Other"`, which yields no lineage
credit under the strict penalty regime.
"""

from .hierarchical_scoring import HierarchyConfig, score_hierarchical
import pandas as pd

# -----------------------------------------------------------------------------
# CD4 hierarchical scoring configuration
# -----------------------------------------------------------------------------

CD4_HIER_CFG = HierarchyConfig(
    # 1) Prediction-side lineage aliases
    lineage_aliases={
        "B": [
            "b cell",
            "b-cell",
            "b lymphocyte",
            "regulatory b cell",
            "breg",
        ],
        "NK": [
            "nk cell",
            "natural killer",
            "cd56",
            "innate lymphoid",
        ],
        "Myeloid": [
            "macrophage",
            "monocyte",
            "myeloid",
            "dendritic",
            "dc",
            "pdc",
            "neutrophil",
        ],
        "T": [
            "t cell",
            "t-cell",
            "cd4",
            "cd8",
            "helper t",
            "th1",
            "th2",
            "th17",
            "t helper",
            "t follicular",
            "tfh",
            "treg",
            "regulatory t",
            "mait",
        ],
        "Other": [],
    },

    # 2) Prediction-side state aliases
    #    Order: more specific states first.
    state_aliases={
        "Treg": [
            "treg",
            "regulatory t",
            "regulatory t cell",
            "regulatory t cells",
            "t regulatory",
            "t regulatory cell",
            "t regulatory cells",
            "foxp3",
        ],
        "Tfh": [
            "t follicular",
            "tfh",
        ],
        "Th17": [
            "th17",
            "il17",
            "il-17",
            "il23r",
            "rorc",
        ],
        "ISG": [
            "isg",
            "interferon",
            "type i ifn",
            "antiviral",
        ],
        "Cycling": [
            "proliferating",
            "proliferation",
            "cycling",
            "dividing",
            "mki67",
            "ki-67",
            "cell cycle",
        ],
        "Naive": [
            "naive",
        ],
        "Exhausted": [
            "exhausted",
            "tex",
            "dysfunctional",
            "pd-1",
            "pdcd1",
            "progenitor exhausted",
            "tpex",
        ],
        "EffMem": [
            "memory",
            "central memory",
            "tcm",
            "effector",
            "tem",
            "temra",
            "cytotoxic",
            "activated",
        ],
        "Other": [],
    },

    # 3) GT-side rules (Ground_Truth → expected (major, state))
    gt_rules=[
        # Treg / Tfh / Th17
        ("treg", ("T", "Treg")),
        ("tfh", ("T", "Tfh")),
        ("th17", ("T", "Th17")),

        # ISG / Cycling
        ("isg", ("T", "ISG")),
        ("cycling", ("T", "Cycling")),

        # Naive
        ("tn.naive", ("T", "Naive")),

        # Effector-memory pools
        ("temra.effmem", ("T", "EffMem")),
        ("tem.effmem", ("T", "EffMem")),
        ("tm.effmem", ("T", "EffMem")),

        # Exhausted pool (if present)
        ("exhausted", ("T", "Exhausted")),
    ],

    # 4) Hard cross-lineage penalties
    major_penalties={
        "T": {"B", "Myeloid", "NK"},
        "NK": {"B", "Myeloid", "T"},
        "B": {"T", "NK", "Myeloid"},
        "Myeloid": {"T", "B", "NK"},
    },

    # 5) Near lineage pairs (none for CD4; CD4/CD8 is treated at state level)
    near_lineage_pairs=set(),

    # 6) Failure keywords: any of these substrings → score = 0
    failure_keywords=[
        "error",
        "unknown",
        "unidentifiable",
        "cannot identify",
        "ribosomal-high",
        "ribosomal high",
        "low-quality",
        "low quality",
        "unidentified",
    ],

    # 7) Weights
    w_lineage=0.7,
    w_state=0.3,

    # IMPORTANT:
    #   For CD4 we do NOT give T-lineage credit by default.
    #   Predictions must explicitly contain T-cell terminology to be scored as "T".
    default_major="Other",
    default_state="Other",
)


def score_cd4(row: pd.Series, col_name: str) -> float:
    """
    Score one prediction column using the CD4 hierarchy configuration.

    This is a thin wrapper around `score_hierarchical(...)` that fixes `cfg=CD4_HIER_CFG`
    for the CD4 benchmark.

    Parameters
    ----------
    row : pandas.Series
        A row from the evaluation table. Must include:
        - `row[col_name]`: prediction text (free-form)
        - `row["Ground_Truth"]`: harmonized ground-truth label
    col_name : str
        Name of the column containing the prediction text to score.

    Returns
    -------
    float
        Ontology-aware hierarchical score in the range [0.0, 1.0].

    Notes
    -----
    - Any `failure_keywords` present in the prediction text force a score of 0.0.
    - Near-lineage partial credit is disabled (`near_lineage_pairs = set()`).
    """
    return score_hierarchical(row, col_name, cfg=CD4_HIER_CFG)
