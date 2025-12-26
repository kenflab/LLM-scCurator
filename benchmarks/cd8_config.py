"""
CD8 benchmark hierarchy configuration.

This module defines the task-specific ontology configuration used to score CD8⁺ T cell
annotations in an ontology-aware manner. The configuration specifies:

- prediction-side aliases for major lineage parsing (T, NK, B, Myeloid, Other),
- prediction-side aliases for within-lineage state parsing (Naive, EffMem, Exhausted, ISG, MAIT, Cycling),
- ground-truth mapping rules (Ground_Truth → expected major/state), and
- scoring constraints (hard cross-lineage penalties, failure keywords, weights).

Notes
-----
This CD8 benchmark is intentionally strict about T vs NK confusion: near-lineage partial
credit is disabled (`near_lineage_pairs = set()`), and T↔NK calls receive a score of 0.0
via `major_penalties`. This enforces a conservative evaluation regime.
"""

from .hierarchical_scoring import HierarchyConfig, score_hierarchical
import pandas as pd

CD8_HIER_CFG = HierarchyConfig(
    # 1) Prediction-side lineage aliases
    lineage_aliases={
        "B": [
            "b cell",
            "b-cell",
            "regulatory b cell",
            "breg",
        ],
        "NK": [
            "nk cell",
            "natural killer",
            "cd56",
            "cd56dim",
            "cd56bright",
        ],
        "Myeloid": [
            "macrophage",
            "monocyte",
            "myeloid",
            "dendritic",
            "dc",
        ],
        "T": [
            "t cell",
            "t-cell",
            "cd4",
            "cd8",
            "trm",
            "tex",
            "mait",
            "t follicular helper",
            "tfh",
            "t helper 17",
            "th17",
        ],
        "Other": [],
    },

    # 2) Prediction-side state aliases
    state_aliases={
        "MAIT": [
            "mait",
            "mucosa-associated invariant",
        ],
        "ISG": [
            "interferon",
            "isg",
            "antiviral",
            "type i ifn",
        ],
        "Cycling": [
            "proliferating",
            "cycling",
            "dividing",
            "mki67",
            "ki-67",
            "cell cycle",
        ],
        # Naive is matched strictly on "naive" (central memory is intentionally not treated as Naive).
        "Naive": [
            "naive",
        ],
        "Resident": [
            "resident",
            "trm",
            "tissue-resident",
            "tissue resident",
        ],
        "Exhausted": [
            "exhausted",
            "tex",
            "dysfunctional",
            "pd-1",
            "pdcd1",
        ],
        "EffMem": [
            "memory",
            "effector",
            "tem",
            "temra",
            "cytotoxic",
            "ctl",
            "activated",
        ],
        "Other": [],
    },

    # 3) GT-side rules (Ground_Truth → expected (major, state))
    gt_rules=[
        # NK / killer pool
        ("nk", ("NK", "EffMem")),
        ("killer", ("NK", "EffMem")),

        # MAIT
        ("mait", ("T", "MAIT")),

        # ISG
        ("isg", ("T", "ISG")),

        # Cycling
        ("cycling", ("T", "Cycling")),

        # Naive / Exhausted / EffMem
        ("naive", ("T", "Naive")),
        ("exhausted", ("T", "Exhausted")),
        ("effector", ("T", "EffMem")),
        ("memory", ("T", "EffMem")),
    ],

    # 4) Hard cross-lineage penalties
    # Hard penalties: T↔NK confusions receive 0.0 (and likewise for other forbidden lineages).
    major_penalties={
        "T": {"B", "Myeloid", "NK"},
        "NK": {"B", "Myeloid", "T"},
    },

    # 5) No near-lineage partial credit for CD8 (T↔NK partial credit disabled).
    near_lineage_pairs=set(),

    # 6) Failure keywords
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

    default_major="Other",
    default_state="Other",
)


def score_answer(row: pd.Series, col_name: str) -> float:
    """
    Score one prediction column using the CD8 hierarchy configuration.

    This is a thin wrapper around `score_hierarchical(...)` that fixes `cfg=CD8_HIER_CFG`
    for the CD8 benchmark.

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
    - Prediction text is normalized inside the scorer (lowercased, stripped).
    - Any `failure_keywords` present in the prediction text force a score of 0.0.
    """
    return score_hierarchical(row, col_name, cfg=CD8_HIER_CFG)
