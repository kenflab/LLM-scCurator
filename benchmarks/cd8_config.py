
# cd8_config.py

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
        # Naive は「naive」だけを見る（central memory は別扱い）
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
    #    T 期待のところを NK と呼んだら 0 点（その逆も同様）
    major_penalties={
        "T": {"B", "Myeloid", "NK"},
        "NK": {"B", "Myeloid", "T"},
    },

    # 5) near_lineage_pairs は空にする（T↔NK の partial credit は廃止）
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
    Thin wrapper around the generic hierarchical scorer using the CD8 config.
    """
    return score_hierarchical(row, col_name, cfg=CD8_HIER_CFG)
