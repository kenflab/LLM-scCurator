# scorer_caf.py
from typing import Any
import pandas as pd
from .hierarchical_scoring import score_hierarchical
from .caf_config import CAF_HIER_CFG


_SUBTYPE_KEYWORDS = [
    # iCAF / MSC-like
    "icaf", "inflammatory", "chemokine", "il6", "il-6",
    "msc", "msc-like", "lipofibroblast", "lipofibroblasts",
    "alveolar fibroblast", "alveolar fibroblasts",
    # myCAF / activated CAFs
    "mycaf", "myofibroblast", "contractile", "acta2", "acta-2",
    "alpha-sma", "α-sma", "αsma",
    "activated fibroblast", "activated fibroblasts",
    "activated caf", "activated stromal cell", "activated stromal cells",
    # PVL / perivascular
    "pvl", "perivascular", "pericyte", "pericytes",
    "rgs5", "mural", "mural cell", "mural cells",
    "smooth muscle", "smooth muscle cell", "smooth muscle cells",
    "vascular smooth muscle", "vascular smooth muscle cell",
    "vascular smooth muscle cells", "vsmc",
    "vascular pericyte", "vascular pericytes",
    # Cycling
    "cycling", "proliferating", "proliferation",
    "dividing", "cell cycle", "mki67", "top2a",
]


def score_caf_hierarchical(row: pd.Series, col_name: str) -> float:
    """
    CAF / PVL benchmark-specific wrapper.

    We no longer assign an automatic 0.0 to generic "fibroblast" answers.
    Instead, we rely on the hierarchical scorer:

      - If the method only says "fibroblast", it gets partial credit
        from the lineage term (w_lineage = 0.3) but 0 for the state.
      - If it specifies the subtype (iCAF / myCAF / PVL / Endothelial),
        and matches the Ground Truth, it reaches a full score of 1.0.
    """
    # We keep `ans` here in case we later want to log or debug,
    # but the scoring itself is delegated to the generic scorer.
    ans = str(row[col_name]).lower()
    return score_hierarchical(row, col_name, cfg=CAF_HIER_CFG)
