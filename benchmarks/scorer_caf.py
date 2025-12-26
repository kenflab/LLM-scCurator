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
    Legacy CAF/PVL scoring wrapper (currently unused).

    This module is retained for backward compatibility with earlier benchmark drafts.
    The current evaluation uses the generic ontology-aware scorer via `CAF_HIER_CFG`,
    and this wrapper simply delegates to `score_hierarchical(...)`.

    Notes
    -----
    - `_SUBTYPE_KEYWORDS` is not used by the current implementation.
    - This file is safe to remove once all downstream scripts are confirmed to import
      scoring wrappers exclusively from `caf_config.py`.
    """

    ans = str(row[col_name]).lower()
    return score_hierarchical(row, col_name, cfg=CAF_HIER_CFG)
