"""
Hierarchy configuration for the Tabula Muris Senis mouse B-lineage benchmark.

This benchmark is designed to quantify robustness against misleading marker programs
(decoys/contaminants) rather than to exhaustively subtype B-cell maturation.

Ground truth (after `get_bcell_ground_truth()`) uses the following labels:
- "Erythrocyte_like"
- "Mast_like"
- "pDC_Myeloid_like"
- "Mature_B"
- "B_Other" (ambiguous B-lineage; falls back to default_state)

Scoring intent
--------------
- Major lineage distinguishes true B-lineage from contaminants (Erythroid/Myeloid).
- State identifies which decoy program (or Mature_B) is present.
- Lineage and state are weighted equally (w_lineage=0.5, w_state=0.5).
- Misclassifying contaminants as B-lineage (or vice versa) receives a score of 0.0
  via hard cross-lineage penalties.

Notes
-----
This configuration is deterministic and intended for reviewer-facing reproducible evaluation.
"""

from .hierarchical_scoring import HierarchyConfig, score_hierarchical

MOUSE_B_CFG = HierarchyConfig(
    # 1) Lineage: separate true B lineage from contaminants
    lineage_aliases={
        "BLineage": [
            "b cell",
            "b-cell",
            "b lymphocyte",
            "b.fo",
        ],
        "Erythroid": [
            "erythrocyte",
            "erythroid",
            "erythroblast",
            "erythroblasts",
            "red blood",
            "rbc",
            "hemoglobin",
        ],
        "Myeloid": [
            "mast",
            "basophil",
            "myeloid",
            "dendritic",
            "pdc",
            "plasmacytoid",
            "macrophage",
        ],
        "Other": [],
    },

    # 2) State: which contaminant (or mature B) it is
    state_aliases={
        "Mature_B": [
            "mature",
            "follicular",
            "b cell",
            "b-cell",
            "b.fo",
            "ms4a1",
            "cd79a",
        ],
        "Erythrocyte_like": [
            "erythrocyte",
            "erythroid",
            "erythroblast",
            "erythroblasts",
            "rbc",
            "red blood",
            "hemoglobin",
            "hbb-bs",
            "hbb-bt",
            "hba-a1",
            "gypa",
        ],
        "Mast_like": [
            "mast",
            "mast cell",
            "mast cells",
            "basophil",
            "basophils",
            "cpa3",
            "gata2",
            "fcer1a",
            "mcpt8",
        ],
        "pDC_Myeloid_like": [
            "pdc",
            "plasmacytoid",
            "dendritic",
            "dendritic cell",
            "dendritic cells",
            "siglech",
            "bst2",
            "irf8",
            "myeloid",
        ],
        "Other": [],
    },

    # 3) Ground_Truth → (major, state)
    gt_rules=[
        ("erythrocyte_like", ("Erythroid", "Erythrocyte_like")),
        ("mast_like", ("Myeloid", "Mast_like")),
        ("pdc_myeloid_like", ("Myeloid", "pDC_Myeloid_like")),
        ("mature_b", ("BLineage", "Mature_B")),
        # "B_Other" intentionally falls back to default_state="Other".
    ],

    # 4) Penalties: miscalling contaminants as B cells (and vice versa) is 0
    major_penalties={
        "Erythroid": {"BLineage"},
        "Myeloid": {"BLineage"},
        "BLineage": {"Erythroid", "Myeloid"},
    },

    failure_keywords=[
        "error",
        "unknown",
        "cannot identify",
        "indeterminate",
        "undetermined",
        "unclassified",
        "ambiguous",
    ],

    # Lineage and state are weighted equally (0.5 / 0.5).
    w_lineage=0.5,
    w_state=0.5,

    default_major="BLineage",
    default_state="Other",
)


def score_mouse_b(row, col_name):
    """
    Score one prediction column using the Mouse B-lineage benchmark configuration.

    This is a thin wrapper around `score_hierarchical(...)` that fixes `cfg=MOUSE_B_CFG`
    to match the scoring API used by other benchmark tasks.

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
    - "B_Other" intentionally maps to (default_major="BLineage", default_state="Other").
    """
    return score_hierarchical(row, col_name, cfg=MOUSE_B_CFG)
