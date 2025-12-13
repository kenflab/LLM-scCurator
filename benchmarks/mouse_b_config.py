# LLM-scCurator/benchmarks/mouse_b_config.py

from .hierarchical_scoring import HierarchyConfig, score_hierarchical

"""
HierarchyConfig for the Tabula Muris Senis mouse B-lineage benchmark.

Ground_Truth labels after get_bcell_ground_truth():
    - "Erythrocyte_like"
    - "Mast_like"
    - "pDC_Myeloid_like"
    - "Mature_B"
    - "B_Other" (ambiguous late pro-B etc.; falls back to default_state)
"""

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
        # "B_Other" は default_state="Other" に落ちる
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

    # Lineage と state を同等に評価 (0.5 / 0.5)
    w_lineage=0.5,
    w_state=0.5,

    default_major="BLineage",
    default_state="Other",
)


def score_mouse_b(row, col_name):
    """
    Thin wrapper so that scoring API matches other benchmarks.
    """
    return score_hierarchical(row, col_name, cfg=MOUSE_B_CFG)
