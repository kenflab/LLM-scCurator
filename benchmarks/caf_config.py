
# caf_config.py
from .hierarchical_scoring import HierarchyConfig
from typing import Set, FrozenSet

CAF_HIER_CFG = HierarchyConfig(
    # 1) Prediction-side lineage parsing
    lineage_aliases={
        "Fibroblast": [
            # fibroblast / stromal lineage
            "fibroblast",
            "fibroblasts",
            "fibrocyte",
            "fibrocytes",
            "caf",
            "cancer-associated fibroblast",
            "cancer associated fibroblast",
            "stromal",
            "stromal cell",
            "stromal cells",
            "stroma",
            "mesenchymal",
            "mesenchymal stem cell",
            "mesenchymal stem cells",
            "mesenchymal stromal cell",
            "mesenchymal stromal cells",
            # perivascular cells are mesenchymal lineage
            "pvl",
            "perivascular",
            "pericyte",
            "pericytes",
            "mural",
        ],
        "Immune": [
            "immune",
            "t cell",
            "t-cell",
            "b cell",
            "b-cell",
            "macrophage",
            "myeloid",
            "lymphocyte",
        ],
        "Tumor": [
            "tumor",
            "cancer cell",
            "epithelial",
            "carcinoma",
        ],
        # explicit endothelial lineage
        "Endothelial": [
            "endothelial",
            "endothelium",
            "blood vessel",
            "blood vessels",
            "vascular",
            "capillary",
            "arterial",
            "venous",
            "vessel",
            "vessels",
        ],
        "Other": [],
    },

    # 2) Prediction-side state parsing
    #    This is what goes into the confusion matrices for the MSC benchmark.
    state_aliases={
        # inflammatory / MSC-like CAFs
        "iCAF": [
            "icaf",
            "inflammatory",
            "chemokine",
            "il6",
            "il-6",
            "msc",
            "msc-like",
            "mesenchymal stem cell",
            "mesenchymal stem cells",
            "mesenchymal stromal cell",
            "mesenchymal stromal cells",
            "lipofibroblast",
            "lipofibroblasts",
            "alveolar fibroblast",
            "alveolar fibroblasts",
        ],
        # myofibroblastic / activated CAFs
        "myCAF": [
            "mycaf",
            "myofibroblast",
            "myofibroblasts",
            "contractile",
            "acta2",
            "acta-2",
            "alpha-sma",
            "α-sma",
            "αsma",
            "activated fibroblast",
            "activated fibroblasts",
            "activated caf",
            "activated stromal cell",
            "activated stromal cells",
        ],
        # perivascular lineage (PVL / pericytes / smooth muscle continuum)
        "PVL": [
            "pvl",
            "perivascular",
            "pericyte",
            "pericytes",
            "rgs5",
            "mural",
            "mural cell",
            "mural cells",
            "smooth muscle",
            "smooth muscle cell",
            "smooth muscle cells",
            "vascular smooth muscle",
            "vascular smooth muscle cell",
            "vascular smooth muscle cells",
            "vsmc",
            "vascular pericyte",
            "vascular pericytes",
        ],
        "Cycling": [
            "cycling",
            "proliferating",
            "proliferation",
            "dividing",
            "cell cycle",
            "mki67",
            "top2a",
        ],
        # Endothelial state for MSC benchmark
        "Endothelial": [
            "endothelial",
            "endothelium",
            "blood vessel",
            "blood vessels",
            "vascular",
            "vessel",
            "vessels",
            "capillary",
            "arterial",
            "venous",
        ],
        "Other": [],
    },

    # 3) GT Rules: Ground_Truth → (major, state)
    #    NOTE: "Endothelial" GT now maps to a real state, so it is not filtered out.
    gt_rules=[
        ("endothelial", ("Endothelial", "Endothelial")),
        ("cycling", ("Fibroblast", "Cycling")),
        ("icaf", ("Fibroblast", "iCAF")),
        ("mycaf", ("Fibroblast", "myCAF")),
        ("pvl", ("Fibroblast", "PVL")),
    ],

    # 4) Logic: penalties & weights
    major_penalties={
        "Fibroblast": {"Tumor", "Immune"},
        "Endothelial": {"Tumor", "Immune"},
    },

    # 5) Near-lineage pairs
    #    Fibroblast ↔ Endothelial は「血管ニッチ」での誤差として 0.5 の lineage 部分点を与える
    near_lineage_pairs={
        frozenset({"Fibroblast", "Endothelial"}),
    },

    failure_keywords=[
        "error",
        "unknown",
        "indeterminate",
        "undetermined",
        "unclassified",
        "ambiguous",
    ],

    # For CAF / MSC we care primarily about state (iCAF/myCAF/PVL/Cycling/Endothelial),
    # but we still award partial credit for getting the mesenchymal vs immune/tumor
    # lineage direction correct.
    w_lineage=0.3,
    w_state=0.7,

    default_major="Fibroblast",
    default_state="Other",
)
