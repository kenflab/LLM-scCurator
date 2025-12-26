"""
Deterministic ground-truth label mappings for benchmarking.

This module defines dataset-specific functions that map author-provided cluster names
(strings) to a harmonized `Ground_Truth` category used in evaluation.

Design principles
-----------------
- Deterministic: mapping depends only on the cluster name string (no expression data).
- Conservative: coarse categories are preferred over overfitting fine subtypes.
- Single source of truth: evaluation scripts should import these functions rather than
  re-implementing ad hoc label logic.

Notes
-----
These mappings are intended for benchmarking and figure generation. They do not claim
to be a biological re-annotation of the source atlases.
"""

# LLM-scCurator/benchmarks/gt_mappings.py
# Source of truth: cluster_name → Ground_Truth category

def get_cd8_ground_truth(cluster_name: str) -> str:
    """
    Map Zheng et al. CD8 meta.cluster names to harmonized ground-truth categories.

    The resulting labels are intentionally moderately granular (e.g., Naive, Effector,
    EffectorMemory, Exhausted, ISG, MAIT, NK-like, Cycling). During evaluation, these
    are further collapsed into (major, state) pairs by the CD8 hierarchy configuration
    (`CD8_HIER_CFG.gt_rules`), enabling ontology-aware scoring.

    Parameters
    ----------
    cluster_name : str
        Author-provided cluster identifier or meta.cluster name.

    Returns
    -------
    str
        Harmonized CD8 ground-truth label (e.g., "CD8_Naive", "CD8_Exhausted").
        If no rule matches, returns "CD8_Other".

    Notes
    -----
    - Matching is case-insensitive and based on substring heuristics.
    - Rule order matters (first match wins).
    """
    s = str(cluster_name).lower()

    # 1) Distinct functional states
    if "mait" in s:
        return "CD8_MAIT"

    if any(k in s for k in ["isg", "interferon", "ifit1"]):
        return "CD8_ISG"

    if any(k in s for k in ["proliferating", "cycle", "mki67", "top2a"]):
        return "CD8_Cycling"

    # 2) NK-like killer pool (exclude explicit T cell labels)
    if "nk" in s and "t cell" not in s:
        return "CD8_NK_Killer"

    # 3) Exhausted pool
    if any(k in s for k in ["tex", "exhausted", "pdcd1"]):
        return "CD8_Exhausted"

    # 4) TRM / resident memory → treated as EffectorMemory in GT
    if any(k in s for k in ["trm", "resident", "znf683", "itgae", "cd69"]):
        return "CD8_EffectorMemory"

    # 5) Naive pool (true naive; use 'tn.' to avoid Tn/Tm confusion)
    if "tn." in s or "naive" in s:
        return "CD8_Naive"

    # 6) Temra / CX3CR1-high killers
    if any(k in s for k in ["temra", "cx3cr1", "klrg1"]):
        return "CD8_Effector"

    # 7) Tem / Tm / GZMK+ effector-memory clusters
    if any(k in s for k in ["tem.", "tm.", "memory", "gzmk", "aqp3", "ltb"]):
        return "CD8_EffectorMemory"

    # 8) Tk / killer T clusters (Zheng's Tk)
    if "tk" in s or "killer" in s:
        return "CD8_Effector"

    # 9) Fallback
    return "CD8_Other"

def get_cd4_ground_truth(cluster_name: str) -> str:
    """
    Map Zheng et al. CD4 meta.cluster names to harmonized ground-truth categories.

    This mapping is intentionally simple and deterministic, relying only on the
    author-provided cluster names (e.g., Tn/Tm/Tem/Temra/Tfh/Th17/Treg/ISG/Cycling).
    Finer biological nuance is handled at the scoring layer rather than here.

    Parameters
    ----------
    cluster_name : str
        Author-provided cluster identifier or meta.cluster name.

    Returns
    -------
    str
        Harmonized CD4 ground-truth label (e.g., "CD4_Treg", "CD4_Tem.EffMem").
        If no rule matches, returns "CD4_Other".

    Notes
    -----
    - Matching is case-insensitive and based on substring heuristics.
    - Rule order matters (first match wins).
    """
    s = str(cluster_name).lower()

    # 1) Regulatory / helper subsets
    if any(k in s for k in ["treg", "foxp3", "regulatory"]):
        return "CD4_Treg"
    if any(k in s for k in ["tfh", "follicular", "cxcl13", "cxcr5"]):
        return "CD4_Tfh"
    if any(k in s for k in ["th17", "il17", "rorc", "il23r"]):
        return "CD4_Th17"

    # 2) ISG & cycling axes
    if any(k in s for k in ["isg", "interferon", "ifit1"]):
        return "CD4_ISG"
    if any(k in s for k in ["cycling", "proliferating", "mki67", "top2a"]):
        return "CD4_Cycling"

    if "mix" in s:
        return "CD4_Other"

    # 3) Exhausted pool (if ever present in CD4 cluster IDs)
    if any(k in s for k in ["tex", "exhausted", "pdcd1"]):
        return "CD4_Exhausted"

    # 4) Naive / memory pools
    if "tn." in s or "naive" in s:
        return "CD4_Tn.Naive"
    if "temra" in s:
        return "CD4_Temra.EffMem"
    if "tem." in s:
        return "CD4_Tem.EffMem"
    if "tm." in s:
        return "CD4_Tm.EffMem"

    # 5) Fallback
    return "CD4_Other"

def get_msc_ground_truth(cluster_name: str) -> str:
    """
    Map CAF/MSC benchmark cluster names to compact stromal ground-truth categories.

    We deliberately collapse stromal heterogeneity into a small set of interpretable
    states suitable for robust benchmarking:

    - "Fibro_iCAF"    : inflammatory CAF / MSC-like
    - "Fibro_myCAF"   : myofibroblastic / activated CAF-like
    - "Fibro_PVL"     : perivascular / pericyte / smooth muscle lineage
    - "Fibro_Cycling" : cycling fibroblasts not clearly PVL/endothelial
    - "Endothelial"   : vascular endothelial lineage

    Parameters
    ----------
    cluster_name : str
        Author-provided cluster label.

    Returns
    -------
    str
        Harmonized stromal ground-truth label.
        If no rule matches, returns "Fibro_Other".

    Notes
    -----
    - PVL detection takes precedence over cycling to avoid mislabeling "Cycling PVL".
    - Matching is case-insensitive and based on substring heuristics.
    """
    s = str(cluster_name).lower()

    # 1) Endothelial lineages
    if "endothelial" in s or "blood vessel" in s or "blood vessels" in s:
        return "Endothelial"

    # 2) PVL / perivascular / pericyte / smooth muscle clusters
    #    (including "Cycling PVL" – PVL is the primary signal here)
    if any(k in s for k in ["pvl", "perivascular", "pericyte", "smooth muscle"]):
        return "Fibro_PVL"

    # 3) iCAF-like MSC clusters
    if any(k in s for k in ["icaf", "msc"]):
        return "Fibro_iCAF"

    # 4) myCAF-like clusters
    if any(k in s for k in ["mycaf", "myofibroblast"]):
        return "Fibro_myCAF"

    # 5) Cycling fibroblasts (that are not clearly PVL / endothelial)
    if any(k in s for k in ["cycling", "proliferating", "mki67", "top2a"]):
        return "Fibro_Cycling"

    # 6) Fallback
    return "Fibro_Other"


def get_bcell_ground_truth(cluster_name: str) -> str:
    """
    Map Tabula Muris Senis B-cell atlas cluster names to contamination-focused labels.

    This benchmark is used to test robustness to misleading marker programs. The mapping
    intentionally assigns several non-B-like strings to decoy categories to reflect the
    benchmark design.

    Parameters
    ----------
    cluster_name : str
        Author-provided cluster label.

    Returns
    -------
    str
        Harmonized label used for the mouse B-cell benchmark.
        If no rule matches, returns "B_Other".

    Notes
    -----
    The returned categories (e.g., "Erythrocyte_like", "Mast_like", "pDC_Myeloid_like")
    are evaluation constructs for contamination/decoy detection rather than definitive
    cell-type claims.
    """
    s = str(cluster_name).lower().strip()
    if "immature b cell" in s:
        return "Erythrocyte_like"
    if "naive b cell" in s:
        return "Mast_like"
    if "precursor b cell" in s:
        return "pDC_Myeloid_like"
    if s.startswith("b cell"):
        return "Mature_B"
    if "late pro-b cell" in s:
        return "B_Other"
    return "B_Other"
