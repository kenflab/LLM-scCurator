
# LLM-scCurator/benchmarks/gt_mappings.py
# Source of truth: cluster_name → Ground_Truth category

def get_cd8_ground_truth(cluster_name: str) -> str:
    """
    Map Zheng et al. CD8 meta.cluster names to GT categories.

    The GT labels are intentionally slightly finer (Naive / Effector / EffectorMemory /
    Exhausted / ISG / MAIT / NK_killer / Cycling), and are later collapsed into
    coarse (major, state) pairs by CD8_HIER_CFG.gt_rules.
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
    Map Zheng et al. CD4 meta.cluster names to GT categories.

    We intentionally keep this mapping simple and deterministic, based only on
    the author-provided cluster names (Tn/Tm/Tem/Temra/Th17/Tfh/Treg/ISG).
    Finer biological nuances are handled at the scoring layer, not here.
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
    Ground-truth mapping for the CAF / MSC benchmark.

    We deliberately collapse the space into a small set of biologically
    interpretable states:
        - Fibro_iCAF      (MSC / inflammatory CAFs)
        - Fibro_myCAF     (myofibroblastic / activated CAFs)
        - Fibro_PVL       (perivascular / pericyte / smooth muscle lineage)
        - Fibro_Cycling   (cycling fibroblasts that are not clearly PVL)
        - Endothelial     (vascular endothelial lineage)
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
    Tabula Muris Senis B-cell atlas (contamination detection).
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
