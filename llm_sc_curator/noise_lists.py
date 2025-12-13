
# Regex patterns for biological noise modules (Human & Mouse compatible)
NOISE_PATTERNS = {
    # --- Technical / Mapping Artifacts ---
    'Ensembl_ID': r'^(ENSG|ENSMUSG)\d+',                   # Meaningless IDs for LLMs
    'LINC_Noise': r'^(LINC|linc)\d+$',                     # Human LINC#### / linc####
    'Mouse_Predicted_Gm': r'^Gm\d+$',                      # Mouse predicted genes
    'Mouse_Rik': r'^[0-9A-Za-z]+Rik$',                     # Mouse Rik
    'LOC_Locus': r'^LOC\d+$',                              # Unannotated locus identifiers

    # --- Lineage Noise (Variable Regions) ---
    'TCR_Clone': r'^TR[ABGD][VDJ]',                        # T-cell Receptors (Human)
    'TCR_Clone_Mouse': r'^Tr[abgd][vdj]',                  # T-cell Receptors (Mouse)
    'Ig_Clone': r'^IG[HKL][VDJ]',                          # Immunoglobulins (Human)
    'Ig_Clone_Mouse': r'^Ig[hkl][vdj]',                    # Immunoglobulins (Mouse)

    # --- Ig Constant Regions (Human) ---
    'Ig_Constant_Heavy': r'^IGH[-_]?((M|D)|G[1-4]|A[1-2]|E)$',  # IGHM, IGHD, IGHG1–4, IGHA1–2, IGHE
    'Ig_Constant_Light_Kappa': r'^IGKC$',                       # IGKC
    'Ig_Constant_Light_Lambda':   r'^IGLC.*$',                  # IGLC1–7

    # --- Ig Constant Regions (Mouse) ---
    'Ig_Constant_Heavy_Mouse': r'^Igh[-_]?((m|d)|g[1-4]|a|e)$', # Ighm, Ighd, Ighg1–4, Igha, I ghe
    'Ig_Constant_Light_Kappa_Mouse': r'^Igkc$',                 # Igkc
    'Ig_Constant_Light_Lambda_Mouse': r'^Iglc.*$',              # Iglc1–7

    # --- Biological State Noise ---
    'Mito_Artifact': r'^[Mm][Tt]-',                        # Mitochondrial (MT- or mt-)
    'Ribo_Artifact': r'^[Rr][Pp][LSls]',                   # Ribosomal (RPS/RPL or Rps/Rpl)
    'HeatShock': r'^[Hh][Ss][Pp]',                         # Heat shock (HSP or Hsp)
    'JunFos_Stress': r'^(JUN|FOS|Jun|Fos)',                # Dissociation stress
    'Hemo_Contam': r'^[Hh][Bb][ABab]',                     # Hemoglobin
    'Translation_Factor': r'^(EEF|EIF|TPT1|Eef|Eif|Tpt1)', # Translation

    # --- Chromatin & Proliferation Artifacts ---
    'Histone': r'^(HIST|Hist)',

    # --- Donor/Batch Confounders ---
    # 1. HLA Class I (Human): Ubiquitous & Interferon-sensitive.
    'HLA_ClassI_Noise': r'^HLA-[ABCEFG]',                  # Keeps Class II (HLA-D) for APC definition.

    # 2. MHC Class I (Mouse): H-2K, H-2D, H-2L.
    'MHC_ClassI_Noise': r'^H2-[DKL]',                      # Keeps Class II (H-2A, H-2E) for APC definition.

    # 3. Sex Chromosome (Gender Batch Effect Removal)
    'SexChromosome': r'^(XIST|UTY|DDX3Y|Xist|Uty|Ddx3y)',  # Removes XIST (Female) and Y-linked genes (Male)
}

# Full Cell Cycle Genes from Tirosh et al. (Science 2016) Table S5
# Combined G1/S, G2/M, and Melanoma Core Cycling Genes
# Defined in Human format (All Caps)
_HUMAN_CC_GENES = {
    # G1/S
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7",
    "DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN",
    "WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45",
    "CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B",
    "BRIP1","E2F8",

    # G2/M
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2",
    "CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2",
    "AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","HJURP","CDCA3",
    "HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8",
    "ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2",
    "G2E3","GAS2L3","CBX5","CENPA",

    # Melanoma Core Cycling (Additional)
    "TYMS","TK1","UBE2T","CKS1B","MCM5","UBE2C","PCNA","MAD2L1","ZWINT","MCM4","GMNN",
    "MCM7","NUSAP1","FEN1","CDK1","BIRC5","KIAA0101","PTTG1","CENPM","KPNA2","CDC20",
    "GINS2","ASF1B","RRM2","MLF1IP","KIF22","CDC45","CDC6","FANCI","HMGB2","TUBA1B",
    "RRM1","CDKN3","WDR34","DTL","CCNB1","AURKB","MCM2","CKS2","PBK","TPX2","RPL39L",
    "SNRNP25","TUBG1","RNASEH2A","TOP2A","DTYMK","RFC3","CENPF","NUF2","BUB1","H2AFZ",
    "NUDT1","SMC4","ANLN","RFC4","RACGAP1","KIFC1","TUBB6","ORC6","CENPW","CCNA2","EZH2",
    "NASP","DEK","TMPO","DSN1","DHFR","KIF2C","TCF19","HAT1","VRK1","SDF2L1","PHF19",
    "SHCBP1","SAE1","CDCA5","OIP5","RANBP1","LMNB1","TROAP","RFC5","DNMT1","MSH2","MND1",
    "TIMELESS","HMGB1","ZWILCH","ASPM","ANP32E","POLA2","FABP5","TMEM194A",
}

# Sentinel proliferation markers we want to keep (not mask by default)
PROLIFERATION_SENTINELS = {
    "MKI67", "CDK1", "CCNB1", "CCNB2", "PCNA", "TOP2A", "BIRC5",
    "Mki67", "Cdk1", "Ccnb1", "Ccnb2", "Pcna", "Top2a", "Birc5",
}

# Automatically generate Mouse format (Title Case: Mki67, Pcna)
# This makes the tool universal without manual listing.
_CELL_CYCLE_ALL = _HUMAN_CC_GENES.union({g.capitalize() for g in _HUMAN_CC_GENES})
CELL_CYCLE_GENES = _CELL_CYCLE_ALL.difference(PROLIFERATION_SENTINELS)

NOISE_LISTS = {
    'CellCycle_State': CELL_CYCLE_GENES
}
