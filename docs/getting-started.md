# Getting started

## Install
```bash
pip install llm-sc-curator
```

From source (development):
```bash
git clone https://github.com/kenflab/LLM-scCurator.git
cd LLM-scCurator
pip install -e .
```

Backend (API key)
```bash
export GEMINI_API_KEY="PASTE_YOUR_KEY"
# export OPENAI_API_KEY="PASTE_YOUR_KEY"  # optional
```

Shortest run (Scanpy)
```pyhton
import scanpy as sc
from llm_sc_curator import LLMscCurator

adata = sc.read_h5ad("my_data.h5ad")
curator = LLMscCurator(model_name="models/gemini-2.5-pro")  # optional
adata = curator.run_hierarchical_discovery(adata)
```

What you get
  - adata.obs["major_type"]: coarse lineage (e.g., T / B / myeloid / stromal)
  - adata.obs["fine_type"]: subtype/state label
  - curated marker audit trail per cluster (kept / masked / rescued)
