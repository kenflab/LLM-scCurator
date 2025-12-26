# Getting started

This page provides a minimal, end-to-end run of **LLM-scCurator** on an AnnData object (Scanpy).
For the conceptual model (masking, rescue, leakage-safe lineage filtering, hierarchical inference), see **[Concepts](concepts.md)**.

---

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

> Note: If you already have a Scanpy/Seurat pipeline environment, installing into that environment is typically sufficient.

## Configure an LLM backend (optional)

LLM-scCurator can run marker distillation locally.  
If you want **automatic LLM-based annotation**, provide an API key for your backend.

### Environment variables

```python
GEMINI_API_KEY = "PASTE_YOUR_KEY_HERE"
# OPENAI_API_KEY = "PASTE_YOUR_KEY_HERE"  # optional
```

### Notebook-friendly (simplest)
```bash
export GEMINI_API_KEY="PASTE_YOUR_KEY"
# export OPENAI_API_KEY="PASTE_YOUR_KEY"  # optional
```

## Shortest run (Python / Scanpy)
```pyhton
import scanpy as sc
from llm_sc_curator import LLMscCurator

adata = sc.read_h5ad("my_data.h5ad")

# Initialize (backend-agnostic; Gemini shown here)
curator = LLMscCurator(
    api_key=GEMINI_API_KEY,              # optional if you only distill markers
    model_name="models/gemini-2.5-pro",  # optional
)

# Fully automated hierarchical annotation (coarse-to-fine)
adata = curator.run_hierarchical_discovery(adata)

# Visualize
sc.pl.umap(adata, color=["major_type", "fine_type"])

```

## What you get

After a successful run, you typically obtain:

- `adata.obs["major_type"]`: coarse lineage (e.g., T / B / myeloid / stromal)
- `adata.obs["fine_type"]`: fine-grained subtype / state labels
- A per-cluster marker audit trail (kept / masked / rescued), enabling transparent review of the pre-prompt distillation step


  
## Docker (official environment)
If you prefer a fully contained environment (Python + R + Jupyter), we provide an official Docker image.

- #### Option A: Prebuilt image (recommended)
  Use the published image from GitHub Container Registry (GHCR).

  ```bash
  # from the repo root (optional, for notebooks / file access)
  docker pull ghcr.io/kenflab/llm-sc-curator:official

  docker run --rm -it \
    -p 8888:8888 \
    -v "$PWD":/work \
    -e GEMINI_API_KEY \
    -e OPENAI_API_KEY \
    ghcr.io/kenflab/llm-sc-curator:official
  ```
  Open Jupyter:
  [http://localhost:8888](http://localhost:8888) 
  (Use the token printed in the container logs.) 
  Notes:
  > For manuscript reproducibility, we also provide versioned tags (e.g., :v0.1.0). Prefer a version tag when matching a paper release.

- #### Option B: Build locally (development)
  ```bash
  # from the repo root
  docker compose -f docker/docker-compose.yml build
  docker compose -f docker/docker-compose.yml up
  ```
  Open Jupyter:
  [http://localhost:8888](http://localhost:8888) 
  Workspace mount: /work

## Next steps
- Design and terminology: **[Concepts](concepts.md)**
- Practical workflows (including R/Seurat export → Python run → re-import): **[User guide](user-guide.md)**
- Full API surface: **[API reference](api-reference.md)**
