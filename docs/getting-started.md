# Getting started

This page provides a minimal, end-to-end run of **LLM-scCurator** on an AnnData object (Scanpy).
For the conceptual model (masking, rescue, leakage-safe lineage filtering, hierarchical inference), see **[Concepts](concepts.md)**.

---

## Install
- #### Option A (recommended): Install from PyPI
  ```bash
  pip install "llm-sc-curator[gemini]"
  # or: pip install "llm-sc-curator[openai]"
  # or: pip install "llm-sc-curator[all]"
  ```
  
  > Notes: Install from PyPI for stable releases; install from GitHub if you need the latest development version.


- #### Option B: Install from GitHub (development)
  ```bash
  # 1. Clone the repository
  git clone https://github.com/kenflab/LLM-scCurator.git
  
  # 2. Navigate to the directory
  cd LLM-scCurator
  
  # 3. Install the package (and dependencies)
  pip install .
  ```
  > Notes: If you already have a Scanpy/Seurat pipeline environment, you can install it into that environment.

## Configure an LLM backend (optional)
LLM-scCurator can run marker distillation locally.  
LLM-scCurator supports both cloud LLM APIs (Gemini / OpenAI) and a local LLM backend (Ollama).
Notes:
> No manual installation is required: the official Docker environment already includes LLM-scCurator and its dependencies.  
> If you use the local Ollama backend, no API key is needed.

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
```python
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

## Annotate existing clusters (v0.1.1+)

If you already have clusters (e.g., Seurat `seurat_clusters`, Leiden, etc.) and want to
annotate each cluster once, then export a shareable table (CSV) and propagate labels to cells,
use the utilities added in **v0.1.1+**.

> Tip: This workflow is ideal when clustering is done upstream (Seurat/Scanpy), and you want a clean
> cluster → label map that can be reviewed by collaborators.
> If your cluster column has a different name, set `cluster_col` accordingly (e.g., `"leiden"`).

```python
import scanpy as sc
from llm_sc_curator import LLMscCurator
from llm_sc_curator import (
    export_cluster_annotation_table,
    apply_cluster_map_to_cells,
)

adata = sc.read_h5ad("my_data.h5ad")
cluster_col = "seurat_clusters"  # change if needed

curator = LLMscCurator(
    api_key=GEMINI_API_KEY,              # required for LLM-based annotation
    model_name="models/gemini-2.5-pro",  # optional
)

clusters = sorted(adata.obs[cluster_col].astype(str).unique())
cluster_results = {}
genes_by_cluster = {}

for cl in clusters:
    genes = curator.curate_features(
        adata,
        group_col=cluster_col,
        target_group=str(cl),
        use_statistics=True,
    )
    genes_by_cluster[str(cl)] = genes or []

    if genes:
        cluster_results[str(cl)] = curator.annotate(genes, use_auto_context=True)
    else:
        cluster_results[str(cl)] = {
            "cell_type": "NoGenes",
            "confidence": "Low",
            "reasoning": "Curated gene list empty",
        }

# 1) Export a cluster-level table (shareable)
df_cluster = export_cluster_annotation_table(
    adata,
    cluster_col=cluster_col,
    cluster_results=cluster_results,
    genes_by_cluster=genes_by_cluster,
    prefix="Curated",
)
df_cluster.to_csv("cluster_curated_map.csv", index=False)

# 2) Propagate cluster labels to per-cell labels
apply_cluster_map_to_cells(
    adata,
    cluster_col=cluster_col,
    df_cluster=df_cluster,
    label_col="Curated_CellType",
    new_col="Curated_CellType",
)

df_cluster.head()

```
**Outputs**
- `cluster_curated_map.csv`: cluster → label/confidence/reasoning/genes (shareable)
- `adata.obs["Curated_CellType"]`: per-cell labels propagated from the cluster map


## Docker (official environment)
If you prefer a fully contained environment (Python + R + Jupyter), we provide an official Docker image.  
Optionally includes Ollama for local LLM annotation (no cloud API key required).

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
 
  > Notes: For manuscript reproducibility, we also provide versioned tags (e.g., :v0.1.0). Prefer a version tag when matching a paper release.

- #### Option B: Build locally (development)

  - ##### Option B-1: Build locally with Compose (recommended for dev)
    ```bash
    # from the repo root
    docker compose -f docker/docker-compose.yml build
    docker compose -f docker/docker-compose.yml up
    ```

    **B-1.1) Open Jupyter**
    - [http://localhost:8888](http://localhost:8888) 
      Workspace mount: `/work`

    **B-1.2) If prompted for "Password or token"**
    - Get the tokenized URL from container logs:
      ```bash
      docker compose -f docker/docker-compose.yml logs -f llm-sc-curator
      ```
    - Then either:
      - open the printed URL (contains `?token=...`) in your browser, or
      - paste the token value into the login prompt.

  - ##### Option B-2: Build locally without Compose (alternative)
    ```bash
    # from the repo root
    docker build -f docker/Dockerfile -t llm-sc-curator:official .
    ```

    **B-2.1) Run Jupyter**
    ```bash
    docker run --rm -it \
      -p 8888:8888 \
      -v "$PWD":/work \
      -e GEMINI_API_KEY \
      -e OPENAI_API_KEY \
      llm-sc-curator:official
    ```

    **B-2.2) Open Jupyter**
    - [http://localhost:8888](http://localhost:8888)
      Workspace mount: `/work`  
 <br>  

---
## Apptainer / Singularity (HPC)
- #### Option A: Prebuilt image (recommended)
  Use the published image from GitHub Container Registry (GHCR).
  ```bash
  apptainer build llm-sc-curator.sif docker://ghcr.io/kenflab/llm-sc-curator:official
  ```

- #### Option B:  a .sif from the Docker image (development)
  ```bash
  docker compose -f docker/docker-compose.yml build
  apptainer build llm-sc-curator.sif docker-daemon://llm-sc-curator:official
  ```

Run Jupyter (either image):
  ```bash
  apptainer exec --cleanenv \
    --bind "$PWD":/work \
    llm-sc-curator.sif \
    bash -lc '
  ```

## Next steps
- Design and terminology: **[Concepts](concepts.md)**
- Practical workflows (including R/Seurat export → Python run → re-import): **[User guide](user-guide.md)**
- Full API surface: **[API reference](api-reference.md)**
