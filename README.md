# LLM-scCurator

<p align="left">
  <img src="https://raw.githubusercontent.com/kenflab/LLM-scCurator/main/docs/assets/LLM-scCurator_logo.png"
       width="90" alt="LLM-scCurator"
       style="vertical-align: middle; margin-right: 10px;">
  <span style="font-size: 28px; font-weight: 700; vertical-align: middle;">
    Dynamic feature masking to improve robustness of zero-shot cell-type annotation with LLMs.
  </span>
</p>


     
[![Docs](https://readthedocs.org/projects/llm-sccurator/badge/?version=latest)](https://llm-sccurator.readthedocs.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970494.svg)](https://doi.org/10.5281/zenodo.17970494)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

---

## 🚀 Overview
**LLM-scCurator** is a Large Language Model–based curator for single-cell and spatial transcriptomics. It performs noise-aware marker distillation—suppressing technical programs (e.g., ribosomal/mitochondrial), clonotype signals (TCR/Ig), and stress signatures while rescuing lineage markers—and applies leakage-safe lineage filters before prompting an LLM. It supports hierarchical (coarse-to-fine) annotation for single-cell and spatial transcriptomics data. See the documentation for full tutorials and API reference: [https://llm-sccurator.readthedocs.io](https://llm-sccurator.readthedocs.io)


### Key Features
- **🛡️ Noise-aware filtering:** Automatically removes lineage-specific noise (TCR/Ig) and state-dependent noise (ribosomal/mitochondrial).
- **🧠 Context-aware inference:** Automatically infers lineage context (e.g., "T cell") to guide LLM reasoning.
- **🔬 Hierarchical discovery:** One-line function to dissect complex tissues into major lineages and fine-grained subtypes.
- **🌍 Spatial ready:** Validated on scRNA-seq (10x) and spatial transcriptomics (Xenium, Visium).

---
## 📦 Installation

- #### Option A (recommended): Install from PyPI
  ```bash
  pip install llm-sc-curator

  ```

- #### Option B: Install from GitHub (development)
  ```bash
  # 1. Clone the repository
  git clone https://github.com/kenflab/LLM-scCurator.git
  
  # 2. Navigate to the directory
  cd LLM-scCurator
  
  # 3. Install the package (and dependencies)
  pip install .
  ```
Notes:
> If you already have a Scanpy/Seurat pipeline environment, you can install it into that environment.
---
## 🐳 Docker (official environment)
We provide an official Docker environment (Python + R + Jupyter), sufficient to run LLM-scCurator and most paper figure generation.

- #### Option A: Prebuilt image (recommended)
  Use the published image from GitHub Container Registry (GHCR).

  ```bash
  # from the repo root (optional, for notebooks / file access)
  docker pull ghcr.io/kenflab/llm-sc-curator:official
  ```
  Run Jupyter:
  ```
  docker run --rm -it \
    -p 8888:8888 \
    -v "$PWD":/work \
    -e GEMINI_API_KEY \
    -e OPENAI_API_KEY \
    ghcr.io/kenflab/llm-sc-curator:official
  ```
  Open Jupyter:
  [http://localhost:8888](http://localhost:8888) <br>  
  (Use the token printed in the container logs.)
  <br>  
  Notes:
  > For manuscript reproducibility, we also provide versioned tags (e.g., :v0.1.0). Prefer a version tag when matching a paper release.


- #### Option B: Build locally (development)
  ```bash
  # Option B1: Build locally with Compose
  # from the repo root
  docker compose -f docker/docker-compose.yml build
  docker compose -f docker/docker-compose.yml up
  ```
  Open Jupyter:
  [http://localhost:8888](http://localhost:8888)  <br>  
  Workspace mount: /work

  
  ```bash
  # Option B2: Build locally without Compose (alternative)
  # from the repo root
  docker build -f docker/Dockerfile -t llm-sc-curator:official .
  ```
  Run Jupyter:
  ```
  docker run --rm -it \
    -p 8888:8888 \
    -v "$PWD":/work \
    -e GEMINI_API_KEY \
    -e OPENAI_API_KEY \
    llm-sc-curator:official
  ```
  Open Jupyter:
  [http://localhost:8888](http://localhost:8888)  <br>  

---
## 🖥️ Apptainer / Singularity (HPC)
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
    bash -lc 'jupyter lab --ip=0.0.0.0 --port=8888 --no-browser 
  ```
---
## 🔒 Privacy
We respect the sensitivity of clinical and biological data. **LLM-scCurator** is architected to ensure that raw expression matrices and cell-level metadata never leave your local environment.
- Local execution: All heavy lifting—preprocessing, confounding gene removal, and feature ranking—occurs locally on your machine.
- Minimal transmission: When interfacing with external LLM APIs, the system transmits only anonymized, cluster-level marker lists (e.g., top 50 ranked gene symbols) and basic tissue context.
- User control: You retain control over any additional background information (e.g., disease state, treatment conditions, and platform) provided via custom prompts. Please review your institution’s data policy and the LLM provider’s terms before sending any information to external LLM APIs.

---
## ⚡ Quick Start
### 🐍 For Python / Scanpy Users
1) Set your API key (simplest: paste in the notebook)
  ```python
  GEMINI_API_KEY = "PASTE_YOUR_KEY_HERE"
  # OPENAI_API_KEY = "PASTE_YOUR_KEY_HERE"  # optional
  ```

2) Run LLM-scCurator
  ```python
  import scanpy as sc
  from llm_sc_curator import LLMscCurator
  
  # Initialize with your API Key (Google AI Studio)
  curator = LLMscCurator(api_key=GEMINI_API_KEY)
  
  # Load your data
  adata = sc.read_h5ad("my_data.h5ad")
  
  # Run fully automated hierarchical annotation
  adata = curator.run_hierarchical_discovery(adata)
  
  # Visualize
  sc.pl.umap(adata, color=['major_type', 'fine_type'])
  ```


### 📊 For R / Seurat Users
You can use **LLM-scCurator** in two ways:

- **Option A (recommended): Export → run in Python** 
  We provide a helper script [`examples/R/export_to_curator.R`](https://github.com/kenflab/LLM-scCurator/blob/main/examples/R/export_to_curator.R) to export your Seurat object seamlessly for processing in Python.
  ```R  
  source("examples/R/export_to_curator.R")
  Rscript examples/R/export_to_curator.R \
    --in_rds path/to/seurat_object.rds \
    --outdir out_seurat \
    --cluster_col seurat_clusters
  ```
  Output:
  - `counts.mtx` (raw counts; recommended)
  - `features.tsv` (gene list)
  - `obs.csv` (cell metadata; includes seurat_clusters)
  - `umap.csv` (optional, if available)
  
  Notes:
   > * The folder will contain: counts.mtx, features.tsv, obs.csv (and umap.csv if available).
   > * Then continue in the Python/Colab tutorial to run LLM-scCurator and write cluster_curated_map.csv,
   > * which can be re-imported into Seurat for plotting.


- #### Option B: Run from R via reticulate (advanced)
  If you prefer to stay in R, you can invoke the Python package via reticulate (Python-in-R).
  This is more sensitive to Python environment configuration, so we recommend Option A for most users.
  - Use the **[official Docker](https://github.com/kenflab/LLM-scCurator/blob/main/README.md#-docker-official-environment) (Python + R + Jupyter)** and follow the step-by-step tutorial notebook: 📓 [`examples/R/run_llm_sccurator_R_reticulate.ipynb`](https://github.com/kenflab/LLM-scCurator/blob/main/examples/R/run_llm_sccurator_R_reticulate.ipynb) <br> 
    The notebook includes:
      - Use LLM-scCurator for robust marker distillation (no API key required)
      - 🔑 [Optional](https://github.com/kenflab/LLM-scCurator/blob/main/README.md#-backends-llm-api-keys-setup): Requires an API key for LLM-scCurator annotation .


---
## 📄 Manuscript reproduction
For manuscript-facing verification (benchmarks, figures, and Source Data), use the versioned assets under [`paper/`](https://github.com/kenflab/LLM-scCurator/blob/main/paper). See [`paper/README.md`](https://github.com/kenflab/LLM-scCurator/blob/main/paper#readme) for the primary instructions.

Notes:
 > * Figures are supported by exported Source Data in [`paper/source_data/`](https://github.com/kenflab/LLM-scCurator/blob/main/paper/source_data) (see [`paper/FIGURE_MAP.csv`](https://github.com/kenflab/LLM-scCurator/blob/main/paper/FIGURE_MAP.csv)  for panel → file mapping).
 > * Re-running LLM/API calls or external reference annotators is optional; LLM API outputs may vary across runs even with temperature=0.
 > * For transparency, we include read-only provenance notebooks with example run logs in [`paper/notebooks/`](https://github.com/kenflab/LLM-scCurator/blob/main/paper/notebooks)

---
### 📓 Colab notebooks

- **Python / Scanpy quickstart (recommended: [colab_quickstart.ipynb](https://github.com/kenflab/LLM-scCurator/blob/main/examples/colab/colab_quickstart.ipynb))**
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenflab/LLM-scCurator/blob/master/examples/colab/colab_quickstart.ipynb) <br>
    ☝️ Runs end-to-end on a public Scanpy dataset (**no API key required** by default).  
    - 🔑 [Optional](https://github.com/kenflab/LLM-scCurator/blob/main/README.md#-backends-llm-api-keys-setup): If an API key is provided (replace `GEMINI_API_KEY = "YOUR_KEY_HERE"`), the notebook can also run **LLM-scCurator automatic hierarchical cell annotation**.

  - **OpenAI quickstart (OpenAI backend: [colab_quickstart_openai.ipynb](https://github.com/kenflab/LLM-scCurator/blob/main/examples/colab/colab_quickstart_openai.ipynb))**
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenflab/LLM-scCurator/blob/master/examples/colab/colab_quickstart_openai.ipynb) <br>
    ☝️ Same workflow as the Python / Scanpy quickstart, but configured for the OpenAI backend.
    - 🔑 [Optional](https://github.com/kenflab/LLM-scCurator/blob/main/README.md#-backends-llm-api-keys-setup): If an API key is provided (replace `OPENAI_API_KEY= "YOUR_KEY_HERE"`), the notebook can also run **LLM-scCurator automatic hierarchical cell annotation**. `OPENAI_API_KEY` requires OpenAI API billing (paid API credits).

- **R / Seurat quickstart (export → Python LLM-scCurator → back to Seurat: [colab_quickstart_R.ipynb](https://github.com/kenflab/LLM-scCurator/blob/main/examples/colab/colab_quickstart_R.ipynb))**
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenflab/LLM-scCurator/blob/master/examples/colab/colab_quickstart_R.ipynb) <br>
    ☝️ Runs a minimal Seurat workflow in R, exports a Seurat object to an AnnData-ready folder, runs LLM-scCurator in Python, then re-imports labels into Seurat for visualization and marker sanity checks.  
    - 🔑 [Optional](https://github.com/kenflab/LLM-scCurator/blob/main/README.md#-backends-llm-api-keys-setup): Requires an API key for LLM-scCurator annotation (same setup as above).
    - Recommended for Seurat users who want to keep Seurat clustering/UMAP but use LLM-scCurator for robust marker distillation and annotation.

---
## 🔑 Backends (LLM API keys) Setup

Set your provider API key as an environment variable:
- `GEMINI_API_KEY` for Google Gemini
- `OPENAI_API_KEY` for OpenAI API

See each provider’s documentation for how to obtain an API key and for current usage policies.
![Get API Key GIF](https://github.com/user-attachments/assets/70791b03-341d-4449-af07-1d181768f01c)



- **Option A (Gemini steps):**
  1.  Go to **[Google AI Studio](https://aistudio.google.com/)**.
  2.  Log in with your Google Account.
  3.  Click **Get API key** (top-left) $\rightarrow$ **Create API key**.
  4.  Copy the key and use it in your code. <br>  


- **Option B (OpenAI steps):**
  1.  Go to **[OpenAI Platform](https://platform.openai.com/api-keys)**.
  2.  Log in with your OpenAI Account.
  3.  Click **Create new secret key** $\rightarrow$ **Create secret key**.
  4.  Copy the key and use it in your code.

Notes:
> Google Gemini can be used within its free-tier limits. <br> 
> OpenAI API usage requires enabling billing (paid API credits); ChatGPT subscriptions (e.g. Plus) do NOT include API usage. 

---
## Citation
Zenodo archive (v0.1.0) DOI: [10.5281/zenodo.17970494](https://doi.org/10.5281/zenodo.17970494)  
GitHub release tag: [v0.1.0](https://github.com/kenflab/LLM-scCurator/releases/tag/v0.1.0)




