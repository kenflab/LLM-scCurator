# LLM-scCurator

LLM-scCurator 🧬🤖

**Dynamic feature masking to improve robustness of zero-shot cell-type annotation with LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)

---

## 🚀 Overview
**LLM-scCurator**  standardizes *noise-aware marker distillation* (clonotype/housekeeping/stress suppression + rescue + lineage leakage filters)
before prompting an LLM, and supports hierarchical (coarse-to-fine) annotation for scRNA-seq and spatial data.


### Key Features
- **🛡️ Noise-Aware Filtering:** Automatically removes lineage-specific noise (TCR/Ig) and state-dependent noise (ribosomal/mitochondrial).
- **🧠 Context-Aware Inference:** Automatically infers lineage context (e.g., "T cell") to guide LLM reasoning.
- **🔬 Hierarchical Discovery:** One-line function to dissect complex tissues into major lineages and fine-grained subtypes.
- **🌍 Spatial Ready:** Validated on scRNA-seq (10x) and spatial transcriptomics (Xenium, Visium).

---
## 📦 Installation

```bash
# 1. Clone the repository
git clone [https://github.com/kenflab/LLM-scCurator.git](https://github.com/kenflab/LLM-scCurator.git)

# 2. Navigate to the directory
cd LLM-scCurator

# 3. Install the package (and dependencies)
pip install .
```
Notes:
> *　If you already have a Scanpy/Seurat pipeline environment, you can install into that environment.
---
## 🐳 Docker (official environment)

This repository provides an official Docker environment (including Python, R, and Jupyter), sufficient to run LLM-scCurator and most paper figure generation.

```bash
# from the repo root
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

Open Jupyter:
[http://localhost:8888](http://localhost:8888)
Workspace mount: /work

---
## Apptainer / Singularity (HPC)
Build a .sif from the Docker image:
```bash
docker compose -f docker/docker-compose.yml build
apptainer build llm-sc-curator.sif docker-daemon://llm-sc-curator:official
```

Run Jupyter:
```bash
apptainer exec --cleanenv \
  --bind "$PWD":/work \
  llm-sc-curator.sif \
  bash -lc 'jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token="" --NotebookApp.password=""'
```

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

# 🚀 Run fully automated hierarchical annotation
adata = curator.run_hierarchical_discovery(adata)

# Visualize
sc.pl.umap(adata, color=['major_type', 'fine_type'])
```


### For R / Seurat Users
You can use LLM-scCurator in two ways:

- #### Option A (recommended): Export to .h5ad → run in Python
  We provide a helper script to export your Seurat object seamlessly to .h5ad for processing in Python.
```R
source("examples/R/export_script.R")
export_for_llm_curator(seurat_obj, "my_data.h5ad")
```


- #### Option B: Use from R via reticulate (advanced)
  Use reticulate to call the Python package installed in your environment.

```R
# install.packages("reticulate")
library(reticulate)

# 1. Install LLM-scCurator (one-time setup)
py_install("llm-sc-curator", pip = TRUE)

# 2. Import and use
lsc <- import("llm_sc_curator")
curator <- lsc$LLMscCurator(api_key = "YOUR_KEY")

# (Assuming you have converted your Seurat obj to AnnData or h5ad)
# result <- curator$run_hierarchical_discovery(adata)
```

---

### 📓 Colab / Notebooks

---
## 🔑 Backends (API keys) Setup

Set your provider API key as an environment variable:
Set your provider API key as an environment variable:
- `GEMINI_API_KEY` for Google Gemini
- `OPENAI_API_KEY` for OpenAI API

See each provider’s documentation for how to obtain an API key and for current usage policies.

![Get API Key GIF](https://github.com/user-attachments/assets/70791b03-341d-4449-af07-1d181768f01c)

**Steps:**
1.  Go to **[Google AI Studio](https://aistudio.google.com/)**.
2.  Log in with your Google Account.
3.  Click **"Get API key"** (top-left) $\rightarrow$ **"Create API key"**.
4.  Copy the key and use it in your code.
