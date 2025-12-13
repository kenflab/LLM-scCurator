# LLM-scCurator

LLM-scCurator 🧬🤖

**Dynamic feature masking to improve robustness of zero-shot cell-type annotation with LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)


## 🚀 Overview
**LLM-scCurator**  standardizes *noise-aware marker distillation* (clonotype/housekeeping/stress suppression + rescue + lineage leakage filters)
before prompting an LLM, and supports hierarchical (coarse-to-fine) annotation for scRNA-seq and spatial data.


### Key Features
- **🛡️ Noise-Aware Filtering:** Automatically removes lineage-specific noise (TCR/Ig) and state-dependent noise (ribosomal/mitochondrial).
- **🧠 Context-Aware Inference:** Automatically infers lineage context (e.g., "T cell") to guide LLM reasoning.
- **🔬 Hierarchical Discovery:** One-line function to dissect complex tissues into major lineages and fine-grained subtypes.
- **🌍 Spatial Ready:** Validated on scRNA-seq (10x) and spatial transcriptomics (Xenium, Visium).


## 📦 Installation

```bash
# 1. Clone the repository
git clone [https://github.com/kenflab/LLM-scCurator.git](https://github.com/kenflab/LLM-scCurator.git)

# 2. Navigate to the directory
cd LLM-scCurator

# 3. Install the package (and dependencies)
pip install .
```

## 🐳 Docker (Reproducible environment)

We provide two images:
- **lite**: Python + R + Jupyter (sufficient for running LLM-scCurator and most paper figure generation)
- **full**: lite + additional/heavier dependencies for extended analyses (spatial + extra R ecosystem)

### Quick start (lite)
```bash
# from the repo root
docker compose -f docker/docker-compose.yml build lite
docker compose -f docker/docker-compose.yml up lite
```

Open Jupyter:
[http://localhost:8888](http://localhost:8888)

The repository is mounted into the container at /work.

### Optional: full image
```bash
docker compose -f docker/docker-compose.yml build full
docker compose -f docker/docker-compose.yml up full
```

### API keys (Docker)
Set environment variables in your shell before docker compose up:
```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
```
Notes:
> * Never commit API keys to the repository. Use environment variables or a local .env file (not tracked).
> * LLM providers’ availability, pricing, and data handling policies may vary; please follow each provider’s terms and your institutional requirements.


## ⚡ Quick Start
### 🐍 Python (Standard Usage)
```python
import scanpy as sc
from llm_sc_curator import LLMscCurator

# Initialize with your API Key (Google AI Studio)
curator = LLMscCurator(api_key="YOUR_GEMINI_API_KEY")

# Load your data
adata = sc.read_h5ad("my_data.h5ad")

# 🚀 Run fully automated hierarchical annotation
adata = curator.run_hierarchical_discovery(adata)

# Visualize
sc.pl.umap(adata, color=['major_type', 'fine_type'])
```


### For R / Seurat Users
You can use LLM-scCurator in two ways:

#### Option 1: Run directly in R (via reticulate)
You can install and import the package directly within your R session.

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

#### Option 2: Export to Python (Recommended for large pipelines)
We provide a helper script to export your Seurat object seamlessly to .h5ad for processing in Python.
```R
source("examples/R/export_script.R")
export_for_llm_curator(seurat_obj, "my_data.h5ad")
```


### 📓 Colab / Notebooks

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
4.  Copy the key and use it in your cod
