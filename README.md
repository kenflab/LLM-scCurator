# LLM-scCurator

LLM-scCurator 🧬🤖

**Dynamic feature masking to improve robustness of zero-shot cell-type annotation with LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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

## ⚡ Quick Start

### Python (Standard Usage)
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
We provide a helper script to export your Seurat object seamlessly.
```R
source("examples/R/export_script.R")
export_for_llm_curator(seurat_obj, "my_data.h5ad")
```

## 🔑 Backends (API keys) Setup

Set your provider API key as an environment variable:
- GEMINI_API_KEY for Google Gemini
- OPENAI_API_KEY for OpenAI API

![Get API Key GIF](https://github.com/user-attachments/assets/70791b03-341d-4449-af07-1d181768f01c)

**Steps:**
1.  Go to **[Google AI Studio](https://aistudio.google.com/)**.
2.  Log in with your Google Account.
3.  Click **"Get API key"** (top-left) $\rightarrow$ **"Create API key"**.
4.  Copy the key and use it in your code.

> * Privacy & Cost Note: > **Free of Charge:** The "Free Tier" is sufficient for testing and small-scale analysis. No payment information is required.
> * Data Privacy: For clinical datasets requiring strict privacy (HIPAA/GDPR compliance) or zero-training policies, we recommend upgrading to the **Vertex AI** endpoint on Google Cloud Platform.
