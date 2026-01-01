# LLM-scCurator

<p align="center">
  <img src="assets/LLM-scCurator_logo.png" style="max-width: 480px; width: 100%;" alt="LLM-scCurator logo">
</p>

LLM-scCurator is a Python framework for **noise-aware marker distillation** that improves the robustness of
**zero-shot cell-type annotation with LLMs** across **single-cell and spatial transcriptomics**.

---

## What it does

LLM-scCurator adds a **pre-prompt feature distillation layer**: it suppresses recurrent biological/technical
programs (e.g., ribosomal/mitochondrial, stress, cell cycle, TCR/Ig) while **rescuing lineage- and state-defining
markers**, then applies leakage-safe lineage filters before LLM prompting.

## Why it helps

- **Stabilizes inputs:** reduces “garbage-in” marker lists by masking clonotype and ubiquitous programs.
- **Preserves biology:** specificity-aware rescue retains informative, lineage-restricted markers.
- **Scales to discovery:** supports hierarchical, coarse-to-fine annotation for complex tissues, including spatial
  modalities (Visium/Xenium).

---

## Quick start

- Start here: **[Getting started](getting-started.md)**
- Key ideas (masking, rescue, leakage filter, hierarchical inference): **[Concepts](concepts.md)**
- Practical workflows and recipes: **[User guide](user-guide.md)**
- Tutorials: **[Tutorials](tutorials.md)**
- Full API docs: **[API reference](api-reference.md)**

---

## Privacy

We respect the sensitivity of clinical and biological data. LLM-scCurator is designed so that raw expression matrices and cell-level metadata can remain within your local environment.

- **Local execution:** Preprocessing, confounder masking, and feature ranking run locally on your machine.
- **Minimal transmission (optional):** If you choose to use an external LLM API, only anonymized, cluster-level marker lists (e.g., top 50 gene symbols) and minimal tissue context are sent.
- **User control:** You decide what additional context (e.g., disease state, treatment, platform) to include. Always follow institutional policy and the LLM provider’s terms before sharing any information.

### Example workflows (institutional-policy friendly)

Many institutions restrict which AI tools can be used with internal clinical or research datasets. To support these real-world constraints, we provide two end-to-end workflows that keep raw matrices and cell-level metadata local and avoid external LLM API calls unless explicitly permitted:

- **Fully local LLM (Ollama):** Curate features and optionally annotate clusters using a local LLM backend (no external transmission). 
[`examples/local/local_quickstart_ollama.ipynb`](https://github.com/kenflab/LLM-scCurator/blob/main/examples/local/local_quickstart_ollama.ipynb)

- **Local feature distillation → Approved chat LLM annotation (no external LLM API calls):** Curate features locally, export a curated cluster→genes table, then annotate it via an institution-approved **chat interface** (e.g., Microsoft Copilot “Work”) by uploading the CSV/Excel or pasting markers. 
[`examples/local/local_quickstart_approved_ai_workflow.ipynb`](https://github.com/kenflab/LLM-scCurator/blob/main/examples/local/local_quickstart_approved_ai_workflow.ipynb)

---

## Links

Project home and issue tracker:

- Source code and releases: **[GitHub](https://github.com/kenflab/LLM-scCurator)**
- Report issues / questions: **[GitHub Issues](https://github.com/kenflab/LLM-scCurator/issues)**

