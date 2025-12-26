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

LLM-scCurator performs preprocessing and feature distillation **locally**. When using external LLM APIs, it typically
transmits only **compact, cluster-level marker summaries** (e.g., ranked gene symbols and minimal context), not raw
expression matrices or cell-level metadata. Please review your institution’s data policy and the LLM provider’s terms
before sending any information to external services.

---

## Links

Project home and issue tracker:

- Source code and releases: **[GitHub](https://github.com/kenflab/LLM-scCurator)**
- Report issues / questions: **[GitHub Issues](https://github.com/kenflab/LLM-scCurator/issues)**

