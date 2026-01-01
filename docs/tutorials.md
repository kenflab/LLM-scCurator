# Tutorials

Short, task-oriented walkthroughs.  
Use these pages when you want an end-to-end example rather than a conceptual overview.

---

## Quickstart

Choose the fastest path for your environment:
- **Python / Scanpy (Colab):** minimal end-to-end run on a public dataset  
  → [Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/master/examples/colab/colab_quickstart.ipynb)

- **OpenAI backend (Colab):** same workflow as the Python / Scanpy quickstart, but configured for the OpenAI backend. `OPENAI_API_KEY` requires OpenAI API billing (paid API credits). <br> 
    → [Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/master/examples/colab/colab_quickstart_openai.ipynb)

- **R / Seurat (Colab):** Seurat-oriented quickstart with an export → Python run → re-import pattern  
  → [Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/master/examples/colab/colab_quickstart_R.ipynb)

---

## Advanced  
- **R via reticulate:** run LLM-scCurator from R through a Python bridge (advanced usage)  
  → [Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/main/examples/R/run_llm_sccurator_R_reticulate.ipynb)

- **Spatial validation (paper):** manuscript notebooks for Xenium/Visium spatial plots and pseudo-bulk summaries  
  → **Colon Xenium (Fig.2h; ED Fig.3d):** [Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/master/paper/notebooks/07_Colon_Xenium.ipynb)  
  → **OSCC Visium (ED Fig.3a–c):** [Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/master/paper/notebooks/08_OSCC_Visium.ipynb)

- **Fully local LLM (Ollama):** Curate features and optionally annotate clusters using a local LLM backend (no external transmission). 
[Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/main/examples/local/local_quickstart_ollama.ipynb)

- **Local feature distillation → Approved chat LLM annotation (no external LLM API calls):** Curate features locally, export a curated cluster→genes table, then annotate it via an institution-approved **chat interface** (e.g., Microsoft Copilot “Work”) by uploading the CSV/Excel or pasting markers. 
[Open notebook on GitHub](https://github.com/kenflab/LLM-scCurator/blob/main/examples/local/local_quickstart_approved_ai_workflow.ipynb)



---

## Notes

- For the underlying design (masking, rescue, leakage filtering, hierarchy), see **[Concepts](concepts.md)**.
- For deterministic reproduction (benchmarks/figures/Source Data), follow **[paper/README.md](https://github.com/kenflab/LLM-scCurator/blob/master/paper/README.md)**.
