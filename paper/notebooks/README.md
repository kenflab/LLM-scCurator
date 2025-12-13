# notebooks/

This folder contains optional, read-only notebooks used during development (e.g., Colab runs and logs).

Notebooks are **not** the primary reproduction path.
The canonical, reproducible workflow lives in `paper/scripts/` and is driven by `paper/config/`.

- Use notebooks for inspection and provenance (human-readable logs).
- Do not rely on notebooks for figure generation in the final pipeline.
- Large outputs (e.g., `.h5ad`, images, caches) should not be committed here.
