"""Prospective HLCA core external validation for LLM-scCurator.

The module deliberately separates marker generation, LLM inference, blinded label
mapping, and scoring. Ground-truth annotations are never included in an LLM prompt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MISSING_LABELS = {"", "none", "nan", "na", "n/a", "unknown", "not applicable"}
CONDITIONS = ("standard", "full_core")
CONFIDENCE_VALUES = {"High", "Medium", "Low"}


@dataclass(frozen=True)
class Config:
    source_path: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing TOML section: [{name}]")
        return value

    def path(self, name: str) -> Path:
        value = self.section("paths").get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing path setting: paths.{name}")
        path = Path(value)
        if not path.is_absolute():
            path = self.source_path.parent / path
        return path.resolve()


def load_config(path: str | Path) -> Config:
    import tomllib

    source = Path(path).resolve()
    with source.open("rb") as handle:
        values = tomllib.load(handle)
    cfg = Config(source_path=source, values=values)
    for section in ("dataset", "columns", "analysis", "llm", "paths"):
        cfg.section(section)
    return cfg


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _head(url: str) -> tuple[int | None, str | None]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        size_raw = response.headers.get("Content-Length")
        etag_raw = response.headers.get("ETag")
    size = int(size_raw) if size_raw else None
    etag = etag_raw.strip('"') if etag_raw else None
    return size, etag


def download_hlca(cfg: Config, *, force: bool = False, compute_sha256: bool = True) -> Path:
    """Download the pinned CELLxGENE LTS source H5AD with safe resume."""

    dataset = cfg.section("dataset")
    url = str(dataset["url"])
    expected_bytes = int(dataset["expected_bytes"])
    expected_etag = str(dataset.get("expected_etag", "")).strip()
    destination = cfg.path("source_h5ad")
    metadata_path = cfg.path("download_metadata")
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote_size, remote_etag = _head(url)
    if remote_size != expected_bytes:
        raise RuntimeError(
            f"Remote size changed: expected {expected_bytes}, observed {remote_size}. "
            "Do not continue until the pinned dataset is verified."
        )
    if expected_etag and remote_etag != expected_etag:
        raise RuntimeError(
            f"Remote ETag changed: expected {expected_etag}, observed {remote_etag}. "
            "Do not continue until the pinned dataset is verified."
        )

    if destination.exists() and not force:
        if destination.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Existing file has the wrong size: {destination}. "
                "Use --force only after checking the target path."
            )
        digest = sha256_file(destination) if compute_sha256 else None
        write_json(
            metadata_path,
            {
                "dataset_id": dataset["dataset_id"],
                "census_version": dataset["census_version"],
                "url": url,
                "bytes": expected_bytes,
                "etag": remote_etag,
                "sha256": digest,
                "verified_at_utc": utc_now(),
                "reused_existing_file": True,
            },
        )
        return destination

    if destination.exists() and force:
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_bytes:
        raise RuntimeError(f"Partial file is larger than expected: {partial}")

    headers: dict[str, str] = {}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if start else "wb"
    with urllib.request.urlopen(request, timeout=300) as response, partial.open(mode) as handle:
        if start and getattr(response, "status", None) != 206:
            raise RuntimeError("Server did not honor the Range request; refusing unsafe append.")
        copied = start
        last_report = time.monotonic()
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            now = time.monotonic()
            if now - last_report >= 10:
                print(f"Downloaded {copied / 1e9:.2f}/{expected_bytes / 1e9:.2f} GB", flush=True)
                last_report = now

    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Incomplete download: {partial.stat().st_size} of {expected_bytes} bytes. "
            "Run the same command again to resume."
        )
    os.replace(partial, destination)
    digest = sha256_file(destination) if compute_sha256 else None
    write_json(
        metadata_path,
        {
            "dataset_id": dataset["dataset_id"],
            "census_version": dataset["census_version"],
            "url": url,
            "bytes": expected_bytes,
            "etag": remote_etag,
            "sha256": digest,
            "completed_at_utc": utc_now(),
            "reused_existing_file": False,
        },
    )
    return destination


def clean_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    label = str(value).strip()
    return None if label.casefold() in MISSING_LABELS else label


def normalized_label(value: Any) -> str:
    text = clean_label(value) or ""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def majority_label(values: Iterable[Any]) -> tuple[str | None, int, int]:
    cleaned = [x for x in (clean_label(value) for value in values) if x is not None]
    if not cleaned:
        return None, 0, 0
    counts = Counter(cleaned)
    best_count = max(counts.values())
    best = sorted(label for label, count in counts.items() if count == best_count)[0]
    return best, best_count, len(cleaned)


def build_cluster_manifest(obs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    columns = cfg.section("columns")
    cluster_col = str(columns["cluster"])
    l1_col = str(columns["level_1"])
    l2_col = str(columns["level_2"])
    l3_col = str(columns["level_3"])
    donor_col = str(columns["donor"])
    dataset_col = str(columns["source_dataset"])
    required = [cluster_col, l1_col, l2_col, l3_col, donor_col, dataset_col]
    missing = [name for name in required if name not in obs.columns]
    if missing:
        raise ValueError(f"HLCA object is missing required obs columns: {missing}")

    rows: list[dict[str, Any]] = []
    cluster_values = obs[cluster_col].map(clean_label)
    for cluster_id in sorted(x for x in cluster_values.dropna().unique()):
        sub = obs.loc[cluster_values == cluster_id]
        gt3, gt3_count, _ = majority_label(sub[l3_col])
        if gt3 is None:
            gt2 = None
            gt1 = None
            evaluable = False
            reason = "all ann_level_3 values missing"
        else:
            state_cells = sub.loc[sub[l3_col].map(clean_label) == gt3]
            gt2, _, _ = majority_label(state_cells[l2_col])
            gt1, _, _ = majority_label(state_cells[l1_col])
            evaluable = gt1 is not None and gt2 is not None
            reason = "" if evaluable else "majority ann_level_3 lacks a complete level 1-3 path"
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_cells_full": int(len(sub)),
                "n_donors": int(sub[donor_col].map(clean_label).nunique(dropna=True)),
                "n_source_datasets": int(
                    sub[dataset_col].map(clean_label).nunique(dropna=True)
                ),
                "gt_ann_level_1": gt1 or "",
                "gt_ann_level_2": gt2 or "",
                "gt_ann_level_3": gt3 or "",
                "gt_ann_level_3_purity": (
                    float(gt3_count / len(sub)) if len(sub) else np.nan
                ),
                "evaluable": bool(evaluable),
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_label_hierarchy(obs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    columns = cfg.section("columns")
    names = [str(columns[f"level_{level}"]) for level in (1, 2, 3)]
    tmp = obs.loc[:, names].copy()
    for name in names:
        tmp[name] = tmp[name].map(clean_label)
    tmp = tmp.dropna(subset=names)
    hierarchy = (
        tmp.groupby(names, observed=True, dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .rename(
            columns={
                names[0]: "ann_level_1",
                names[1]: "ann_level_2",
                names[2]: "ann_level_3",
            }
        )
    )
    hierarchy = hierarchy.sort_values(
        ["ann_level_1", "ann_level_2", "ann_level_3", "n_cells"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    hierarchy.insert(
        0,
        "path_id",
        [f"HLCA_PATH_{i:03d}" for i in range(1, len(hierarchy) + 1)],
    )
    return hierarchy


class NullBackend:
    """Backend placeholder for marker-only LLM-scCurator execution."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:  # pragma: no cover
        raise RuntimeError("NullBackend cannot make annotation calls")


def _make_curator_without_llm():
    from llm_sc_curator import LLMscCurator
    from llm_sc_curator.backends import BaseLLMBackend

    class _Backend(NullBackend, BaseLLMBackend):
        pass

    return LLMscCurator(backend=_Backend())


def _gene_symbol_series(var: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in var.columns:
            symbols = var[candidate].astype("string").str.strip()
            if symbols.notna().any():
                return symbols
    return pd.Series(var.index.astype(str), index=var.index, dtype="string")


def _select_sample_positions(obs: pd.DataFrame, manifest: pd.DataFrame, cfg: Config) -> np.ndarray:
    analysis = cfg.section("analysis")
    cluster_col = str(cfg.section("columns")["cluster"])
    maximum = int(analysis["max_cells_per_cluster"])
    seed = int(analysis["seed"])
    rng = np.random.default_rng(seed)
    clusters = set(manifest.loc[manifest["evaluable"], "cluster_id"].astype(str))
    values = obs[cluster_col].map(clean_label)
    positions: list[np.ndarray] = []
    for cluster_id in sorted(clusters):
        idx = np.flatnonzero((values == cluster_id).to_numpy())
        if len(idx) > maximum:
            idx = np.sort(rng.choice(idx, size=maximum, replace=False))
        positions.append(idx)
    if not positions:
        raise RuntimeError("No evaluable clusters were found")
    return np.sort(np.concatenate(positions))


def _standard_markers(adata: Any, cluster_col: str, n_markers: int) -> dict[str, list[str]]:
    import scanpy as sc

    key = "hlca_standard_wilcoxon"
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_col,
        reference="rest",
        method="wilcoxon",
        use_raw=False,
        key_added=key,
    )
    output: dict[str, list[str]] = {}
    for cluster_id in sorted(adata.obs[cluster_col].astype(str).unique()):
        table = sc.get.rank_genes_groups_df(adata, group=cluster_id, key=key)
        output[cluster_id] = table["names"].dropna().astype(str).head(n_markers).tolist()
    return output


def prepare_markers(cfg: Config, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Create the fixed cell sample, audit manifest, and two marker lists."""

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    source = cfg.path("source_h5ad")
    processed_path = cfg.path("processed_h5ad")
    manifest_path = cfg.path("cluster_manifest")
    hierarchy_path = cfg.path("label_hierarchy")
    markers_path = cfg.path("marker_lists")
    if not source.exists():
        raise FileNotFoundError(f"Run the download stage first: {source}")
    if markers_path.exists() and not overwrite:
        raise FileExistsError(f"Marker output already exists: {markers_path}")
    for path in (processed_path, manifest_path, hierarchy_path, markers_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening backed H5AD: {source}")
    backed = sc.read_h5ad(source, backed="r")
    try:
        expected_n_obs = int(cfg.section("dataset")["expected_n_obs"])
        if backed.n_obs != expected_n_obs:
            raise ValueError(
                f"HLCA cell count changed: expected {expected_n_obs}, observed {backed.n_obs}"
            )
        obs = backed.obs.copy()
        if backed.raw is None:
            raise ValueError("HLCA H5AD does not contain adata.raw.X raw counts")
        manifest = build_cluster_manifest(obs, cfg)
        hierarchy = build_label_hierarchy(obs, cfg)
        selected_positions = _select_sample_positions(obs, manifest, cfg)
        sampled = backed[selected_positions, :].to_memory()
    finally:
        if getattr(backed, "file", None) is not None:
            backed.file.close()

    cluster_col = str(cfg.section("columns")["cluster"])
    sampled.obs[cluster_col] = sampled.obs[cluster_col].astype(str)
    sampled_obs = sampled.obs.copy()
    raw = sampled.raw.to_adata()
    del sampled
    candidates = [str(x) for x in cfg.section("columns")["gene_symbol_candidates"]]
    symbols = _gene_symbol_series(raw.var, candidates)
    valid = symbols.notna() & symbols.ne("")
    duplicate = symbols.duplicated(keep=False)
    keep = (valid & ~duplicate).to_numpy()
    if not keep.any():
        raise RuntimeError("No unique gene symbols were available in adata.raw.var")

    matrix = raw.X[:, keep]
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    var = raw.var.iloc[np.flatnonzero(keep)].copy()
    var.insert(0, "source_feature_id", var.index.astype(str))
    var.index = pd.Index(symbols.iloc[np.flatnonzero(keep)].astype(str), name="gene_symbol")
    adata = ad.AnnData(X=matrix, obs=sampled_obs, var=var)
    adata.uns["hlca_external_validation"] = {
        "source_h5ad": str(source),
        "source_matrix": "adata.raw.X",
        "sample_seed": int(cfg.section("analysis")["seed"]),
        "max_cells_per_cluster": int(cfg.section("analysis")["max_cells_per_cluster"]),
        "dropped_missing_gene_symbols": int((~valid).sum()),
        "dropped_duplicated_gene_symbol_features": int(duplicate.sum()),
    }

    sc.pp.filter_genes(adata, min_cells=int(cfg.section("analysis")["min_cells_per_gene"]))
    sc.pp.normalize_total(
        adata,
        target_sum=float(cfg.section("analysis")["normalization_target_sum"]),
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(2000, adata.n_vars),
        flavor="seurat",
        subset=False,
    )
    adata.write_h5ad(processed_path, compression="gzip")

    sampled_counts = adata.obs[cluster_col].value_counts()
    manifest["n_cells_sampled"] = manifest["cluster_id"].map(sampled_counts).fillna(0).astype(int)
    manifest.to_csv(manifest_path, index=False)
    hierarchy.to_csv(hierarchy_path, index=False)

    n_markers = int(cfg.section("analysis")["n_markers"])
    standard = _standard_markers(adata, cluster_col, n_markers)
    curator = _make_curator_without_llm()
    curator.set_global_context(
        adata,
        balance_by=cluster_col,
        max_cells_per_group=int(cfg.section("analysis")["max_cells_per_cluster"]),
        min_cells_per_group=1,
        random_state=int(cfg.section("analysis")["seed"]),
    )

    progress_path = markers_path.with_suffix(".progress.csv")
    existing: dict[tuple[str, str], list[str]] = {}
    if progress_path.exists() and not overwrite:
        prior = pd.read_csv(progress_path)
        for row in prior.itertuples(index=False):
            existing[(str(row.cluster_id), str(row.condition))] = str(row.genes).split(";")
    if overwrite and progress_path.exists():
        progress_path.unlink()

    fieldnames = ["cluster_id", "condition", "n_genes", "genes"]
    write_header = not progress_path.exists()
    with progress_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for cluster_id in sorted(standard):
            key = (cluster_id, "standard")
            if key not in existing:
                genes = standard[cluster_id]
                writer.writerow(
                    {"cluster_id": cluster_id, "condition": "standard", "n_genes": len(genes), "genes": ";".join(genes)}
                )
                handle.flush()
            key = (cluster_id, "full_core")
            if key in existing:
                continue
            print(f"Curating markers for Leiden-3 cluster {cluster_id}", flush=True)
            genes = curator.curate_features(
                adata,
                group_col=cluster_col,
                target_group=cluster_id,
                n_top=n_markers,
                use_statistics=True,
                use_hvg=True,
                coarse_col=None,
            )
            genes = [str(gene) for gene in genes]
            if not genes:
                raise RuntimeError(f"full_core returned no genes for cluster {cluster_id}")
            writer.writerow(
                {"cluster_id": cluster_id, "condition": "full_core", "n_genes": len(genes), "genes": ";".join(genes)}
            )
            handle.flush()

    marker_table = pd.read_csv(progress_path, dtype={"cluster_id": str})
    marker_table = marker_table.drop_duplicates(["cluster_id", "condition"], keep="last")
    expected_clusters = set(standard)
    observed = set(zip(marker_table["cluster_id"].astype(str), marker_table["condition"].astype(str)))
    expected = {(cluster, condition) for cluster in expected_clusters for condition in CONDITIONS}
    missing = expected - observed
    if missing:
        raise RuntimeError(f"Marker preparation incomplete; missing {len(missing)} rows")
    marker_table = marker_table.sort_values(["cluster_id", "condition"]).reset_index(drop=True)
    marker_table.to_csv(markers_path, index=False)
    progress_path.unlink()
    return manifest_path, markers_path


def build_prompt(genes: Sequence[str], *, tissue: str, dataset_context: str) -> str:
    """Reproduce the package annotation prompt with fixed, identical context."""

    from llm_sc_curator.noise_lists import PROLIFERATION_SENTINELS

    context_str = "\n[Biological Context]\n"
    context_str += f"- Tissue/Condition: {tissue}\n"
    context_str += f"- Dataset: {dataset_context}\n"
    proliferation = [gene for gene in genes if gene in PROLIFERATION_SENTINELS]
    if proliferation:
        show = ", ".join(proliferation[:3])
        context_str += (
            f"- Note: Proliferation markers detected ({show}). "
            "Prioritize identifying the lineage/subtype, but append "
            "'(proliferating)' after the main cell type when appropriate.\n"
        )
    genes_str = ", ".join(genes)
    return f"""
        Role: Expert in single-cell transcriptomics.
        Context:
        {context_str}
        Input Genes: [{genes_str}]

        Task:
          Identify the SINGLE BEST Subtype & Lineage.
          The main part of `cell_type` MUST describe the lineage/subtype
          (e.g., "CD8+ exhausted T cell", "CD4 Temra.EffMem T cell", "MAIT cell", "Naive B cell").
          State information (ISG-high, proliferating, etc.) should be placed
          in parentheses after the lineage, e.g. "CD8+ T cell (ISG-high)".

        Output STRICTLY in JSON. Do not include any additional text:
        {{
          "cell_type": "The precise subtype name",
          "confidence": "High/Medium/Low",
          "reasoning": "Brief justification based on key markers"
        }}
        """


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    for method_name in ("model_dump", "to_json_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:
                pass
    return str(value)


def parse_llm_json(text: str) -> dict[str, str]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, dict)), None)
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    cell_type = str(value.get("cell_type", "")).strip()
    confidence = str(value.get("confidence", "Low")).strip()
    reasoning = str(value.get("reasoning", "")).strip()
    if not cell_type or cell_type.casefold() in {"error", "parseerror"}:
        raise ValueError(f"Invalid cell_type in LLM response: {cell_type!r}")
    if confidence not in CONFIDENCE_VALUES:
        confidence = "Low"
    return {"cell_type": cell_type, "confidence": confidence, "reasoning": reasoning}


def safe_cluster_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def response_path(cfg: Config, cluster_id: str, condition: str, repeat: int) -> Path:
    directory = cfg.path("raw_responses_dir")
    name = f"{safe_cluster_id(cluster_id)}__{condition}__repeat_{repeat:02d}.json"
    return directory / name


def _gemini_call(client: Any, model: str, prompt: str, temperature: float) -> Any:
    from google.genai import types

    schema = {
        "type": "object",
        "properties": {
            "cell_type": {"type": "string"},
            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "reasoning": {"type": "string"},
        },
        "required": ["cell_type", "confidence", "reasoning"],
    }
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )


def run_gemini(
    cfg: Config,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """Run prospective repeated inference and save one complete JSON per call."""

    markers_path = cfg.path("marker_lists")
    if not markers_path.exists():
        raise FileNotFoundError(f"Run the prepare stage first: {markers_path}")
    marker_table = pd.read_csv(markers_path, dtype={"cluster_id": str})
    required = {"cluster_id", "condition", "genes"}
    if not required.issubset(marker_table.columns):
        raise ValueError(f"Marker table lacks columns: {sorted(required - set(marker_table.columns))}")

    llm = cfg.section("llm")
    analysis = cfg.section("analysis")
    model = str(llm["model"])
    temperature = float(llm["temperature"])
    repeats = int(analysis["n_repeats"])
    max_attempts = int(llm["max_attempts"])
    base_sleep = float(llm["retry_base_seconds"])
    api_key = os.environ.get("GEMINI_API_KEY")
    if not dry_run and not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment; it is never written to disk.")

    client = None
    sdk_version = None
    if not dry_run:
        from google import genai
        from importlib.metadata import version

        client = genai.Client(api_key=api_key)
        sdk_version = version("google-genai")

    jobs: list[tuple[str, str, int, list[str]]] = []
    for row in marker_table.itertuples(index=False):
        genes = [gene for gene in str(row.genes).split(";") if gene]
        for repeat in range(1, repeats + 1):
            jobs.append((str(row.cluster_id), str(row.condition), repeat, genes))
    jobs.sort(key=lambda value: (value[0], value[1], value[2]))
    if limit is not None:
        jobs = jobs[:limit]

    completed = 0
    for cluster_id, condition, repeat, genes in jobs:
        path = response_path(cfg, cluster_id, condition, repeat)
        if path.exists() and not overwrite:
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                if prior.get("status") == "success":
                    completed += 1
                    continue
            except Exception:
                pass
        prompt = build_prompt(
            genes,
            tissue=str(llm["tissue"]),
            dataset_context=str(llm["dataset_context"]),
        )
        if dry_run:
            print(f"\n--- {cluster_id} | {condition} | repeat {repeat} ---\n{prompt}")
            completed += 1
            continue

        record: dict[str, Any] = {
            "status": "failed",
            "cluster_id": cluster_id,
            "condition": condition,
            "repeat": repeat,
            "genes": genes,
            "n_genes": len(genes),
            "prompt": prompt,
            "provider": llm["provider"],
            "requested_model": model,
            "temperature": temperature,
            "sdk": "google-genai",
            "sdk_version": sdk_version,
            "started_at_utc": utc_now(),
            "attempts": [],
        }
        for attempt in range(1, max_attempts + 1):
            try:
                assert client is not None
                response = _gemini_call(client, model, prompt, temperature)
                raw_text = str(response.text)
                parsed = parse_llm_json(raw_text)
                record.update(
                    {
                        "status": "success",
                        "completed_at_utc": utc_now(),
                        "raw_response": raw_text,
                        "parsed": parsed,
                        "returned_model_version": getattr(response, "model_version", None),
                        "response_id": getattr(response, "response_id", None),
                        "usage_metadata": _jsonable(getattr(response, "usage_metadata", None)),
                        "prompt_feedback": _jsonable(getattr(response, "prompt_feedback", None)),
                    }
                )
                record["attempts"].append({"attempt": attempt, "status": "success"})
                break
            except Exception as exc:
                record["attempts"].append(
                    {"attempt": attempt, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
                if attempt < max_attempts:
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
        write_json(path, record)
        if record["status"] != "success":
            raise RuntimeError(
                f"Gemini call failed after {max_attempts} attempts: {cluster_id}, {condition}, repeat {repeat}. "
                f"Failure record: {path}"
            )
        print(f"Saved {path.name}", flush=True)
        completed += 1
    return completed


def load_successful_responses(cfg: Config) -> pd.DataFrame:
    directory = cfg.path("raw_responses_dir")
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return pd.DataFrame()
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "success":
            continue
        parsed = record.get("parsed") or {}
        rows.append(
            {
                "cluster_id": str(record["cluster_id"]),
                "condition": str(record["condition"]),
                "repeat": int(record["repeat"]),
                "cell_type": str(parsed.get("cell_type", "")),
                "confidence": str(parsed.get("confidence", "")),
                "reasoning": str(parsed.get("reasoning", "")),
                "response_file": str(
                    path.resolve().relative_to(cfg.source_path.parent.resolve())
                ),
                "requested_model": str(record.get("requested_model", "")),
                "returned_model_version": str(record.get("returned_model_version") or ""),
            }
        )
    return pd.DataFrame(rows)


def validate_inference_complete(cfg: Config, responses: pd.DataFrame) -> None:
    manifest = pd.read_csv(cfg.path("cluster_manifest"), dtype={"cluster_id": str})
    clusters = set(manifest.loc[manifest["evaluable"], "cluster_id"].astype(str))
    repeats = int(cfg.section("analysis")["n_repeats"])
    expected = {(cluster, condition, repeat) for cluster in clusters for condition in CONDITIONS for repeat in range(1, repeats + 1)}
    observed = set(
        zip(
            responses.get("cluster_id", pd.Series(dtype=str)).astype(str),
            responses.get("condition", pd.Series(dtype=str)).astype(str),
            responses.get("repeat", pd.Series(dtype=int)).astype(int),
        )
    )
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise RuntimeError(
            f"Inference set is not complete (expected={len(expected)}, observed={len(observed)}, "
            f"missing={len(missing)}, extra={len(extra)})."
        )


def create_mapping_template(cfg: Config, *, overwrite: bool = False) -> Path:
    """Create a condition- and GT-blinded prediction-to-HLCA mapping table."""

    output = cfg.path("mapping")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Mapping already exists and may contain manual work: {output}. Use --overwrite deliberately."
        )
    responses = load_successful_responses(cfg)
    validate_inference_complete(cfg, responses)
    hierarchy = pd.read_csv(cfg.path("label_hierarchy"), dtype=str)
    exact_lookup: dict[str, list[pd.Series]] = {}
    for _, row in hierarchy.iterrows():
        exact_lookup.setdefault(normalized_label(row["ann_level_3"]), []).append(row)

    rows: list[dict[str, Any]] = []
    unique_predictions = sorted(responses["cell_type"].dropna().astype(str).unique())
    for prediction in unique_predictions:
        normalized = normalized_label(prediction)
        candidates = exact_lookup.get(normalized, [])
        exact = candidates[0] if len(candidates) == 1 else None
        rows.append(
            {
                "prediction_id": hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12],
                "prediction_text": prediction,
                "prediction_normalized": normalized,
                "n_calls": int((responses["cell_type"] == prediction).sum()),
                "auto_match_type": "normalized exact" if exact is not None else "",
                "mapped_path_id": str(exact["path_id"]) if exact is not None else "",
                "mapped_ann_level_1": str(exact["ann_level_1"]) if exact is not None else "",
                "mapped_ann_level_2": str(exact["ann_level_2"]) if exact is not None else "",
                "mapped_ann_level_3": str(exact["ann_level_3"]) if exact is not None else "",
                "mapping_note": "",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _validated_mapping(cfg: Config) -> dict[str, tuple[str, str, str]]:
    mapping_path = cfg.path("mapping")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Create and review the blinded mapping first: {mapping_path}")
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    hierarchy = pd.read_csv(cfg.path("label_hierarchy"), dtype=str)
    by_path = hierarchy.set_index("path_id")
    valid_level_1 = set(hierarchy["ann_level_1"].astype(str))
    valid_level_2 = set(
        zip(
            hierarchy["ann_level_1"].astype(str),
            hierarchy["ann_level_2"].astype(str),
        )
    )
    unresolved = mapping.loc[mapping["mapped_path_id"].str.strip().eq(""), "prediction_text"].tolist()
    if unresolved:
        preview = "; ".join(unresolved[:8])
        raise RuntimeError(f"Prediction mapping has {len(unresolved)} unresolved rows: {preview}")
    output: dict[str, tuple[str, str, str]] = {}
    by_prediction_id: dict[str, tuple[str, str, str]] = {}
    for row in mapping.itertuples(index=False):
        path_id = str(row.mapped_path_id).strip()
        entered = (
            str(row.mapped_ann_level_1).strip(),
            str(row.mapped_ann_level_2).strip(),
            str(row.mapped_ann_level_3).strip(),
        )
        if path_id == "UNMAPPED":
            if any(entered):
                raise ValueError(
                    f"UNMAPPED prediction {row.prediction_text!r} must have blank mapped-label cells"
                )
            canonical = ("__UNMAPPED__",) * 3
        elif path_id == "LEVEL_1":
            if not entered[0] or entered[1] or entered[2]:
                raise ValueError(
                    f"LEVEL_1 prediction {row.prediction_text!r} requires only mapped_ann_level_1"
                )
            if entered[0] not in valid_level_1:
                raise ValueError(
                    f"Unknown HLCA level-1 label {entered[0]!r} for prediction {row.prediction_text!r}"
                )
            canonical = (entered[0], "__NO_LEVEL_2__", "__NO_LEVEL_3__")
        elif path_id == "LEVEL_2":
            if not entered[0] or not entered[1] or entered[2]:
                raise ValueError(
                    f"LEVEL_2 prediction {row.prediction_text!r} requires mapped_ann_level_1 and "
                    "mapped_ann_level_2, with mapped_ann_level_3 blank"
                )
            if entered[:2] not in valid_level_2:
                raise ValueError(
                    f"Unknown HLCA level-1/2 path {entered[:2]!r} for prediction "
                    f"{row.prediction_text!r}"
                )
            canonical = (entered[0], entered[1], "__NO_LEVEL_3__")
        else:
            if path_id not in by_path.index:
                raise ValueError(
                    f"Unknown mapped_path_id {path_id!r} for prediction {row.prediction_text!r}"
                )
            path = by_path.loc[path_id]
            if isinstance(path, pd.DataFrame):
                raise ValueError(f"Duplicate hierarchy path_id: {path_id}")
            canonical = (str(path.ann_level_1), str(path.ann_level_2), str(path.ann_level_3))
            if any(entered) and entered != canonical:
                raise ValueError(
                    f"Mapped labels disagree with {path_id} for prediction {row.prediction_text!r}: "
                    f"entered={entered}, canonical={canonical}"
                )

        prediction_id = str(row.prediction_id).strip()
        if prediction_id in by_prediction_id and by_prediction_id[prediction_id] != canonical:
            raise ValueError(
                f"Rows sharing prediction_id {prediction_id!r} have inconsistent mappings"
            )
        by_prediction_id[prediction_id] = canonical
        output[str(row.prediction_text)] = canonical
    return output


def score_path(
    truth: tuple[str, str, str],
    prediction: tuple[str, str, str],
    *,
    lineage_weight: float,
    state_weight: float,
    level_2_partial_credit: float,
) -> dict[str, float]:
    l1 = float(prediction[0] == truth[0])
    l2 = float(l1 == 1.0 and prediction[1] == truth[1])
    l3 = float(l2 == 1.0 and prediction[2] == truth[2])
    state = l3 if l3 else level_2_partial_credit * l2
    sanno = lineage_weight * l1 + state_weight * state
    return {
        "major_lineage_accuracy": l1,
        "level_2_accuracy": l2,
        "exact_level_3_accuracy": l3,
        "sanno_hlca": float(sanno),
    }


def bootstrap_mean_ci(values: Sequence[float], *, iterations: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_wilcoxon(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    # The cluster scores are averages of a small number of discrete call-level
    # values. Round away machine-level subtraction noise so theoretically tied
    # differences retain the same ranks before and after a CSV round trip.
    array = np.round(array, decimals=12)
    if len(array) == 0 or np.allclose(array, 0.0):
        return 1.0
    try:
        return float(
            wilcoxon(array, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
        )
    except TypeError:  # older scipy
        return float(wilcoxon(array, zero_method="wilcox", alternative="two-sided").pvalue)


def summarize_paired(cluster_scores: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    analysis = cfg.section("analysis")
    iterations = int(analysis["bootstrap_iterations"])
    seed = int(analysis["seed"])
    metrics = ["sanno_hlca", "exact_level_3_accuracy", "major_lineage_accuracy"]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        wide = cluster_scores.pivot(index="cluster_id", columns="condition", values=metric)
        wide = wide.dropna(subset=list(CONDITIONS))
        difference = wide["full_core"] - wide["standard"]
        low, high = bootstrap_mean_ci(difference, iterations=iterations, seed=seed)
        tolerance = 1e-12
        rows.append(
            {
                "metric": metric,
                "n_clusters": int(len(wide)),
                "mean_standard": float(wide["standard"].mean()),
                "mean_full_core": float(wide["full_core"].mean()),
                "mean_paired_difference": float(difference.mean()),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "wilcoxon_p_two_sided": paired_wilcoxon(difference),
                "n_improved": int((difference > tolerance).sum()),
                "n_unchanged": int((difference.abs() <= tolerance).sum()),
                "n_worsened": int((difference < -tolerance).sum()),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _set_publication_figure_style() -> None:
    """Apply the manuscript's compact, editable Matplotlib style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.0,
        }
    )


def make_figures(cluster_scores: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_publication_figure_style()
    colors = {"standard": "#5DA5DA", "full_core": "#D62728"}
    display = {
        "exact_level_3_accuracy": "Exact level 3",
        "major_lineage_accuracy": "Major lineage",
        "sanno_hlca": r"$S_{anno}$",
    }

    # Main-text panel: show the estimand directly. The prior x-y scatter was
    # dominated by ceiling values and required a paragraph of in-panel text.
    # Here, positive values unambiguously favor feature distillation.
    stats = summary.set_index("metric").loc[list(display)]
    estimates = stats["mean_paired_difference"].to_numpy(dtype=float) * 100
    low = stats["bootstrap_95ci_low"].to_numpy(dtype=float) * 100
    high = stats["bootstrap_95ci_high"].to_numpy(dtype=float) * 100
    y = np.arange(len(display))
    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - low, high - estimates]),
        fmt="D",
        markersize=6.5,
        color=colors["full_core"],
        ecolor="#333333",
        elinewidth=1.4,
        capsize=4,
        markeredgecolor="black",
        markeredgewidth=0.55,
        zorder=3,
    )
    extent = max(5.0, float(np.nanmax(np.abs(np.r_[low, high]))))
    extent = float(np.ceil(extent / 5.0) * 5.0)
    ax.set_xlim(-extent, extent)
    ax.set_yticks(y, list(display.values()))
    ax.invert_yaxis()
    ax.set_xlabel(r"Full core $-$ Standard (pp)", labelpad=8)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, cfg.path("figure_main"))
    plt.close(fig)

    # Supplementary panel: retain absolute cluster-level scores and CIs.
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    x = np.arange(len(display))
    width = 0.34
    for offset, condition in ((-width / 2, "standard"), (width / 2, "full_core")):
        means = []
        lows = []
        highs = []
        for metric in display:
            values = cluster_scores.loc[cluster_scores["condition"] == condition, metric]
            mean = float(values.mean())
            low, high = bootstrap_mean_ci(
                values,
                iterations=int(cfg.section("analysis")["bootstrap_iterations"]),
                seed=int(cfg.section("analysis")["seed"]),
            )
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        label = "Standard" if condition == "standard" else "Feature distillation (full_core)"
        ax.bar(
            x + offset,
            np.asarray(means) * 100,
            width,
            color=colors[condition],
            edgecolor="black",
            linewidth=0.6,
            label=label,
            zorder=2,
        )
        ax.errorbar(
            x + offset,
            np.asarray(means) * 100,
            yerr=np.asarray([lows, highs]) * 100,
            fmt="none",
            color="black",
            capsize=3,
            linewidth=0.9,
        )
    ax.set_xticks(x, list(display.values()))
    ax.set_ylabel("Mean cluster-level score (%)", labelpad=8)
    ax.set_ylim(0, 105)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.4,
        columnspacing=1.5,
    )
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save_figure(fig, cfg.path("figure_supp"))
    plt.close(fig)


def _qualitative_colors(n: int) -> list[Any]:
    """Return a deterministic qualitative palette for supplementary UMAPs."""

    import matplotlib.pyplot as plt

    colors: list[Any] = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(name)
        colors.extend(cmap(i) for i in range(cmap.N))
    if n > len(colors):
        colors.extend(plt.get_cmap("hsv")(i / n) for i in range(n - len(colors)))
    return colors[:n]


def make_scope_figure(cfg: Config) -> tuple[Path, Path]:
    """Render sampled-cell UMAPs showing HLCA compartment and level-3 scope.

    The embedding is visualization-only and does not enter marker generation,
    prediction mapping, or scoring. Coordinates are cached as Source Data so
    re-rendering the figure does not recompute UMAP.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    _set_publication_figure_style()
    coordinate_path = cfg.path("umap_coordinates")
    required = {"umap_1", "umap_2", "cluster_id", "ann_level_1", "ann_level_3"}
    if coordinate_path.exists():
        coordinates = pd.read_csv(coordinate_path, dtype={"cluster_id": str})
        missing = required - set(coordinates.columns)
        if missing:
            raise ValueError(f"Cached UMAP coordinates are missing columns: {sorted(missing)}")
    else:
        import scanpy as sc

        processed_path = cfg.path("processed_h5ad")
        if not processed_path.exists():
            raise FileNotFoundError(f"Run the prepare stage first: {processed_path}")
        adata = sc.read_h5ad(processed_path)
        columns = cfg.section("columns")
        cluster_col = str(columns["cluster"])
        level_1_col = str(columns["level_1"])
        level_3_col = str(columns["level_3"])
        for column in (cluster_col, level_1_col, level_3_col):
            if column not in adata.obs:
                raise ValueError(f"Processed H5AD is missing obs column: {column}")
        manifest_path = cfg.path("cluster_manifest")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cluster manifest is required for the scope figure: {manifest_path}")
        manifest = pd.read_csv(manifest_path, dtype={"cluster_id": str}).set_index("cluster_id")
        cluster_ids = adata.obs[cluster_col].astype(str)
        level_1_labels = cluster_ids.map(manifest["gt_ann_level_1"])
        level_3_labels = cluster_ids.map(manifest["gt_ann_level_3"])
        if level_1_labels.isna().any() or level_3_labels.isna().any():
            raise ValueError("Some sampled cells lack a cluster-level evaluation ground truth")
        seed = int(cfg.section("analysis")["seed"])
        use_hvg = "highly_variable" in adata.var and bool(adata.var["highly_variable"].any())
        available_features = (
            int(adata.var["highly_variable"].sum()) if use_hvg else adata.n_vars
        )
        n_comps = min(30, adata.n_obs - 1, available_features - 1)
        if n_comps < 2:
            raise ValueError("Too few cells or genes to compute the supplementary UMAP")
        sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=use_hvg, random_state=seed)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps, random_state=seed)
        sc.tl.umap(adata, min_dist=0.4, random_state=seed)
        coordinates = pd.DataFrame(
            {
                "cell_id": adata.obs_names.astype(str),
                "cluster_id": cluster_ids.to_numpy(),
                "ann_level_1": level_1_labels.astype(str).to_numpy(),
                "ann_level_3": level_3_labels.astype(str).to_numpy(),
                "umap_1": adata.obsm["X_umap"][:, 0],
                "umap_2": adata.obsm["X_umap"][:, 1],
            }
        )
        coordinate_path.parent.mkdir(parents=True, exist_ok=True)
        coordinates.to_csv(coordinate_path, index=False)

    for column in ("ann_level_1", "ann_level_3"):
        coordinates[column] = coordinates[column].map(
            lambda value: clean_label(value) or "Unassigned"
        )

    rng = np.random.default_rng(int(cfg.section("analysis")["seed"]))
    order = rng.permutation(len(coordinates))
    coordinates = coordinates.iloc[order].reset_index(drop=True)
    compartment_order = [
        label
        for label in ("Epithelial", "Endothelial", "Stroma", "Immune")
        if label in set(coordinates["ann_level_1"])
    ]
    compartment_order.extend(
        sorted(set(coordinates["ann_level_1"]) - set(compartment_order))
    )
    compartment_palette = {
        "Epithelial": "#E69F00",
        "Endothelial": "#009E73",
        "Stroma": "#CC79A7",
        "Immune": "#0072B2",
    }
    for label, color in zip(compartment_order, _qualitative_colors(len(compartment_order))):
        compartment_palette.setdefault(label, color)
    level_3_order = sorted(coordinates["ann_level_3"].dropna().astype(str).unique())
    level_3_palette = dict(zip(level_3_order, _qualitative_colors(len(level_3_order))))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    panels = (
        (axes[0], "ann_level_1", compartment_order, compartment_palette, "Major compartment"),
        (
            axes[1],
            "ann_level_3",
            level_3_order,
            level_3_palette,
            "Evaluation level-3 ground truth",
        ),
    )
    for panel_index, (ax, column, labels, palette, title) in enumerate(panels):
        point_colors = coordinates[column].astype(str).map(palette).fillna("#BDBDBD")
        ax.scatter(
            coordinates["umap_1"],
            coordinates["umap_2"],
            c=point_colors,
            s=2.2,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            -0.06,
            1.03,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
        )
        handles = [
            mlines.Line2D([], [], marker="o", linestyle="", markersize=5, color=palette[label], label=label)
            for label in labels
        ]
        if panel_index == 0:
            ax.legend(
                handles=handles,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
                fontsize=9,
                handletextpad=0.35,
                columnspacing=0.9,
            )
        else:
            ax.legend(
                handles=handles,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=2,
                fontsize=10,
                handletextpad=0.3,
                columnspacing=0.8,
            )
    fig.subplots_adjust(left=0.06, right=0.75, bottom=0.20, top=0.92, wspace=0.20)
    base = cfg.path("figure_scope")
    _save_figure(fig, base)
    plt.close(fig)
    return base.with_suffix(".pdf"), coordinate_path


def regenerate_figures(cfg: Config, *, include_scope: bool = True) -> tuple[Path, ...]:
    """Regenerate figures from frozen score tables without inference or remapping."""

    cluster_path = cfg.path("cluster_scores")
    summary_path = cfg.path("summary")
    if not cluster_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run the score stage before regenerating figures")
    cluster_scores = pd.read_csv(cluster_path, dtype={"cluster_id": str})
    summary = pd.read_csv(summary_path)
    make_figures(cluster_scores, summary, cfg)
    outputs: list[Path] = [
        cfg.path("figure_main").with_suffix(".pdf"),
        cfg.path("figure_supp").with_suffix(".pdf"),
    ]
    if include_scope:
        scope_figure, coordinate_path = make_scope_figure(cfg)
        outputs.extend([scope_figure, coordinate_path])
    return tuple(outputs)


def score_results(cfg: Config) -> tuple[Path, Path, Path]:
    responses = load_successful_responses(cfg)
    validate_inference_complete(cfg, responses)
    mapping = _validated_mapping(cfg)
    manifest = pd.read_csv(cfg.path("cluster_manifest"), dtype={"cluster_id": str})
    manifest = manifest.loc[manifest["evaluable"]].set_index("cluster_id")
    analysis = cfg.section("analysis")
    rows: list[dict[str, Any]] = []
    for row in responses.itertuples(index=False):
        truth = (
            str(manifest.at[row.cluster_id, "gt_ann_level_1"]),
            str(manifest.at[row.cluster_id, "gt_ann_level_2"]),
            str(manifest.at[row.cluster_id, "gt_ann_level_3"]),
        )
        prediction = mapping[row.cell_type]
        scores = score_path(
            truth,
            prediction,
            lineage_weight=float(analysis["sanno_lineage_weight"]),
            state_weight=float(analysis["sanno_state_weight"]),
            level_2_partial_credit=float(analysis["sanno_level_2_partial_credit"]),
        )
        rows.append(
            {
                **row._asdict(),
                "gt_ann_level_1": truth[0],
                "gt_ann_level_2": truth[1],
                "gt_ann_level_3": truth[2],
                "pred_ann_level_1": prediction[0],
                "pred_ann_level_2": prediction[1],
                "pred_ann_level_3": prediction[2],
                **scores,
            }
        )
    call_scores = pd.DataFrame(rows).sort_values(["cluster_id", "condition", "repeat"])
    metric_cols = ["major_lineage_accuracy", "level_2_accuracy", "exact_level_3_accuracy", "sanno_hlca"]
    cluster_scores = (
        call_scores.groupby(["cluster_id", "condition"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["cluster_id", "condition"])
    )
    audit_columns = [
        "n_cells_full",
        "n_cells_sampled",
        "n_donors",
        "n_source_datasets",
        "gt_ann_level_3_purity",
    ]
    cluster_scores = cluster_scores.merge(
        manifest.reset_index()[["cluster_id", *audit_columns]], on="cluster_id", how="left"
    )
    summary = summarize_paired(cluster_scores, cfg)

    call_path = cfg.path("call_scores")
    cluster_path = cfg.path("cluster_scores")
    summary_path = cfg.path("summary")
    for path in (call_path, cluster_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    call_scores.to_csv(call_path, index=False)
    cluster_scores.to_csv(cluster_path, index=False)
    summary.to_csv(summary_path, index=False)
    make_figures(cluster_scores, summary, cfg)
    write_run_metadata(cfg, responses)
    return call_path, cluster_path, summary_path


def write_run_metadata(cfg: Config, responses: pd.DataFrame) -> Path:
    from importlib.metadata import PackageNotFoundError, version

    package_versions = {}
    for name in ("llm-sc-curator", "scanpy", "anndata", "numpy", "pandas", "scipy", "google-genai"):
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = "not installed"
    metadata = {
        "completed_at_utc": utc_now(),
        "config_file": cfg.source_path.name,
        "config_sha256": sha256_file(cfg.source_path),
        "dataset": cfg.section("dataset"),
        "columns": cfg.section("columns"),
        "analysis": cfg.section("analysis"),
        "llm": cfg.section("llm"),
        "n_successful_calls": int(len(responses)),
        "returned_model_versions": sorted(
            x for x in responses["returned_model_version"].dropna().astype(str).unique() if x
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
    }
    path = cfg.path("run_metadata")
    write_json(path, metadata)
    return path


def pipeline_status(cfg: Config) -> dict[str, Any]:
    responses = load_successful_responses(cfg)
    manifest_path = cfg.path("cluster_manifest")
    clusters = 0
    expected_calls = 0
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        clusters = int(manifest["evaluable"].sum())
        expected_calls = clusters * len(CONDITIONS) * int(cfg.section("analysis")["n_repeats"])
    mapping_path = cfg.path("mapping")
    unresolved = None
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
        unresolved = int(mapping["mapped_path_id"].str.strip().eq("").sum())
    return {
        "source_h5ad": cfg.path("source_h5ad").exists(),
        "processed_h5ad": cfg.path("processed_h5ad").exists(),
        "cluster_manifest": manifest_path.exists(),
        "marker_lists": cfg.path("marker_lists").exists(),
        "evaluable_clusters": clusters,
        "successful_calls": int(len(responses)),
        "expected_calls": expected_calls,
        "mapping_exists": mapping_path.exists(),
        "mapping_unresolved_rows": unresolved,
        "scoring_complete": cfg.path("summary").exists(),
    }
