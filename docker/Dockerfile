# syntax=docker/dockerfile:1
FROM rocker/r-ver:4.5.2

ARG DEBIAN_FRONTEND=noninteractive
ARG SEURAT_VERSION=5.3.1

# -----------------------------
# 0) OS deps (R/Bioc/Seurat + build tooling)
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    build-essential patch pkg-config cmake gfortran \
    bzip2 xz-utils unzip \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libicu-dev \
    libhdf5-dev zlib1g-dev libbz2-dev liblzma-dev libzstd-dev \
    libfftw3-dev \
    libcairo2-dev libxt-dev libx11-dev libxext-dev libxrender-dev \
    libudunits2-dev libgdal-dev libgeos-dev libproj-dev \
    libglpk-dev libgmp-dev \
    libopenblas-dev liblapack-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev \
    libpng-dev libtiff-dev libjpeg-dev \
    libmagick++-dev libgit2-dev \
    libgl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

ENV RGL_USE_NULL=TRUE
ENV MPLBACKEND=Agg

RUN echo 'options(timeout = 777, repos=c(CRAN="https://cloud.r-project.org"))' \
  >> /usr/local/lib/R/etc/Rprofile.site

# -----------------------------
# 1) micromamba (safe install)
# -----------------------------
ENV MAMBA_ROOT_PREFIX=/opt/conda

RUN ARCH="$(uname -m)" && \
    case "$ARCH" in \
      aarch64|arm64) MAMBA_PLATFORM="linux-aarch64" ;; \
      *)             MAMBA_PLATFORM="linux-64" ;; \
    esac && \
    mkdir -p /tmp/micromamba && \
    cd /tmp/micromamba && \
    curl -L "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" | tar -xj && \
    install -m 0755 ./bin/micromamba /usr/local/bin/micromamba && \
    cd / && rm -rf /tmp/micromamba

RUN micromamba create -y -n py -c conda-forge \
      python=3.12 pip \
      jupyterlab ipykernel \
      numpy pandas scipy \
      scanpy anndata h5py \
      python-igraph leidenalg \
      matplotlib \
      scikit-learn \
      scikit-misc \
      adjusttext \
    && micromamba clean -a -y

ENV RETICULATE_PYTHON=/opt/conda/envs/py/bin/python

RUN micromamba run -n py python -m ipykernel install --sys-prefix \
    --name python3 --display-name "Python (LLM-scCurator)"

# -----------------------------
# 2) R packages (explicit ordering; no deps=TRUE traps)
# -----------------------------
RUN /usr/local/bin/R --vanilla <<'EOF'
options(repos = c(CRAN="https://cloud.r-project.org"))

# Core CRAN utilities (keep lean)
install.packages(c(
  "remotes",
  "reticulate",
  "dplyr","tidyr",
  "data.table",
  "ggplot2","patchwork",
  "pheatmap",
  "hdf5r",
  "Matrix","igraph",
  "zip",
  "IRkernel"
))

# Bioconductor (pin to 3.22 for R 4.5.x)
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(version="3.22", ask=FALSE)

# Install Bioc deps needed by downstream CRAN packages FIRST
BiocManager::install(c(
  "BiocGenerics","Biobase",
  "multtest",
  "qvalue",
  "SingleCellExperiment","SummarizedExperiment",
  "SingleR","celldex",
  "zellkonverter",
  "scran","scater","scuttle",
  "DropletUtils","BiocParallel",
  "SpatialExperiment","SpatialExperimentIO"
), ask=FALSE, update=FALSE)

# Now CRAN packages that depend on Bioc (mutoss needs multtest; metap needs mutoss+qqconf+qvalue)
install.packages(c("qqconf","mutoss","metap"))

q(save="no")
EOF

# Seurat: install WITHOUT Suggests (avoids BPCells/presto/monocle/etc whack-a-mole)
RUN /usr/local/bin/R -q -e ' \
  options(repos=c(CRAN="https://cloud.r-project.org")); \
  if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes"); \
  remotes::install_version("Seurat", version=Sys.getenv("SEURAT_VERSION", unset="5.3.1"), dependencies=NA, repos="https://cloud.r-project.org")' \
  && true

# IRkernel: register into the conda prefix so Jupyter sees it
RUN PATH=/opt/conda/envs/py/bin:$PATH /usr/local/bin/R -q -e \
  'IRkernel::installspec(user=FALSE, prefix="/opt/conda/envs/py")'

# -----------------------------
# 3) Python LLM SDKs + your package
# -----------------------------
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV R_HOME=/usr/local/lib/R

RUN micromamba run -n py python -m pip install -U \
      openai \
      google-genai \
      google-generativeai

WORKDIR /opt/LLM-scCurator
COPY . /opt/LLM-scCurator
RUN micromamba run -n py python -m pip install -e ".[all]"

# -----------------------------
# 4) Sanity checks
# -----------------------------
RUN /usr/local/bin/R -q -e ' \
  library(reticulate); \
  library(scran); library(scater); library(scuttle); library(SingleR); \
  library(SpatialExperiment); library(SpatialExperimentIO); \
  TRUE' \
 && micromamba run -n py python -c \
 "import scanpy as sc; import pandas as pd; import scipy; import igraph, leidenalg; import adjustText; import skmisc; from openai import OpenAI; from google import genai; print('OK')"

ENV PATH=/opt/conda/envs/py/bin:$PATH

# -----------------------------
# 5) Jupyter
# -----------------------------
WORKDIR /work
EXPOSE 8888
CMD ["/bin/sh","-lc","micromamba run -n py jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
