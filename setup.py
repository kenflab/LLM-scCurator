# LLM-scCurator/setup.py
from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="llm-sc-curator",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        "scanpy>=1.11.5",
        "pandas>=2.3.3",
        "numpy>=2.0.2",
        "scipy>=1.16.3",
        "python-igraph>=1.0.0",
        "leidenalg>=0.11.0",
    ],
    extras_require={
        # LLM backend integrations (optional)
        "openai": [
            "openai>=1.0.0",
        ],
        "gemini": [
            "google-generativeai>=0.8.5",
        ],
        # Convenience group: everything
        "all": [
            "openai>=1.0.0",
            "google-generativeai>=0.8.5",
        ],
    },
    python_requires=">=3.10",
    author="Ken Furudate",
    description="Dynamic feature distillation framework for robust zero-shot LLM annotation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kenflab/LLM-scCurator",
)
