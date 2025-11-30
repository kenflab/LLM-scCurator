from setuptools import setup, find_packages

setup(
    name="llm-sc-curator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scanpy",
        "pandas",
        "numpy",
        "scipy",
        "igraph", 
        "louvain",
        "google-generativeai",
        "matplotlib",
        "seaborn",
        "adjustText"
    ],
    author="Ken Furudate,
    author_email="KFurudate@mdanderson.org",
    description="A noise-aware feature selection framework for robust LLM-based single-cell annotation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LLM-scCurator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.9',
)
