# embpy: Universal Biological Embeddings

![embpy architecture](embpy_architecture.svg)

**`embpy`** is a unified Python package for creating, managing, comparing, and benchmarking multi-modal biological embeddings. It provides a standardized interface to over 100 state-of-the-art embedding models across proteins, genes, cells, molecules, DNA, and natural language.

---

## Core Capabilities

- **Universal Interface (`BioEmbedder`):** A single class replaces dozens of disconnected model pipelines with unified `.embed()` calls.
- **Multi-Modal Foundation Models:** Embed proteins (ESM-2, ProtT5, ESM-3), DNA (Evo, Borzoi), molecules (ChemBERTa, MolFormer), and cells (scGPT, Geneformer).
- **Standardized Output Contracts:** Seamless export to `anndata` (for scverse ecosystems), pandas DataFrames (`table`), and raw dictionary payloads (`payload`).
- **Hardware Scalability:** Runs smoothly on laptops (CPU / Apple Silicon MPS) and automatically scales to multi-GPU NVIDIA clusters.

---

## Modern Packaging & Cross-Platform Features

`embpy` brings key structural and usability improvements:

### 1. Pure `pip` & `uv` Compatibility
- **Lightweight Core:** A bare `pip install embpy` installs only essential libraries (`numpy`, `pandas`, `anndata`, `scikit-learn`, `rdkit`) without requiring a C++ compiler or CUDA drivers.
- **Modular Dependency Extras:** Heavy deep learning stacks are isolated into clean optional extras (`[models]`, `[cpu]`, `[gpu]`, `[scib]`, `[all]`).
- **Elimination of Git Dependencies:** Replaced non-standard git repository dependencies (e.g., `scooby`) with pure PyPI distributions to ensure reproducibility.

### 2. Cross-Platform POSIX Gating
- **Native Windows Support:** POSIX-only C extensions (such as `pysam` and `pyensembl`) are automatically gated on Windows. Core biological embedding workflows function natively on Windows without build errors or WSL.
- **Multiprocessing Safety:** Windows multiprocessing defaults to `spawn` rather than `fork`. `embpy` ensures transparent process isolation when scaling batch operations across CPU cores.

### 3. Decoupled Architecture & Lazy Imports
- Heavy neural network weights and dependencies (`torch`, `transformers`, `torch_geometric`) are lazily loaded only when a specific model wrapper is invoked. Importing `embpy` completes in milliseconds.

---

## Next Steps

- Explore the [Getting Started Guide](getting-started.md) for quick installation and minimal examples.
- Work through practical tutorials in the [Hands-On Workflows & Recipes Guide](hands-on.md).
- Understand OS and hardware specifics in the [Cross-Platform & Hardware Guide](cross-platform.md).
- Dive into the [Architecture Overview](technical.md) for architectural details.
- Review the [API Reference](api-reference.md) for automated module documentation.
- Check the [Contributing Guide](contributing.md) to join the development workflow.
