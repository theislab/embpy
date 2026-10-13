# Getting Started

## Installation

`embpy` supports Python ≥ 3.11 on Linux, macOS, and Windows. Depending on your platform and available accelerators, select the appropriate installation profile:

=== "Linux (CPU)"

    ```bash
    # Install PyTorch CPU wheels first to avoid bulky CUDA runtime downloads
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "embpy[cpu]"
    ```

=== "Linux (NVIDIA GPU / CUDA)"

    ```bash
    # Ensure CUDA toolkit or compatible drivers are available
    pip install "embpy[gpu]"
    ```

=== "macOS (Apple Silicon / MPS)"

    ```bash
    # PyTorch automatically supports Metal Performance Shaders (MPS)
    pip install "embpy[cpu]"
    ```

=== "Windows (CPU)"

    ```powershell
    # Windows PowerShell installation
    py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    py -3.12 -m pip install "embpy[cpu]"
    ```

=== "Windows (NVIDIA GPU)"

    ```powershell
    # Windows PowerShell installation with CUDA
    py -3.12 -m pip install "embpy[gpu]"
    ```

### Modular Extras

`embpy` isolates heavy dependencies into pip extras:

| Extra | Description | Example Command |
|---|---|---|
| `[cpu]` | Standard CPU setup with PyTorch and Transformers | `pip install "embpy[cpu]"` |
| `[gpu]` | CUDA-accelerated embedding backends | `pip install "embpy[gpu]"` |
| `[models]` | Full transformer and graph neural network wrappers | `pip install "embpy[models]"` |
| `[scib]` | Single-cell integration benchmarking (`scIB`) | `pip install "embpy[models,scib]"` |
| `[all]` | Full suite of pip-installable capabilities | `pip install "embpy[all]"` |

---

## Quickstart Examples

### Minimal Protein Embedding

Embed protein sequences with ESM-2 in just a few lines:

```python
import torch
from embpy import BioEmbedder

# Automatically select accelerator (CUDA, MPS, or CPU)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the embedder
embedder = BioEmbedder(device=device)

# Protein sequences (e.g., insulin chains)
sequences = [
    "GIVEQCCTSICSLYQLENYCN",
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
]

# Compute embeddings as a NumPy array
embeddings = embedder.embed(
    sequences,
    entity_type="protein",
    model="esm2_8M",
    output="numpy",
)

print(f"Embedding shape: {embeddings.shape}")
```

### Attaching Embeddings to AnnData

In scverse single-cell and genomics workflows, attach embeddings directly to an AnnData object:

```python
import anndata as ad
import numpy as np
from embpy import BioEmbedder

# Create sample AnnData object with proteins in var
adata = ad.AnnData(
    X=np.zeros((5, 2)),
    var={"protein_sequence": [
        "GIVEQCCTSICSLYQLENYCN",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    ]},
)

embedder = BioEmbedder(device="cpu")
adata = embedder.embed(
    adata,
    entity_type="protein",
    var_column="protein_sequence",
    model="esm2_8M",
    output="anndata",
    key="X_esm2",
)

print("Stored representation:", adata.varm["X_esm2"].shape)
```
