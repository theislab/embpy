# Architecture & Technical Design

![embpy architecture](embpy_architecture.svg)

`embpy` relies on a highly modular architecture that isolates heavy dependencies, ensuring that the core package remains lightweight, responsive, and cross-platform.

---

## The Embedder Protocol

At the heart of `embpy` is the `BioEmbedder`. It acts as a unified factory, instantiating and dispatching specific model wrappers based on the requested entity type and model name.

All underlying embedding wrappers inherit from the `BaseModelWrapper` interface:

```python
class BaseModelWrapper:
    def load(self, device: torch.device):
        """Load tokenizer, weights, and move model to device."""
        pass
        
    def embed(self, sequences: list[str]) -> np.ndarray:
        """Execute forward pass and pool representations."""
        pass
```

### Lazy Imports
Wrappers perform lazy imports of heavy frameworks (`transformers`, `torch_geometric`, `rdkit`, etc.) inside their respective `.load()` methods. This allows users to import `embpy` in fractions of a second without forcing large deep learning dependencies into process memory until needed.

---

## Standardized Output Contracts

The `.embed()` method formats generated representations according to user specifications:

- `output="numpy"` / `output="tensor"`: Returns numerical matrices for standard data science workflows.
- `output="anndata"`: Automatically aligns representations with `.obsm` (for observation/cell/molecule embeddings) or `.varm` (for feature/protein/gene embeddings), preserving metadata and dimensions.
- `output="table"`: Returns a pandas DataFrame with canonical index identifiers.
- `output="payload"`: Returns raw dictionaries with metadata, aliases, and provenance for downstream pipeline integration.

---

## Adding a New Model

To integrate a new embedding model into `embpy`:
1. Subclass `BaseModelWrapper` in the appropriate modality module (`src/embpy/models/`).
2. Implement `.load(device)` using lazy imports for external libraries.
3. Implement `.embed(inputs)` returning pooled NumPy arrays or PyTorch tensors.
4. Register the model string identifier in the modality registry.
