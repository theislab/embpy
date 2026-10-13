# MCP Tools Reference

This reference documents the 7 bioinformatics tools registered in the `embpy` Model Context Protocol (MCP) server. Each tool can be invoked programmatically via standard MCP client libraries or autonomously by LLM agents.

---

## 1. `inspect_anndata`

Inspects an AnnData (`.h5ad`) file metadata and summarizes high-level dataset statistics without loading the full dense count matrix into RAM.

### Function Signature
```python
inspect_anndata(path: str, max_categories: int = 10) -> str
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | *required* | Filesystem path to the `.h5ad` file to inspect. |
| `max_categories` | `int` | `10` | Maximum number of category levels to list in `.obs` categorical column summaries. |

### Return Value
Returns a JSON-serialized dictionary containing:
- `file_path`: Absolute path to the inspected file.
- `file_size_bytes`: Size of the `.h5ad` file on disk.
- `n_obs`: Number of observations (cells).
- `n_vars`: Number of variables (genes).
- `obs_columns`: List of column names in `adata.obs`.
- `obs_categories`: Dictionary mapping categorical columns to category lists and value frequencies.
- `var_preview`: First 10 gene identifiers in `adata.var_names`.
- `gene_id_type_hint`: Inferred gene ID type (`"ensembl"` if starting with `ENSG`, or `"symbol"`).
- `layers`: List of data layers present (e.g., `["counts", "spliced"]`).
- `obsm_keys`: List of precomputed embedding matrices in `adata.obsm` (e.g., `["X_pca", "X_scvi"]`).
- `is_raw_counts`: Boolean heuristic indicating whether `.X` or the detected counts layer contains non-negative integer counts.
- `summary`: Human-readable summary string of dataset characteristics.

### Example Tool Call
```json
{
  "name": "inspect_anndata",
  "arguments": {
    "path": "data/pbmc3k.h5ad",
    "max_categories": 5
  }
}
```

---

## 2. `list_available_models`

Queries the local `embpy` installation and underlying runtime hardware to report which single-cell foundation models, compute devices, and optional dependencies are available.

### Function Signature
```python
list_available_models() -> str
```

### Parameters
*None.*

### Return Value
Returns a JSON-serialized dictionary containing:
- `available_models`: List of registered foundation model keys (e.g., `["pca", "scvi", "scgpt", "geneformer_v2_12L", "state"]`).
- `default_device`: Auto-detected primary compute device (`"cuda"`, `"mps"`, or `"cpu"`).
- `device_diagnostics`:
  - `cuda_available`: Boolean indicating NVIDIA CUDA GPU availability.
  - `cuda_device_count`: Number of detected CUDA devices.
  - `mps_available`: Boolean indicating Apple Silicon MPS availability.
  - `cpu_count`: Logical CPU cores detected.
- `optional_extras`:
  - `embpy[seqmodels]`: Status of sequence model dependencies (`helical`, etc.).
  - `embpy[genome]`: Status of genomic reader backends (`pyensembl`, `pysam`).
  - `embpy[eval]`: Status of evaluation suites (`cell-eval`, `scib`).
  - `embpy[scib]`: Status of `scib` package availability.
- `recommendations`: Actionable installation suggestions if optional extras are missing.

### Example Tool Call
```json
{
  "name": "list_available_models",
  "arguments": {}
}
```

---

## 3. `map_gene_identifiers`

Validates and converts gene identifiers between Ensembl IDs (e.g., `ENSG00000141510`) and HGNC symbols (e.g., `TP53`) using `embpy.GeneResolver`, preventing vocabulary mismatch errors in foundation models like Geneformer or scGPT.

### Function Signature
```python
map_gene_identifiers(
    identifiers: list[str] | str,
    source_id_type: str = "auto",
    target_id_type: str = "auto",
    organism: str = "human",
) -> str
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `identifiers` | `list[str]` \| `str` | *required* | Gene identifiers to convert (either a Python list of strings or comma-separated string). |
| `source_id_type` | `str` | `"auto"` | Source format (`"auto"`, `"ensembl"`, or `"symbol"`). When `"auto"`, inferred from prefix. |
| `target_id_type` | `str` | `"auto"` | Target format (`"auto"`, `"symbol"`, or `"ensembl"`). If `"auto"`, converts to the opposite format. |
| `organism` | `str` | `"human"` | Organism taxonomy (`"human"` or `"mouse"`). |

### Return Value
Returns a JSON-serialized dictionary containing:
- `organism`: Target organism.
- `source_id_type`: Resolved source format.
- `target_id_type`: Resolved target format.
- `total_submitted`: Total count of query identifiers.
- `mapped_count`: Number of successfully resolved identifiers.
- `unmapped_count`: Number of unrecognized identifiers.
- `mapping`: Dictionary mapping input identifiers to converted targets.
- `unmapped_identifiers`: List of query identifiers that could not be resolved.
- `summary`: Diagnostic summary string.

### Example Tool Call
```json
{
  "name": "map_gene_identifiers",
  "arguments": {
    "identifiers": ["ENSG00000141510", "ENSG00000146648", "ENSG00000136997"],
    "source_id_type": "ensembl",
    "target_id_type": "symbol"
  }
}
```

---

## 4. `embed_cells`

Executes `BioEmbedder.embed_cells()` to compute cellular latent representations across one or more model backends.

### Function Signature
```python
embed_cells(
    path: str,
    models: list[str] | str = "pca",
    batch_size: int = 32,
    output_key: str | None = None,
    output_path: str | None = None,
    preprocessing: str = "auto",
    device: str = "auto",
) -> str
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | *required* | Path to the target single-cell `.h5ad` dataset. |
| `models` | `list[str]` \| `str` | `"pca"` | Model key or list of keys to run (`"pca"`, `"scvi"`, `"scgpt"`, `"geneformer_v2_12L"`, `"state"`). |
| `batch_size` | `int` | `32` | Inference batch size for deep learning models. |
| `output_key` | `str` | `None` | Custom `.obsm` key name (applicable when running a single model). |
| `output_path` | `str` | `None` | Filesystem path to save the modified AnnData containing the new embeddings. |
| `preprocessing` | `str` | `"auto"` | Preprocessing pipeline (`"auto"`, `"standard"`, `"raw"`, `"none"`). |
| `device` | `str` | `"auto"` | Compute device override (`"auto"`, `"cpu"`, `"cuda"`, `"mps"`). |

### Return Value
Returns a JSON-serialized dictionary containing:
- `n_cells`: Total cells embedded.
- `models_computed`: List of model keys successfully computed.
- `embedding_keys`: List of `.obsm` keys populated.
- `shapes`: Dictionary mapping each embedding key to its coordinate dimensions (e.g., `{"X_pca": [500, 50]}`).
- `output_path`: Filesystem destination where the updated AnnData was saved.
- `execution_time_seconds`: Wall-clock execution time.

### Example Tool Call
```json
{
  "name": "embed_cells",
  "arguments": {
    "path": "data/pbmc3k.h5ad",
    "models": ["pca", "scvi"],
    "output_path": "data/pbmc3k_embedded.h5ad",
    "device": "cpu"
  }
}
```

---

## 5. `evaluate_scib`

Wraps `embpy.tl.compute_scib_metrics` to benchmark embedding biological conservation (NMI, ARI, ASW label, cLISI) and batch correction (ASW batch, graph connectivity) across precomputed embeddings in `.obsm`.

### Function Signature
```python
evaluate_scib(
    path: str,
    embedding_keys: list[str] | str,
    label_key: str,
    batch_key: str | None = None,
    output_csv: str | None = None,
) -> str
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | *required* | Path to the `.h5ad` file containing computed embeddings in `.obsm`. |
| `embedding_keys` | `list[str]` \| `str` | *required* | One or more `.obsm` keys to benchmark (e.g., `["X_pca", "X_scvi"]`). |
| `label_key` | `str` | *required* | Column in `adata.obs` holding biological ground-truth labels (e.g., `cell_type`). |
| `batch_key` | `str` | `None` | Column in `adata.obs` holding batch annotations. |
| `output_csv` | `str` | `None` | Optional destination path to export the resulting metrics table. |

### Return Value
Returns a structured response containing:
- **Markdown Table:** Formatted ranking comparison table showing total composite scores and individual bio-conservation metrics.
- **Structured JSON Data:**
  - `evaluated_models`: List of evaluated embedding keys.
  - `label_key`: Biological ground-truth column used.
  - `batch_key`: Batch column used.
  - `metrics`: Dictionary of metric dictionaries for each model (NMI, ARI, ASW, cLISI).
  - `rankings`: List of models ranked by composite score.
  - `output_csv`: Path to the saved CSV file, if requested.

### Example Tool Call
```json
{
  "name": "evaluate_scib",
  "arguments": {
    "path": "data/pbmc3k_embedded.h5ad",
    "embedding_keys": ["X_pca", "X_scvi"],
    "label_key": "cell_type"
  }
}
```

---

## 6. `evaluate_perturbations`

Wraps `embpy.tl.run_cell_eval` to score perturbation responses across latent embedding spaces using standard single-cell perturbation evaluation benchmarks.

### Function Signature
```python
evaluate_perturbations(
    path: str,
    control_pert: str,
    pert_col: str = "perturbation",
    split_col: str = "split",
    embedding_keys: list[str] | str = "X_pca",
    output_dir: str | None = None,
) -> str
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | *required* | Path to the `.h5ad` file containing perturbed single-cell data. |
| `control_pert` | `str` | *required* | Identifier corresponding to the unperturbed/control condition (e.g., `"control"` or `"DMSO"`). |
| `pert_col` | `str` | `"perturbation"` | Column in `adata.obs` containing perturbation labels. |
| `split_col` | `str` | `"split"` | Column in `adata.obs` indicating data splits (must include `"predicted"` and `"real"`). |
| `embedding_keys` | `list[str]` \| `str` | `"X_pca"` | Embedding keys in `.obsm` to evaluate. |
| `output_dir` | `str` | `None` | Directory path to store detailed `cell-eval` artifacts and reports. |

### Return Value
Returns a JSON-serialized dictionary and markdown table detailing:
- `evaluated_keys`: Embedding keys evaluated.
- `control_pert`: Control condition identifier.
- `perturbation_count`: Number of distinct perturbation conditions evaluated.
- `summary_metrics`: Summary metric dictionary (R2, MSE, correlation, directionality).
- `output_dir`: Path to exported evaluation logs.

### Example Tool Call
```json
{
  "name": "evaluate_perturbations",
  "arguments": {
    "path": "data/perturbation_screen.h5ad",
    "control_pert": "control",
    "pert_col": "perturbation",
    "embedding_keys": ["X_pca"]
  }
}
```

---

## 7. `compute_umap`

Computes a k-nearest neighbor graph and 2D UMAP coordinates on a selected latent embedding space, storing the results in `.obsm['X_umap_<model>']` for visual inspection.

### Function Signature
```python
compute_umap(
    path: str,
    embedding_key: str = "X_pca",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    output_key: str | None = None,
    output_path: str | None = None,
) -> str
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | *required* | Path to the `.h5ad` dataset containing the source latent embedding. |
| `embedding_key` | `str` | `"X_pca"` | Source embedding key in `.obsm` to construct neighbor graph upon. |
| `n_neighbors` | `int` | `15` | Number of nearest neighbors to compute in latent space. |
| `min_dist` | `float` | `0.5` | Effective minimum distance between embedded points in 2D space. |
| `output_key` | `str` | `None` | Destination `.obsm` key (defaults to `X_umap_<cleaned_model_name>`). |
| `output_path` | `str` | `None` | Filesystem destination to write the updated `.h5ad` file. |

### Return Value
Returns a JSON-serialized dictionary containing:
- `embedding_key`: Source embedding used.
- `umap_key`: Generated 2D coordinate key in `.obsm`.
- `n_cells`: Total cells projected.
- `umap_shape`: Coordinate matrix shape `[n_cells, 2]`.
- `output_path`: Filesystem location of saved AnnData.

### Example Tool Call
```json
{
  "name": "compute_umap",
  "arguments": {
    "path": "data/pbmc3k_embedded.h5ad",
    "embedding_key": "X_pca",
    "output_key": "X_umap_pca",
    "output_path": "data/pbmc3k_umap.h5ad"
  }
}
```

---

## 8. Resilience & Error Diagnostics Schema

Every tool in the `embpy` MCP suite catches unhandled runtime exceptions (e.g. missing files, missing categorical columns, unsupported devices, or out-of-memory errors) and translates them into structured JSON error payloads:

```json
{
  "error": true,
  "tool": "evaluate_scib",
  "error_type": "KeyError",
  "message": "Column 'cell_type' not found in adata.obs. Available columns: ['sample_id', 'batch', 'donor']",
  "suggestion": "Inspect dataset with inspect_anndata to identify existing observation columns."
}
```

Because errors are delivered within the JSON-RPC message framing rather than aborting Python execution, your AI assistant can parse the diagnostic error, inspect the problem, and dynamically adjust its parameters without interrupting your session.
