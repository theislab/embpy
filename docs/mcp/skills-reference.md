# Universal MCP Skills Reference

This document provides a comprehensive technical reference for the 9 core bioinformatics tools registered on the `embpy` Model Context Protocol (MCP) server.

All tools adhere to strict **Token-Saving Guardrails**:
- Large arrays stay on disk in `.h5ad` AnnData files or memory caches.
- Tools return compact JSON summaries with array shapes, counts, and Markdown tables.
- Exceptions return descriptive diagnostic strings to enable autonomous self-correction by the AI agent.

---

## Tool Summary Table

| Tool Name | Category | Primary Function | Token-Saving Return |
| :--- | :--- | :--- | :--- |
| [`inspect_data`](#1-inspect_data) | Preprocessing | Inspects `.h5ad` structure, layers, and candidate keys | Shape `(n_obs, n_vars)`, candidate keys, layer names |
| [`resolve_identifiers`](#2-resolve_identifiers) | Preprocessing | Harmonizes IDs (Ensembl, HGNC, SMILES) | Mapped percentage, sample pairs |
| [`preprocess_counts`](#3-preprocess_counts) | Preprocessing | Normalizes (`normalize_total`, `log1p`), selects HVGs | Output path, cell/gene counts, HVG count |
| [`list_model_registry`](#4-list_model_registry) | BioEmbedder | Catalogs ~150 foundation models and hardware | Hardware device, model rosters by modality |
| [`compute_embeddings`](#5-compute_embeddings) | BioEmbedder | Extracts multi-model latent representations | Output path, embedding keys, array shapes |
| [`annotate_metadata`](#6-annotate_metadata) | Annotation | Enriches `.obs`/`.uns` with GO, pathways, disease data | Added column names, entity count |
| [`compute_umap`](#7-compute_umap) | Analysis | Computes kNN graph and 2D UMAP coordinates | Coordinate array shape `(n_obs, 2)`, umap key |
| [`benchmark_scib`](#8-benchmark_scib) | Benchmarking | Evaluates bio-conservation & batch integration | Formatted Markdown table of NMI, ARI, ASW, cLISI |
| [`benchmark_perturbations`](#9-benchmark_perturbations) | Benchmarking | Scores differential perturbation predictions | Formatted Markdown table of cell-eval metrics |

---

## 1. `inspect_data`

Profiles an AnnData `.h5ad` file or a list of items to inspect metadata, observation columns, embeddings, layers, and count distributions.

### Parameters
- `path` (*str*, required): Filesystem path to the `.h5ad` AnnData dataset.
- `max_categories` (*int*, default=10): Maximum categorical levels to list per column.
- `sample_peek` (*int*, default=5): Number of sample observation and feature names to return.

### Response Schema
```json
{
  "status": "success",
  "path": "data/pbmc3k.h5ad",
  "n_cells": 2700,
  "n_genes": 32738,
  "is_log_normalized": false,
  "obs_columns": ["n_genes", "percent_mito", "cell_type", "batch"],
  "candidate_label_columns": ["cell_type"],
  "candidate_batch_columns": ["batch"],
  "obsm_keys": ["X_pca"],
  "layers": ["counts"],
  "sample_genes": ["CD3D", "MS4A1", "GNLY", "CD14", "FCER1A"]
}
```

---

## 2. `resolve_identifiers`

Harmonizes biological entity vocabularies (Ensembl IDs <-> HGNC symbols, molecule names <-> SMILES, UniProt <-> proteins) to prevent vocabulary mismatches in downstream models.

### Parameters
- `identifiers` (*list[str] | str*, optional): List or comma-separated string of identifiers to map.
- `path` (*str*, optional): Path to AnnData `.h5ad` whose `var_names` or `.obs` column to resolve.
- `entity_type` (*str*, default="auto"): Entity class (`"gene"`, `"molecule"`, `"protein"`, `"cell_line"`).
- `target_id_type` (*str*, default="auto"): Target ID system (`"ensembl"`, `"symbol"`, `"smiles"`).
- `organism` (*str*, default="human"): Species for resolution (`"human"`, `"mouse"`).
- `update_h5ad_path` (*str*, optional): If provided, writes the updated AnnData with mapped IDs to disk.

### Response Schema
```json
{
  "status": "success",
  "entity_type": "gene",
  "total_submitted": 2000,
  "mapped_count": 1984,
  "mapping_success_rate": 99.2,
  "sample_mapping": {
    "TP53": "ENSG00000141510",
    "GAPDH": "ENSG00000111640"
  },
  "output_path": "/data/pbmc_resolved.h5ad"
}
```

---

## 3. `preprocess_counts`

Runs standard single-cell quality control, library size normalization (`sc.pp.normalize_total`), log1p transformation (`sc.pp.log1p`), and highly variable gene (HVG) selection.

### Parameters
- `path` (*str*, required): Path to the raw `.h5ad` AnnData dataset.
- `output_path` (*str*, optional): Destination path for the preprocessed dataset.
- `min_genes` (*int*, default=200): Minimum genes per cell.
- `min_cells` (*int*, default=3): Minimum cells expressing a gene.
- `target_sum` (*float*, default=1e4): Target library count per cell.
- `n_top_genes` (*int*, default=2000): Number of highly variable genes to select.
- `select_hvg` (*bool*, default=true): Whether to compute HVGs.
- `log_transform` (*bool*, default=true): Whether to apply `sc.pp.log1p`.

### Response Schema
```json
{
  "status": "success",
  "input_path": "/data/raw.h5ad",
  "output_path": "/data/preprocessed.h5ad",
  "n_cells": 2650,
  "n_genes": 18200,
  "n_hvg": 2000,
  "layers": ["counts", "log_normalized"],
  "summary": "Successfully preprocessed 2650 cells and 18200 genes."
}
```

---

## 4. `list_model_registry`

Reports available biological foundation models (~150), supported modalities, and active hardware acceleration.

### Parameters
- `entity_type` (*str*, optional): Modality filter (`"cell"`, `"gene"`, `"protein"`, `"molecule"`).

### Response Schema
```json
{
  "status": "success",
  "total_models_available": 150,
  "compute_device": "cuda",
  "device_details": "NVIDIA GPU (Tesla V100-SXM2-16GB)",
  "supported_entities": ["cell", "gene", "protein", "molecule", "morphology", "text"],
  "runnable_models": ["pca", "scvi", "scanvi", "scgpt"],
  "recommended_models": {
    "cell": ["pca", "scvi", "scgpt", "geneformer_v2_12L", "state"]
  }
}
```

---

## 5. `compute_embeddings`

Wraps `BioEmbedder.embed()` to extract multi-model latent representations and stores them directly in `.obsm` (or `.varm` / `.uns`) on disk.

### Parameters
- `path` (*str*, required): Path to the preprocessed `.h5ad` dataset.
- `models` (*list[str]*, required): Model roster to compute (e.g., `["pca", "scvi"]`).
- `output_path` (*str*, optional): Filesystem path to save the embedded dataset.
- `entity_type` (*str*, default="cell"): Modality (`"cell"`, `"gene"`, `"protein"`, `"molecule"`).
- `batch_size` (*int*, default=64): Inference batch size.
- `device` (*str*, default="auto"): Compute device (`"auto"`, `"cuda"`, `"mps"`, `"cpu"`).
- `preprocessing` (*str*, default="auto"): Preprocessing profile (`"auto"`, `"standard"`, `"none"`).

### Response Schema
```json
{
  "status": "success",
  "output_path": "/data/pbmc_embedded.h5ad",
  "n_cells": 2650,
  "models_computed": ["pca", "scvi"],
  "embedding_keys": ["X_pca", "X_scvi"],
  "embedding_shapes": {
    "X_pca": [2650, 50],
    "X_scvi": [2650, 30]
  },
  "execution_time_seconds": 12.4
}
```

---

## 6. `annotate_metadata`

Enriches `.obs` or `.uns` with external biological annotations (GO terms, pathways, disease mappings, pharmacology) based on entities present.

### Parameters
- `path` (*str*, required): Path to the AnnData `.h5ad` dataset.
- `annotation_type` (*str*, default="pathway"): Annotation type (`"pathway"`, `"go"`, `"disease"`, `"moa"`).
- `entity_type` (*str*, default="auto"): Entity type (`"gene"`, `"protein"`, `"molecule"`, `"cell_line"`).
- `column` (*str*, optional): Target column to use as query.
- `output_path` (*str*, optional): Destination path to save the annotated dataset.

### Response Schema
```json
{
  "status": "success",
  "output_path": "/data/pbmc_annotated.h5ad",
  "entity_type": "gene",
  "annotation_type": "pathway",
  "target_column": "perturbation",
  "new_columns_added": ["pathway_name", "go_biological_process"],
  "summary": "Annotated gene entities using column 'perturbation'. Added 2 new metadata features to .obs."
}
```

---

## 7. `compute_umap`

Computes kNN neighborhood graphs and 2D UMAP coordinates over a selected representation in `.obsm`.

### Parameters
- `path` (*str*, required): Path to the embedded `.h5ad` AnnData dataset.
- `embedding_key` (*str*, default="X_pca"): Source embedding representation in `.obsm`.
- `output_key` (*str*, optional): Custom `.obsm` key (defaults to `X_umap_<model>`).
- `output_path` (*str*, optional): Filesystem path to write updated AnnData.
- `n_neighbors` (*int*, default=15): Number of neighbors for graph construction.
- `min_dist` (*float*, default=0.5): Minimum distance parameter for UMAP.

### Response Schema
```json
{
  "status": "success",
  "embedding_key": "X_pca",
  "umap_key": "X_umap_pca",
  "n_cells": 2650,
  "umap_shape": [2650, 2],
  "output_path": "/data/pbmc_umap.h5ad"
}
```

---

## 8. `benchmark_scib`

Wraps `tl.compute_scib_metrics()` to score embeddings on biological conservation (NMI, ARI, ASW label) and batch correction (ASW batch, graph connectivity, cLISI). Returns a concise Markdown comparison table.

### Parameters
- `path` (*str*, required): Path to the `.h5ad` dataset containing embeddings in `.obsm`.
- `embedding_keys` (*list[str] | str*, required): List of representation keys to benchmark.
- `label_key` (*str*, required): Column in `.obs` holding cell type / cluster labels.
- `batch_key` (*str*, optional): Column in `.obs` holding technical batch annotations.
- `output_csv` (*str*, optional): Path to export metric results as CSV.

### Response Schema
```json
{
  "status": "success",
  "embedding_keys": ["X_pca", "X_scvi"],
  "rankings": ["X_scvi", "X_pca"],
  "markdown_table": "| | NMI | ARI | ASW_label | ASW_batch | Bio_Conservation | Total |\n|---|---|---|---|---|---|---|\n| X_scvi | 0.7421 | 0.6812 | 0.8123 | 0.7954 | 0.7452 | 0.7703 |\n| X_pca  | 0.6830 | 0.6120 | 0.7450 | 0.6210 | 0.6800 | 0.6505 |",
  "output_csv": "/data/scib_rankings.csv"
}
```

---

## 9. `benchmark_perturbations`

Wraps `tl.run_cell_eval()` across paired or split datasets, scoring differential perturbation responses against reference controls. Returns a concise Markdown summary table.

### Parameters
- `path` (*str*, required): Path to the real observations dataset (or split dataset).
- `control_pert` (*str*, required): Identifier corresponding to unperturbed control (e.g. `'ctrl'`).
- `pert_col` (*str*, default="perturbation"): Column identifying perturbation conditions.
- `split_col` (*str*, optional): Column indicating `'real'` vs `'predicted'` observations.
- `path_pred` (*str*, optional): Path to predicted AnnData dataset when using paired files.
- `embedding_keys` (*list[str] | str*, optional): Target representation keys to evaluate.
- `profile` (*str*, default="minimal"): Evaluation profile (`"full"`, `"minimal"`, `"de"`, `"vcc"`).
- `output_csv` (*str*, optional): Filesystem path to export results as CSV.

### Response Schema
```json
{
  "status": "success",
  "control_pert": "ctrl",
  "pert_col": "perturbation",
  "n_perturbations": 12,
  "markdown_table": "| Perturbation | Mean_R2 | DEG_Overlap | Delta_L2 | Direction_Agreement |\n|---|---|---|---|---|\n| drug_A | 0.8421 | 0.7912 | 0.1245 | 0.9123 |\n| drug_B | 0.7910 | 0.7104 | 0.1832 | 0.8654 |",
  "output_csv": "/data/pert_metrics.csv"
}
```
