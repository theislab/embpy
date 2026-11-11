---
name: embpy-single-cell-workflow
description: Embed, visualize and benchmark single-cell AnnData (.h5ad) datasets with the embpy MCP tools - inspect a dataset, preprocess raw counts, compute PCA/scVI/foundation-model cell embeddings, project a UMAP, score embeddings with scIB metrics, and evaluate perturbation predictions with cell-eval. Use when the user points at an .h5ad file and asks to embed or cluster cells, compare cell embeddings, run scIB, or benchmark predicted against observed perturbation responses.
---

# Single-cell workflow with embpy

These steps use the tools of the `embpy` MCP server (`inspect_data`, `preprocess_counts`,
`list_model_registry`, `compute_embeddings`, `compute_umap`, `benchmark_scib`,
`benchmark_perturbations`). If the tools are missing, the server is not installed:
`pip install "embpy[mcp]"`, then run `embpy-mcp`.

Every tool reads an `.h5ad` path, writes a new `.h5ad` next to it (or to `output_path`) and
returns compact JSON with an `output_path`. Pass that path to the next step. The tools never
return matrices, so do not ask for them.

## 1. Inspect before anything else

```python
inspect_data(path="data/pbmc.h5ad")
```

Read from the result:
- `n_obs` / `n_vars`, `layers`, `obsm_keys` (embeddings that already exist).
- `expression_nature.detected_nature`: `raw_counts` needs preprocessing; `log_normalized` does not.
- `recommended_label_key` / `recommended_batch_key`: candidate columns for scIB. They come from
  column names only, so confirm them with the user when the choice matters.

## 2. Preprocess raw counts

```python
preprocess_counts(path="data/pbmc.h5ad", preprocessing="standard", n_top_genes=2000)
```

- `standard` applies `min_genes` / `min_cells` exactly as given. On small or toy datasets this
  can remove every cell (`n_cells: 0` in the result). Use `preprocessing="auto"` to scale the
  thresholds to the dataset, and tell the user which thresholds were changed
  (`adjustments_applied`).
- Raw counts stay in `.X` and `.layers["counts"]`; normalized values go to
  `.layers["log_normalized"]` (and `.layers["scaled"]` with `scale=True`).

## 3. Choose models, then embed

```python
list_model_registry(entity_type="single_cell")
```

Use only names from `recommended_models` / `available_models`. `runnable_singlecell_models`
lists the ones whose dependencies are installed here; `pca` always works, and `scvi` needs
`scvi-tools`. Foundation models (`scgpt`, `geneformer_v2_12L`, `state`) are slow on CPU, so
check `compute_device` and warn the user before starting one on a CPU.

```python
compute_embeddings(path="data/pbmc_preprocessed.h5ad", models=["pca", "scvi"], entity_type="single_cell", device="auto")
```

- Each model adds `.obsm["X_<model>"]` (e.g. `X_pca`); the result lists `embedding_keys` and
  `embedding_shapes`.
- With the default `preprocessing="auto"`, small datasets get lower QC/HVG thresholds so no
  cell is filtered out; the result lists them in `preprocessing_adjustments`. Mention them.
- `obs_column` / `var_column` do not apply to `single_cell`; it embeds the expression matrix.
- `status: "error"` with `error_type: "UnsupportedModelError"` means the model is not a
  single-cell model. Its message names the `entity_type` it belongs to.
- `ModelOOMError`: retry with a smaller `batch_size` or `device="cpu"`.

## 4. Visualize

```python
compute_umap(path="data/pbmc_preprocessed_embedded.h5ad", embedding_key="X_pca", n_neighbors=15)
```

The coordinates are stored as `.obsm["X_umap_<name>"]` (here `X_umap_pca`). The tool returns
their shape, not a plot.

## 5. Benchmark embeddings with scIB

```python
benchmark_scib(path="data/pbmc_preprocessed_embedded.h5ad", embedding_keys=["X_pca", "X_scvi"], label_key="cell_type", batch_key="batch")
```

- `label_key` must hold biological labels (cell types). `batch_key` is optional; without it
  only bio-conservation metrics are meaningful.
- Show the user `markdown_table` as is, and use `rankings` (best first by `total`) to
  summarize. Do not invent scores that are not in the table.
- Needs `pip install "embpy[scib]"`; the error says so when it is missing.

## 6. Benchmark perturbation predictions

Prediction and reference must be different groups of cells. Either split one file:

```python
benchmark_perturbations(path="data/norman.h5ad", control_pert="ctrl", pert_col="perturbation", split_col="split", pred_label="predicted", ref_label="observed")
```

or compare two files:

```python
benchmark_perturbations(path="data/observed.h5ad", path_pred="data/predicted.h5ad", control_pert="ctrl", pert_col="perturbation")
```

- With `split_col`, both `pred_label` and `ref_label` are required and must differ; ask the user
  which category holds predictions if the column does not make it obvious.
- The tool refuses identical labels, the same file twice, or two files with the same cells and
  values, because those would score predictions against themselves.
- `control_pert` is the label of unperturbed cells in `pert_col`.
- Needs `pip install "embpy[eval]"` (cell-eval).

## Reading results

| `status` | Meaning |
|---|---|
| `success` | The step did everything that was asked. |
| `partial_success` | Some inputs worked; report the counts in the result to the user. |
| `error` | Nothing usable was produced. Read `error_type` and `error`, fix the arguments, retry. |

Report numbers from the tool results only, and tell the user where files were written
(`output_path`).
