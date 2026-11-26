---
name: embpy-export-and-reuse-embeddings
description: Save embpy embeddings to portable files (NPZ, CSV, Zarr) with a provenance sidecar, read them back, export embeddings already stored in an .h5ad, and reduce embeddings from different models to a common dimension. Use when the user wants embeddings as a file for another tool (PyTorch, R, scikit-learn, a collaborator), wants to cache embeddings instead of recomputing them, or needs several models' embeddings at the same width.
---

# Exporting and reusing embeddings

## 1. Write embeddings straight to a file

```python
import embpy as ep

embedder = ep.BioEmbedder(device="cpu")
genes = ["TP53", "MDM2", "GAPDH", "ACTB", "BRCA1", "EGFR"]
table = embedder.embed(genes, entity_type="gene", model="gene2vec", output_path="exports/gene2vec.npz")
print(table.shape)
embedder.embed(genes, entity_type="gene", model="gene2vec", output_path="exports/gene2vec.csv", fmt="csv")
```

- With `output_path`, `embed` writes the file (creating folders) and returns the same data as a
  table: one row per entity, indexed by the canonical ID (Ensembl for genes), with alias columns
  such as `gene_symbol` and then `dim_0 ... dim_N`.
- `fmt` is `"npz"` (default, compact), `"csv"` (readable anywhere) or `"zarr"`.
- A `<file>.meta.json` sidecar records the model, embpy version and settings. Keep it with the
  file.
- Entities the model has no vector for are left out; compare the row count with the input.

## 2. Read an export back

```python
import json

import numpy as np

saved = np.load("exports/gene2vec.npz")
matrix, ids = saved["matrix"], saved["entity_ids"]
metadata = json.loads(str(saved["metadata_json"]))
print(matrix.shape, ids[:3], metadata.get("entity_type"))
```

The `.npz` holds `matrix`, `entity_ids`, `dim_names` and JSON metadata. Read it with NumPy as
above; `embpy.io.load_legacy_embedding` is for older third-party embedding files and does not
read embpy's own exports. A CSV export reads with `pandas.read_csv(path, index_col=0)`.

## 3. Export embeddings already in an .h5ad

```python
import anndata as ad
import pandas as pd

adata = ad.read_h5ad("data/cells_embedded.h5ad")
pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names).to_csv("exports/cells_pca.csv")
```

Use `.obsm` for row embeddings (cells, compounds) and `.varm` with `var_names` for gene-level
ones. Rows that are all NaN (controls embedded with `missing="nan"`) are kept; tell the user.

## 4. Same width for several models

```python
reduced = embedder.embed(genes, entity_type="gene", model="gene2vec", output="anndata", harmonize_dim=4, is_perturbation=True)
print({key: value.shape for key, value in reduced.obsm.items()})
```

`harmonize_dim=N` PCA-projects each model's embedding to `N` dimensions (the key gets a
`__dim_N` suffix), so models of different widths can be concatenated or compared. It is fitted
on the entities in this call, so reduce all sets you want to compare in the same call.

## Reporting

Give the file path, format, number of rows and dimensions, the model, and any entities that
were dropped.
