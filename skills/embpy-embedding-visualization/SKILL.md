---
name: embpy-embedding-visualization
description: Plot embpy embeddings as UMAP, t-SNE or PCA scatter plots coloured by metadata, grids comparing several embeddings, highlighted gene sets, and per-label similarity diagnostics, saved as image files. Use when the user asks to plot, visualize or show an embedding or UMAP, colour cells or perturbations by a column, compare embeddings side by side, or check visually whether an embedding separates labels.
---

# Visualizing embeddings

`embpy.pl` functions return a matplotlib `Figure`. In a terminal or agent session nothing is
displayed: always save to a file and give the user the path. You can open the PNG with your
file-reading tool to look at it before describing it.

```python
import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import embpy as ep

os.makedirs("figures", exist_ok=True)
adata = ad.read_h5ad("data/cells_embedded.h5ad")
print(list(adata.obsm), list(adata.obs.columns))
```

Pick `obsm_key` and `color` columns from what the file actually contains.

## One embedding, coloured by metadata

```python
fig = ep.pl.plot_embedding_space(adata, obsm_key="X_pca", color="cell_type", method="umap")
fig.savefig("figures/umap_cell_type.png", dpi=150, bbox_inches="tight")

fig = ep.pl.embedding_color_panel(adata, color_keys=["cell_type", "batch"], obsm_key="X_pca", method="umap")
fig.savefig("figures/umap_panels.png", dpi=150, bbox_inches="tight")
```

- `method` is `"umap"`, `"tsne"` or `"pca"`; the 2-D layout is computed from `obsm_key`.
- If 2-D coordinates already exist (e.g. `X_umap_pca` from the MCP `compute_umap` tool), pass
  `basis="X_umap_pca"` to reuse them instead of recomputing.
- For small sets of genes or compounds (under ~50 rows), `annotate=True` labels each point.

## Several embeddings side by side

```python
fig = ep.pl.all_embeddings(adata, obsm_keys=["X_pca", "X_scvi"], method="umap", color="cell_type")
fig.savefig("figures/all_embeddings.png", dpi=150, bbox_inches="tight")
```

## Does the embedding separate the labels?

```python
stats = ep.pl.within_vs_between_similarity(adata, label_key="cell_type", obsm_key="X_pca")
print(stats)
centroids = ep.pl.category_centroid_similarity(adata, label_key="cell_type", obsm_key="X_pca")
```

`within_vs_between_similarity` returns `mean_within`, `mean_between` and a one-sided
Mann-Whitney `p_value`: within-label similarity should be clearly higher. The centroid heatmap
shows which labels the embedding confuses. Quote these numbers rather than describing a
UMAP by eye; UMAP distances between clusters are not meaningful.

## Gene or compound sets

For per-gene or per-compound embeddings (one row per entity), `ep.pl.highlight_gene_sets`
draws one panel per named set, e.g. `gene_sets={"p53 pathway": ["TP53", "MDM2", "CDKN1A"]}`.
Members are matched against `obs_names` (or the column given as `label_col`), so make sure the
rows are named by symbol, as described in the `embpy-similarity-search` skill.

## Close figures in loops

Call `matplotlib.pyplot.close(fig)` after saving when making many figures, to free memory.
