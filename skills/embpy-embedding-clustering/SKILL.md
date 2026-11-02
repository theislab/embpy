---
name: embpy-embedding-clustering
description: Cluster cells, genes, perturbations or compounds in an embpy embedding with Leiden, k-means or spectral clustering, check the clusters against known labels, and summarize what each cluster contains. Use when the user wants to cluster an embedding, find groups of similar perturbations or compounds, compare clusters with cell types, or ask what a cluster represents.
---

# Clustering embeddings

Clustering runs on an `.obsm` matrix (rows = cells, perturbations or compounds). If the data
has no embedding yet, compute one with the embpy MCP server first:

```python
compute_embeddings(path="data/cells.h5ad", models="pca", entity_type="single_cell", output_path="data/cells_embedded.h5ad")
```

## 1. Cluster

```python
import anndata as ad
import embpy as ep

adata = ad.read_h5ad("data/cells_embedded.h5ad")
adata = ep.tl.cluster_embeddings(adata, obsm_key="X_pca", method="leiden", resolution=1.0, key_added="cluster")
print(adata.obs["cluster"].value_counts())
```

- `method="leiden"` (graph-based; `resolution` sets granularity, higher = more clusters,
  `n_neighbors` sets the graph) or `"kmeans"` / `"spectral"` with `n_clusters`.
- Leiden needs `igraph` and `leidenalg` (`pip install "embpy[test]"` or install them directly).
- When the user expects a known number of groups, `method="kmeans", n_clusters=...` is easier
  to explain; otherwise start with Leiden at `resolution=1.0` and adjust.

## 2. Check clusters against labels the user already has

```python
purity = ep.pl.knn_label_purity(adata, label_key="cell_type", obsm_key="X_pca", k=10)
print(purity["__overall__"], purity["__baseline__"])

from sklearn.metrics import adjusted_rand_score

print(adjusted_rand_score(adata.obs["cell_type"], adata.obs["cluster"]))
```

- `knn_label_purity` returns, per label, the fraction of neighbours sharing it, plus
  `__overall__` and the random expectation `__baseline__`. Overall well above baseline means
  the embedding separates the labels.
- The adjusted Rand index compares the clustering with the labels (1 = identical, ~0 = random).

## 3. Describe each cluster

```python
enrichment = ep.tl.cluster_annotation_enrichment(adata, cluster_key="cluster", annotation_key="cell_type", top_k=3)
print(enrichment)
```

`annotation_key` can hold one label per row or a list of terms per row (e.g. GO terms per
gene). The table lists, per cluster, each term's in-cluster and overall frequency and the
fold `enrichment`. It is a quick summary, not a statistical test; say so if the user wants
significance.

## 4. Plot and save

```python
import matplotlib

matplotlib.use("Agg")
fig = ep.pl.plot_cluster_composition(adata, cluster_key="cluster", color_by="cell_type")
fig.savefig("cluster_composition.png", dpi=150, bbox_inches="tight")
adata.write_h5ad("data/cells_clustered.h5ad")
```

`ep.pl.leiden_overview(adata, obsm_key="X_pca", color_by="cell_type")` draws UMAP, composition
and cluster-size panels in one figure.

## Reporting

Give the number of clusters, their sizes and the parameters used (method, resolution or
`n_clusters`, embedding key). Cluster numbers are arbitrary labels; describe clusters by their
content, not their number.
