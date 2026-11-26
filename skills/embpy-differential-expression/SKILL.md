---
name: embpy-differential-expression
description: Find differentially expressed genes between groups of cells with embpy - each perturbation against control, each cluster or cell type against the rest - and get clean ranked tables of marker genes with fold changes and adjusted p-values. Use when the user asks for marker genes, DE genes, upregulated or downregulated genes for a knockout, drug or cluster, or wants a DEG table from an .h5ad.
---

# Differential expression

## 1. Normalize first

DE tests compare log-normalized expression. If `inspect_data` reports `raw_counts`, normalize
with the embpy MCP server:

```python
preprocess_counts(path="data/screen.h5ad", preprocessing="auto", select_hvg=False, output_path="data/screen_norm.h5ad")
```

`preprocess_counts` keeps raw counts in `.X` and writes log-normalized values to
`.layers["log_normalized"]`, so switch to that layer before testing. Keep all genes
(`select_hvg=False` keeps every gene; filtering still drops genes expressed in too few cells).

## 2. Test

```python
import anndata as ad
import embpy as ep

adata = ad.read_h5ad("data/screen_norm.h5ad")
adata.X = adata.layers["log_normalized"]
adata = ep.tl.rank_genes_groups(adata, groupby="target_gene", reference="control", method="wilcoxon")
```

- `groupby` is the `.obs` column with the groups; `reference="control"` compares every group with
  the control cells, `reference="rest"` (default) compares each group with all other cells
  (marker genes per cluster or cell type).
- `method`: `"wilcoxon"` (default, robust), `"t-test"`, `"logreg"`.
- Results are stored in `adata.uns["rank_genes_groups"]`.

## 3. Tables

```python
tp53 = ep.tl.get_deg_dataframe(adata, group="TP53", pval_cutoff=0.05, log2fc_min=0.5)
print(tp53.head(10))

groups = sorted(set(adata.obs["target_gene"]) - {"control"})
counts = {g: len(ep.tl.get_deg_dataframe(adata, group=g, pval_cutoff=0.05, log2fc_min=0.5)) for g in groups}
print(counts)
```

- Columns: `gene`, `score`, `logfoldchange`, `pval`, `pval_adj` (Benjamini-Hochberg), sorted by
  absolute score. Filter on `pval_adj`, not `pval`.
- Positive `logfoldchange` = higher in the group than in the reference.

## Interpreting

- Cell-level tests treat every cell as independent, so p-values are very small with many cells.
  Rank by effect size and report the number of cells per group. When the data has replicates
  or donors, recommend a pseudobulk analysis (e.g. DESeq2) for formal claims.
- Groups with fewer than ~10 cells give unstable results; say so.
- To check predicted against observed DE genes, use `embpy-perturbation-prediction-metrics`.
