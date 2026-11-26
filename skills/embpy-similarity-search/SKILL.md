---
name: embpy-similarity-search
description: Find the genes, perturbations or compounds most similar to a query in an embpy embedding space - nearest-neighbour tables, similarity rankings and similarity heatmaps. Use when the user asks which genes are most similar to TP53, which perturbations phenocopy a knockout, which compounds resemble a reference drug, or wants a pairwise similarity matrix of entities.
---

# Similarity search in embedding space

Similarity tools in `embpy.tl` work on **rows** of an `.obsm` matrix and identify rows by
`obs_names`. Get the entities into rows first, then query.

## 1. Put one entity per row

From a list of genes (prior-knowledge embedding):

```python
import embpy as ep

embedder = ep.BioEmbedder(device="cpu")
genes = embedder.embed(
    ["TP53", "MDM2", "CDKN1A", "BAX", "GAPDH", "ACTB", "BRCA1", "ATM"],
    entity_type="gene",
    model="gene2vec",
    output="anndata",
    is_perturbation=True,
)
key = list(genes.obsm)[0]
# Rows are named by canonical Ensembl id; the input symbol is kept in obs["gene_symbol"].
genes.obs_names = genes.obs["gene_symbol"].astype(str)
```

- `is_perturbation=True` makes each gene a row in `.obsm`. Without it, standalone gene
  embeddings are stored per feature in `.varm`, which these tools do not read.
- Genes the model has no vector for are dropped; compare `genes.n_obs` with your list and tell
  the user which are missing.
- Molecules and proteins are already rows: `embedder.embed(smiles_list, entity_type="molecule", model="morgan_fp", output="anndata")`.

From per-cell data (one row per cell or well), average replicates into one row per
perturbation first:

```python
import anndata as ad

cells = ad.read_h5ad("data/screen_embedded.h5ad")
per_gene = ep.tl.pseudobulk_embeddings(cells, group_col="target_gene", obsm_key="X_pca")
effects = per_gene[per_gene.obs_names != "control"].copy()
effects.obsm["X_pca"] = effects.obsm["X_pca"] - per_gene["control"].obsm["X_pca"]
```

The pseudobulk has one row per value of `target_gene` (its `obs_names`), with the means in
`.obsm["X_pca"]`. For measured phenotypes, compare **effects** (mean minus the control mean):
raw means share the baseline cell state, so everything looks similar to everything, and the
control tends to come out as the nearest neighbour.

## 2. Query

```python
neighbors = ep.tl.nearest_neighbors(genes, obsm_key=key, query="TP53", k=5)
print(neighbors)

ranking = ep.tl.rank_perturbations(genes, "TP53", obsm_key=key, top_k=5)
print(ranking)

similar_to_tp53_ko = ep.tl.rank_perturbations(effects, "TP53", obsm_key="X_pca", top_k=3)
print(similar_to_tp53_ko)
```

- `nearest_neighbors` returns a table with `query_id`, `neighbor_id`, `rank`, `distance` and,
  for cosine, `similarity`. The query itself is excluded.
- `rank_perturbations` returns `(id, similarity)` pairs, highest first; `metric` is `"cosine"`
  (default) or `"pearson"`.
- The query must be an `obs_names` value. A `KeyError` saying the query was not found usually
  means the rows are Ensembl ids or well ids; rename or pseudobulk as above.

## 3. Show the structure

```python
import matplotlib

matplotlib.use("Agg")
fig = ep.pl.plot_similarity_heatmap(adata=genes, obsm_key=key, title="gene2vec cosine similarity")
fig.savefig("gene_similarity.png", dpi=150, bbox_inches="tight")
matrix = ep.tl.compute_similarity(genes, obsm_key=key, metric="cosine")
```

## Interpreting

- Similarity in a prior-knowledge space (gene2vec, GenePT, ESM-2) reflects what the model
  learned about the genes, not what happened in the user's experiment. Similarity in a
  phenotype space (PCA of perturbed cells) reflects the measured response. Say which one a
  ranking comes from.
- Report scores with the metric. Cosine values are only comparable within one embedding.
- With few rows (under ~20), rankings are fragile; say so instead of over-interpreting.
