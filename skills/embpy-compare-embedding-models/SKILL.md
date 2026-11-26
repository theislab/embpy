---
name: embpy-compare-embedding-models
description: Compare how different embedding models represent the same genes, proteins, molecules or cells - linear CKA, TSI/QSI and mutual-kNN alignment, similarity-structure correlation and kNN-neighbourhood overlap, with heatmaps. Use when the user asks whether two foundation models encode the same information, which models agree or disagree, or how much an embedding changes between models or settings.
---

# Comparing embedding models

All comparisons need **the same entities in the same row order** in every embedding.

## 1. Embed the same entities with each model

One `embed` call with several models keeps the rows aligned:

```python
import embpy as ep

smiles = [
    "CCO", "CCCO", "CCCCO", "CC(=O)O", "CC(=O)OC1=CC=CC=C1C(=O)O",
    "C1=CC=CC=C1", "CC1=CC=CC=C1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "OC1=CC=CC=C1",
]
mols = ep.BioEmbedder(device="cpu").embed(
    smiles, entity_type="molecule", model=["morgan_fp", "maccs_fp", "rdkit_fp"], output="anndata"
)
keys = list(mols.obsm)
matrices = {key.split("__")[2]: mols.obsm[key] for key in keys}
```

For cells, compute several models into one file (e.g. `compute_embeddings(..., models=["pca", "scvi"])`)
and use those `.obsm` keys. If embeddings were computed separately, subset each to the shared
`obs_names` in the same order before comparing: models drop entities they cannot embed.

## 2. Score agreement

```python
cka = ep.tl.alignment_matrix(matrices, metric="linear_cka")
print(cka.round(3))

mutual = ep.tl.alignment_matrix(matrices, metric="mutual_knn", k=3)
summary = ep.tl.compare_embedding_matrices(matrices, metric="cosine", k=3)
print(summary)

per_entity, mean_jaccard = ep.tl.compute_knn_overlap(mols, obsm_key_a=keys[0], obsm_key_b=keys[1], k=3)
print(mean_jaccard)
```

- `alignment_matrix` returns a symmetric table (1.0 on the diagonal). `metric`: `"linear_cka"`
  (global geometry, fast), `"tsi"` / `"qsi"` (triplet / quadruplet ordering of distances),
  `"mutual_knn"` (local neighbourhoods; pass `k`). Dimensions may differ between models.
- `compare_embedding_matrices` correlates the pairwise-similarity structures and reports kNN
  overlap for each pair.
- `compute_knn_overlap` gives the per-entity Jaccard overlap of k-nearest neighbours, useful to
  find *which* entities the two models disagree on.
- Keep `k` well below the number of entities.

## 3. Plot

```python
import matplotlib

matplotlib.use("Agg")
fig = ep.pl.knn_overlap(mols, obsm_keys=keys, k=3)
fig.savefig("knn_overlap.png", dpi=150, bbox_inches="tight")
```

`ep.pl.cross_model_similarity(adata, obsm_keys=..., method="cosine_correlation")` draws the
same comparison as a heatmap of similarity-structure correlations.

## Interpreting

High CKA with low mutual-kNN means the models agree globally but order close neighbours
differently. Agreement is not quality: two models can agree and both miss the biology. To ask
which model is *better* for a task, use the `embpy-embedding-task-benchmark` skill.
