---
name: embpy-embedding-quality-checks
description: Sanity-check an embedding before using it - count NaN, constant and duplicate rows and dimensions, detect collapsed embeddings where everything looks alike, compare norms and value distributions across models, and check agreement between two embeddings, with plots. Use when the user asks whether an embedding looks right, gets suspicious results downstream, compares a new model's output to an old one, or before benchmarking or training on embeddings.
---

# Embedding quality checks

Run these on any `.obsm` embedding before trusting downstream results.

## 1. Basic integrity

```python
import anndata as ad
import numpy as np
import embpy as ep

adata = ad.read_h5ad("data/cells_embedded.h5ad")
key = "X_pca"
X = np.asarray(adata.obsm[key], dtype=float)
finite_rows = np.isfinite(X).all(axis=1)
report = {
    "shape": X.shape,
    "rows_with_nan_or_inf": int((~finite_rows).sum()),
    "constant_dims": int((X[finite_rows].std(axis=0) == 0).sum()),
    "all_zero_rows": int((np.abs(X[finite_rows]).sum(axis=1) == 0).sum()),
    "duplicate_rows": int(finite_rows.sum() - len(np.unique(X[finite_rows].round(6), axis=0))),
}
print(report)
```

- NaN rows are expected only for inputs that could not be embedded (e.g. controls embedded with
  `missing="nan"`); any others point to a failed model run.
- Constant dimensions carry no information; many of them suggest a truncated or padded output.
- Duplicate rows mean different inputs got the same vector, e.g. the same sequence twice, or a
  lookup table returning one vector for unknown identifiers. Find which inputs they are.

## 2. Collapse: does everything look the same?

```python
n = ep.tl.sample_size_for(epsilon=0.05, delta=0.05)
rows = np.random.default_rng(0).choice(adata.n_obs, size=min(adata.n_obs, n), replace=False)
sample = adata[rows].copy()
similarity = ep.tl.compute_similarity(sample, obsm_key=key, metric="cosine")
off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]
print(len(rows), round(float(off_diagonal.mean()), 3), round(float(off_diagonal.std()), 3))
```

- `sample_size_for(epsilon, delta)` is the number of rows needed to estimate a mean within
  `epsilon` with probability `1 - delta` (738 for 0.05 / 0.05), whatever the dataset size, so the
  check stays fast on large data.
- A mean cosine similarity near 1 with a small spread means the vectors share one dominant
  direction (common for raw transformer embeddings). Similarity rankings and clustering are then
  unreliable: centre the embedding (subtract the column means) or reduce it with PCA first.

## 3. Compare models or versions

```python
import matplotlib

matplotlib.use("Agg")
fig = ep.pl.embedding_norms(adata, obsm_keys=["X_pca", "X_scvi"])
fig.savefig("embedding_norms.png", dpi=150, bbox_inches="tight")
fig = ep.pl.embedding_distributions(adata, obsm_keys=["X_pca", "X_scvi"], n_dims=5)
fig.savefig("embedding_distributions.png", dpi=150, bbox_inches="tight")
fig = ep.pl.cross_embedding_correlation(adata, obsm_key_a="X_pca", obsm_key_b="X_scvi")
fig.savefig("pca_vs_scvi_similarity.png", dpi=150, bbox_inches="tight")
```

- Norm and value distributions should look similar across runs of the same model; a shift means
  a different model version, layer, pooling or preprocessing.
- `cross_embedding_correlation` plots pairwise similarities in one space against the other; the
  correlation in the title summarises how much the two agree.

Whether the embedding separates known labels is covered by `embpy-embedding-visualization`
(within- vs between-label similarity) and `embpy-embedding-clustering` (kNN label purity).

## Reporting

List each check with its number and say plainly whether anything failed. Do not call an
embedding "good" from these checks alone: they catch broken embeddings, not uninformative ones.
