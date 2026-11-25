---
name: embpy-embedding-task-benchmark
description: Benchmark which embedding model best predicts a measured outcome with embpy - train ridge, kNN, random-forest or XGBoost regressors on gene, protein, molecule or text embeddings to predict a numeric readout (IC50, viability, expression) and compare models in one table. Use when the user asks which embedding is most informative for their task, wants to compare foundation models or fingerprints for predicting drug response or perturbation effects, or needs a model-selection table.
---

# Benchmarking embeddings on a prediction task

The question: *given an entity's embedding, how well can a simple regressor predict the
user's readout?* Better embeddings for the task give better held-out scores.

Expected input: an `.h5ad` with one row per entity (or per measurement), an `.obs` column
with the entity identifier (here `smiles`) and either a numeric `.obs` target (here `ic50`)
or an expression matrix in `.X` (`target="X"`).

## 1. One embedding at a time

```python
import anndata as ad
import embpy as ep

adata = ad.read_h5ad("data/ic50.h5ad")
results = ep.tl.benchmark_embeddings(
    adata,
    perturbation_column="smiles",
    perturbation_type="chemical",
    target="ic50",
    embedding_model="morgan_fp",
    models=["ridge", "knn"],
    mode="quick",
    groupby="smiles",
)
print(results)
```

- `groupby` keeps all rows with the same value on the same side of every split. When the goal is
  to predict compounds the model has not seen and a compound has several rows (replicates, doses,
  cell lines, plates), set it to the column that names the compound, e.g. `groupby="compound_id"`.
  Without it the rows are split one by one, the test compounds were already seen in training, and
  the score measures recall: on toy data with random targets and five rows per compound, a split
  by row gave R² 0.96 and holding out whole compounds gave R² below 0, as it should for noise.
- A SMILES column works as the group only if each compound is written the same way in every row.
  The same molecule has many SMILES strings; canonicalize them first
  (`resolve_identifiers(identifiers=[...], entity_type="molecule")`) or group by a compound id.
- `perturbation_type` is `"chemical"` (SMILES) or `"genetic"` (gene names; set `id_type` and
  `organism`).
- Pass `obsm_key=` instead of `embedding_model=` to benchmark an embedding already in the file.
- `models` chooses regressors from `"linear"`, `"ridge"`, `"knn"`, `"random_forest"`.
- `mode="quick"` is one train/test split; use `mode="rigorous"` (5-fold CV with tuning) for
  anything you report, and always with fewer than ~200 rows.

## 2. Several embedding models in one table

```python
table = ep.tl.run_pipeline(
    adata,
    perturbation_column="smiles",
    use_case="small_molecule",
    target="ic50",
    embedding_models=["morgan_fp", "maccs_fp"],
    prediction_models=["ridge", "knn"],
    reduce_dim=None,
)
print(table)
```

- `use_case` is `"protein"`, `"gene"`, `"small_molecule"` or `"text"`;
  `ep.tl.list_embedding_models(use_case)` lists the models it can run. `embedding_models`
  is required (a list, or `"all"`, which can be slow and download large models).
- Models that fail (missing dependency) are skipped with a warning; check that every requested
  model appears in the result and tell the user about any that do not.
- `reduce_dim=50` applies PCA first, which makes large embeddings comparable; with very few
  rows use `None`.
- `run_pipeline` has no `groupby` and splits by row. With several rows per compound, average
  them to one row per compound first, or call `benchmark_embeddings` with `groupby` for each
  embedding model.

## 3. Plot

```python
import matplotlib

matplotlib.use("Agg")
fig = ep.pl.plot_benchmark(results, metrics=["r2", "pearson"])
fig.savefig("benchmark.png", dpi=150, bbox_inches="tight")
```

## Reporting

Report held-out metrics (`r2`, `pearson`, `spearman`) per embedding and regressor, the number of
rows and the split, including what `groupby` held out together. An `r2` near or below 0 means no
better than predicting the mean. Small
differences between embeddings on a few dozen rows are noise; say so instead of declaring a winner.
