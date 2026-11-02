---
name: embpy-perturbation-screen-analysis
description: Analyze a genetic perturbation screen (CRISPR knockout / knockdown, Perturb-seq, optical screen) stored as an .h5ad with one row per cell or well - harmonize target gene names, attach prior-knowledge gene embeddings per cell, measure which perturbations have a reproducible phenotype (mean average precision vs controls), and find perturbations with similar phenotypes. Use when the user has a perturbation screen and asks which knockouts had an effect, which perturbations phenocopy each other, or wants perturbation embeddings for modelling.
---

# Genetic perturbation screen analysis

Expected input: an `.h5ad` with one row per cell (or well), an `.obs` column naming the
perturbed gene (here `target_gene`) and a label for controls (here `"control"`). Run
`inspect_data` first and confirm the column and control label with the user.

## 1. Check the target gene names

```python
resolve_identifiers(path="data/screen.h5ad", obs_column="target_gene", entity_type="gene", target_id_type="ensembl", organism="human")
```

- Controls (`control`, `non-targeting`, ...) never map, so `partial_success` is expected;
  check that the unmapped entries are only controls.
- Aliases are reported in `notes` and ambiguous names in `ambiguous`. Fix them in the data
  before step 2, or their cells get no gene embedding.

## 2. Two kinds of embedding

**Prior knowledge about the perturbed gene** (same vector for every cell with that target):

```python
compute_embeddings(path="data/screen.h5ad", models="gene2vec", entity_type="perturbation", obs_column="target_gene", missing="nan", output_path="data/screen_prior.h5ad")
```

`missing="nan"` keeps control cells (no gene vector) as NaN rows instead of failing.

**Measured phenotype** of each cell (from its expression):

```python
compute_embeddings(path="data/screen_prior.h5ad", models="pca", entity_type="single_cell", output_path="data/screen_embedded.h5ad")
```

## 3. Which perturbations have a reproducible phenotype?

```python
import anndata as ad
import embpy as ep

adata = ad.read_h5ad("data/screen_embedded.h5ad")
activity = ep.tl.phenotypic_activity(
    adata,
    obsm_key="X_pca",
    perturbation_col="target_gene",
    control_col="target_gene",
    control_ids={"control"},
)
print(activity.sort_values("normalized_mean_ap", ascending=False))
```

- For each perturbation, mean average precision (`mean_ap`) measures how well its replicate
  cells retrieve each other against all other cells; `normalized_mean_ap` corrects for the
  number of replicates (0 = random). Control rows are excluded via `control_col` /
  `control_ids`.
- Rank perturbations by `normalized_mean_ap` and report `n_wells` alongside: values from
  2-3 replicates are noisy.

## 4. Which perturbations look alike?

```python
per_gene = ep.tl.pseudobulk_embeddings(adata, group_col="target_gene", obsm_key="X_pca")
effects = per_gene[per_gene.obs_names != "control"].copy()
effects.obsm["X_pca"] = effects.obsm["X_pca"] - per_gene["control"].obsm["X_pca"]
print(ep.tl.rank_perturbations(effects, "TP53", obsm_key="X_pca", top_k=3))
```

- `pseudobulk_embeddings` averages cells into one row per perturbation (`obs_names` = gene).
- Compare **effects** (each perturbation's mean minus the control mean), not raw means. Raw
  means share the baseline state, so they all look alike and the control often ranks as the
  closest "phenocopy".
- Low similarities (e.g. all below ~0.3) mean no perturbation clearly phenocopies the query;
  say that rather than naming a winner.
- For heatmaps, clustering and neighbour tables over these rows, use the
  `embpy-similarity-search` and `embpy-embedding-clustering` skills.

## Reporting

State the control label, the number of cells per perturbation, and which embedding each
result comes from. Prior-knowledge similarity says what is known about the genes; phenotype
similarity says what the screen measured.
