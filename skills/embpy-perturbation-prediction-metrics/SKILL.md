---
name: embpy-perturbation-prediction-metrics
description: Score predicted perturbation responses (from GEARS, scGPT, STATE, CPA or any model) against observed single-cell data with embpy - pseudobulk R2, Pearson and delta-L2 metrics, phenocopy retrieval scores, differential-expression gene overlap and direction agreement, and the cell-eval suite. Use when the user has predicted and observed expression for perturbations and asks how good the predictions are, wants benchmark metrics, or wants to compare prediction models.
---

# Scoring perturbation predictions

Needs two AnnData objects with **the same genes** (`var_names`) and a shared perturbation
column (here `perturbation`, controls labelled `"control"`): observed data and predictions.
Both must be on the same scale (e.g. both log-normalized). Check this first and stop if not.

```python
import anndata as ad
import numpy as np
import embpy as ep

observed = ad.read_h5ad("data/observed.h5ad")
predicted = ad.read_h5ad("data/predicted.h5ad")
assert list(observed.var_names) == list(predicted.var_names), "align genes before scoring"
groups = sorted(set(observed.obs["perturbation"]) - {"control"})
print(len(groups), "perturbations")
```

## 1. Pseudobulk regression metrics

```python
def group_means(adata, labels):
    return np.vstack([np.asarray(adata[adata.obs["perturbation"] == g].X.mean(axis=0)).ravel() for g in labels])


control_mean = group_means(observed, ["control"])[0]
y_true, y_pred = group_means(observed, groups), group_means(predicted, groups)
metrics = ep.tl.compute_metrics(y_true, y_pred, mean_control=control_mean)
baseline = ep.tl.compute_metrics(y_true, np.tile(control_mean, (len(groups), 1)), mean_control=control_mean)
print(metrics)
print(baseline)
```

- Returns `mse`, `r2`, `pearson`, `spearman` and the delta-L2 metrics (`delta_l2_mae`,
  `delta_l2_pearson`: does the model get the *size* of each response right).
- Always show the "predict the control mean" baseline next to the model. Raw expression
  correlations are high for any model because baseline expression dominates; a model is only
  useful if it beats the baseline.

## 2. Does each prediction look like the right perturbation?

```python
phenocopy = ep.tl.phenocopy_score_adata(observed, predicted, pert_col="perturbation", control="control")
print(phenocopy)
```

Phenocopy metrics ask whether each predicted profile is closest to the observed profile of the
same perturbation (retrieval / recall), after subtracting the control mean.

## 3. Differentially expressed genes

```python
deg = ep.tl.compare_deg(observed, predicted, groupby="perturbation", reference="control", n_top_genes=20)
print(deg[["group", "jaccard", "precision", "recall", "direction_agreement"]])
```

Per perturbation: overlap of the top DE genes and the fraction whose direction of change agrees.
`predicted` must have the same `.obs` labels as `observed`.

## 4. cell-eval suite (MCP tool)

```python
benchmark_perturbations(path="data/observed.h5ad", path_pred="data/predicted.h5ad", control_pert="control", pert_col="perturbation")
```

Runs ArcInstitute cell-eval (`pip install "embpy[eval]"`) and returns `markdown_table`. The tool
rejects comparing a dataset with itself.

## Reporting

Report metrics per perturbation and averaged, the number of perturbations, and the baseline.
Do not average away failures: list the perturbations with the worst scores too.
