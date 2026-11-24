# Protein–phenotype alignment

These tools measure whether proteins close in an embedding space correspond to
genetic perturbations with similar transcriptomic responses. They compare
representations; they do not train a response predictor.

Use an AnnData with matched observations in two `.obsm` matrices. Protein and
phenotype vectors may have different dimensions. Each row must refer to the same
perturbation in both matrices. Start with one cell line, dose and time point;
combining contexts can confound the comparison.

```python
from embpy import tl

# adata_pb: one row per perturbed gene in a single biological context.
# Protein vectors and phenotype vectors are already aligned to its rows.
scores = tl.protein_phenotype_alignment(
    adata_pb,
    protein_obsm_key="X_esm2_650M",
    sc_obsm_key="X_scgpt",
    k=10,
    n_pca_components=50,
    n_permutations=999,
    random_state=42,
)
display(tl.alignment_summary(scores))
```

For cell-level observations, pass `perturbation_col="gene"`. Both matrices are
averaged per perturbation after filtering controls. Protein vectors should
correspond to the perturbed gene and be repeated across its replicate cells.
Do not pass replicate cells as independent rows without `perturbation_col`.

```python
scores = tl.protein_phenotype_alignment(
    adata,
    protein_obsm_key="X_esm2_650M",
    sc_obsm_key="X_scgpt",
    perturbation_col="gene",
    control_col="is_control",
    control_ids={True},
    random_state=42,
)
```

Controls are excluded for both cell-level and already aggregated inputs.
At least three perturbations must remain. Missing perturbation labels and
non-finite embedding values raise errors. The input AnnData is not modified.

| Output | Meaning |
|---|---|
| `mantel_rho` | Spearman correlation between pairwise cosine distances in the two original spaces. |
| `mantel_pval` | Two-sided p-value from jointly permuting perturbation labels on both axes of one distance matrix. |
| `knn_jaccard_mean` | Mean intersection/union of the two neighbour sets; `k` is capped at the number of other perturbations. |
| `cross_modal_auroc_mad_*` | Protein-similarity AUROC for recovering phenotype neighbours above a median-plus-MAD threshold. Rows without both outcome classes are omitted; no eligible rows gives NaN. |
| `cross_modal_recall_*` | Fraction of top-K phenotype neighbours recovered by protein similarity. K at least the number of perturbations gives NaN. |

PCA applies only to the phenocopy AUROC and recall metrics, independently in
each space; `n_pca_components=None` disables it. `mad_thresholds=()` and
`recall_ks=()` disable those metric families. Neighbourhood scores are
descriptive, with no attached significance test. Tied similarities can make
top-K membership ambiguous, so inspect constant or duplicate embeddings before
interpreting these scores.

The Mantel test uses exchangeable perturbation labels, not independent pairwise
distances. With 999 random permutations its smallest p-value is 0.001; use more
permutations for finer resolution. `n_permutations=0` returns only the correlation
and a NaN p-value. Constant distances make both values undefined. Correct for
multiple testing when comparing many models, contexts or subsets. Distance
matrices require memory quadratic in the number of perturbations. See the
[scikit-bio Mantel documentation](https://scikit.bio/docs/dev/generated/skbio.stats.distance.mantel.html)
for the permutation principle.

For several protein models, collect their scorecards in a dictionary and pass
it to `tl.alignment_summary({"esm2": esm2_scores, "prott5": prott5_scores})`.
For the Mantel test alone, call
`tl.cross_modal_mantel(adata_pb, "X_esm2_650M", "X_scgpt", random_state=42)`;
it also supports `metric="euclidean"`.
