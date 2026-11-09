# CCA, GCCA and Procrustes fusion

Stage 2 adds three fitted linear alignment methods under `embpy.tl.fusion`.
They consume named `EmbeddingResult` views of the **same entities**, and return
another `EmbeddingResult`. They do not generate embeddings or predict expression.
A downstream readout is evaluated separately.

| Method | Views | What is fitted | Use when |
|---|---|---|---|
| `CCAFusion` | Exactly two | Regularized canonical projections | Two representations may share correlated directions |
| `GCCAFusion` | Two or more | Regularized **MAXVAR** common space and per-view projections | Several representations may contain complementary shared signal |
| `ProcrustesFusion` | Two or more | Per-view PCA, then rotations/reflections toward a named reference | You want to align geometry before averaging |

Download the [executed tutorial](notebooks/08_fusion_alignment.ipynb).
It demonstrates training and unseen-entity transformation, serialization, nested
prediction and retrieval benchmarks, using synthetic data only.

## Fit once, transform new entities

```python
from embpy.tl.fusion import CCAFusion, GCCAFusion, ProcrustesFusion

cca = CCAFusion(n_components=64, regularization=0.1, pre_components=128)
train_fused = cca.fit_transform({"morgan": train_morgan, "molformer": train_molformer})
test_fused = cca.transform({"morgan": test_morgan, "molformer": test_molformer})

multi = GCCAFusion(n_components=64, regularization=0.1, pre_components=128)
geometry = ProcrustesFusion(n_components=64, reference="morgan")
```

Inputs are aligned by canonical ID, preserving the first view's entity order.
The default strict join rejects unequal entity sets; `join="inner"` explicitly
intersects them and records exclusions. All fitted views must also be provided
at transformation time, with the same feature widths and identifier schema.
Changing the dictionary order is safe. Inputs are not mutated. Use sklearn
`clone` for an unfitted copy and joblib to save fitted projections.

### Preprocessing and dimensions

All three methods default to row L2 normalization, followed by training-fitted
centering. `normalization="none"` and `"zscore"`, or an exact mapping from view
names to these choices, are also supported. Z-score scaling uses training
statistics only. L2 is a useful bounded-input default for the audited LINCS
cohort, whose raw RDKit descriptors include extreme values; it is not a universal
recommendation for every dataset.

CCA/GCCA optionally use train-fitted per-view PCA (`pre_components`) before
alignment. Procrustes always uses per-view PCA to `n_components`. Widths are
validated rather than silently reduced. CCA and Procrustes require that every
view retain at least that many non-null directions. GCCA allows lower-rank
individual views when their combined support has sufficient rank. Zero-rank
views fail clearly. Numerical support uses a relative singular-value threshold
`rank_tol=1e-7`, appropriate to the canonical float32 embedding inputs.

### What the methods compute

CCA whitens each centered view with a covariance ridge and uses the SVD of the
whitened cross-covariance. `regularization` is a scalar or exact per-view mapping.
For each view, the covariance ridge is the specified multiplier times the mean
**retained** covariance eigenvalue, after optional PCA and rank filtering. Zero
means ordinary CCA on the numerical support. This parameter is not a convex
shrinkage fraction. Scores from each view have unit training standard deviation
before averaging. `canonical_correlations_` reports actual training Pearson
correlations; `regularized_correlations_` reports the regularized singular values.
Neither is a held-out performance estimate.

GCCA uses the same ridge convention. It solves weighted ridge least squares to
a common matrix G with `G.T @ G = (n_train - 1) I` (MAXVAR). Its eigensystem comes
from the weighted sum of ridge projection matrices, implemented through compact
SVD factors. Large, low-rank solves use seeded randomized SVD with seven power
iterations. Each view is then mapped to G by ridge regression. The weighted
mean of these projections is scaled to unit training standard deviation.
**Both training and unseen entities use this same mapping**; the training-only
latent G is not returned in place of a transform. This is not SUMCOR MCCA.

Procrustes projects each view to PCA scores and finds the orthogonal map that
minimizes its squared distance to the reference scores. Reflections are allowed.
It preserves geometry within the reduced space; it does not whiten or rescale
the scores after alignment. `reference=None` means the first fitted view;
prefer an explicit name in recorded experiments.

For all three, fixed nonnegative `weights` are normalized to sum one for score
averaging. This differs from Stage 1 concatenation's square-root block weights.
No response or MoA labels enter these operators. Source provenance, training-ID
hash, preprocessing choices, fitted ranks and projection-map sizes accompany
outputs. `n_projection_coefficients_` counts stored map entries, **not** independent
statistical degrees of freedom or total model memory.

## Benchmark integration

```python
from embpy.tl import benchmark_fusion, benchmark_fusion_retrieval
from embpy.tl.fusion import CCAFusion, FusionConfig, PCAFusion

configs = {
    "pair_pca": FusionConfig(PCAFusion(64, normalization="l2"), views=("morgan", "molformer")),
    "pair_cca": FusionConfig(
        CCAFusion(64, pre_components=128),
        views=("morgan", "molformer"),
        param_grid={"regularization": [0.01, 0.1, 1.0]},
    ),
}
result = benchmark_fusion(
    views, targets, groups=compound_groups, fusions=configs,
    cv=3, seeds=(0, 1, 2), tune=True,
    param_grid={"alpha": [0.1, 1, 10, 100]},
    single_n_components=64,
)
retrieval = benchmark_fusion_retrieval(
    views, moa_labels, groups=compound_groups, fusions=configs,
    cv=3, seeds=(0, 1, 2), k=10, tune=True, single_n_components=64,
)
```

Targets, labels and groups are pandas objects indexed by canonical ID. Every
configuration uses the same cohort, aligned across **all supplied views**, even
when its `FusionConfig` selects a subset. Compare methods on identical input
subsets and matched output widths. A different reference view or normalization
is a separate model choice, not a harmless plotting change.

Outer folds estimate generalization to held-out groups. Inner folds select Ridge
and fusion parameters by negative MSE for regression; retrieval selects fusion
parameters by mean inner-fold precision@k. Thus label-informed parameter choice
is part of the benchmark, while the operators themselves remain unsupervised.
No outer-test entities, labels or preprocessing statistics enter fitting or
selection. Grids require `tune=True`. The supplied operator instances remain
unfitted. Optional `cache_dir` caches regression pipeline fits across readout
candidates; use a temporary directory and manage its lifetime.

Retrieval queries only training references, using exact categorical label
matches. Precision includes queries with no relevant training candidate; recall
for those queries is undefined. Score/rank averaging uses all supplied raw views
as separately named controls, not the view subset of any one feature fusion.

The [cluster benchmark report](fusion_alignment_benchmark_lincs.md) records the
real cohort, configuration choices and results. A strong training canonical
correlation is not evidence that fusion improves prediction or retrieval.

## Sources

- [Regularized CCA formulation](https://arxiv.org/abs/2011.01650).
- [Scalable generalized CCA and MAXVAR](https://arxiv.org/abs/1605.09459).
- [SciPy orthogonal Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html).
