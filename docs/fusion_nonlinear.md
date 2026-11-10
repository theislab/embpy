# Kernel, DeepSets and gated fusion

Stage 3 adds nonlinear feature fusion under `embpy.tl.fusion`. All methods take
named `EmbeddingResult` views of the same entities and return another
`EmbeddingResult` with preserved identifiers and fitted provenance.

Download the [executed synthetic-data tutorial](notebooks/09_fusion_nonlinear.ipynb).
Real LINCS results and executed notebooks live in the
[reproducibility repository](https://github.com/theislab/embpy_reproducibility/tree/next-main/small_molecule/lincs/fusion).

| Operator | Uses targets? | Mapping | Dependency |
|---|---|---|---|
| `KernelFusion` | No | Weighted kernel sum → truncated Nyström features | Core |
| `DeepSetsFusion` | Yes | Named projections → shared processing → mean pooling → nonlinear output | PyTorch |
| `GatedFusion` | Yes | Named projections → compound-dependent softmax pooling → nonlinear output | PyTorch |
| `ConcatMLPFusion` | Yes | Concatenation → nonlinear representation | PyTorch; benchmark control |

## Kernels for heterogeneous views

```python
from embpy.tl.fusion import KernelFusion

fusion = KernelFusion(
    n_components=64, n_landmarks=256,
    kernels={"morgan": "tanimoto", "language": "rbf"},
    normalization={"morgan": "none", "language": "l2"},
    random_state=0,
)
fusion.fit(training_views)
test_features = fusion.transform(test_views)
```

The kernel is a weighted sum of per-view RBF, linear or Tanimoto kernels.
Fixed nonnegative weights are normalized to sum to one. Tanimoto requires
nonnegative inputs; raw binary fingerprints are appropriate, z-scored
fingerprints are invalid. Zero/zero similarity is defined as one.
RBF's default `gamma="median"` is the inverse median nonzero squared distance
in at most 512 training rows after normalization. A positive scalar or exact
per-view mapping overrides it. No labels are used: this is **not supervised
multiple-kernel learning**.

Landmarks are sampled without replacement from training entities. The leading
positive eigenpairs of their combined kernel give
`Z(X) = K_XL U / sqrt(eigenvalues)`. By default the training feature mean is
subtracted from all outputs. This is a low-rank kernel map, not kernel PCA.
Output width cannot exceed positive landmark rank and never silently shrinks.
Transformation is batched against landmarks; it never needs an all-entity
kernel matrix. Normalization, bandwidths and landmarks are training-fitted.

## Supervised neural representations

Install `pip install "embpy[torch]"`. Importing the API does not import PyTorch;
neural fitting and model deserialization load it when needed.

```python
from embpy.tl.fusion import DeepSetsFusion, GatedFusion

fusion = GatedFusion(64, hidden_dim=128, pre_components=128, epochs=75, random_state=0)
fusion.fit(training_views, y_train)  # DataFrame indexed by entity ID
test_features = fusion.transform(test_views)
weights = fusion.view_weights(test_views)
training_head_predictions = fusion.predict(test_views)
```

Finite numeric regression targets are standardized on training data only. A
linear head on the learned representation minimizes MSE. `predict` restores
target units, while `transform` returns features for a downstream readout.
Indexed targets align by ID. Array targets follow the first view's row order
and require the default strict join; an inner join requires indexed targets.

DeepSets uses named input adapters, shared nonlinear processing, mean pooling
and a nonlinear output map. Gated fusion replaces mean pooling with softmax
weights derived from projected views. The gate never implicitly receives MoA
labels. Its weights describe the fitted model, not causal feature importance.
All fitted named views must be present at inference. Different widths and
dictionary reordering are supported; missing or arbitrary new views are not.

`pre_components` optionally fits strict-width PCA per training view. Input
normalization supports `l2` (default), `none`, `zscore`, or per-view mappings.
Training uses seeded mini-batches for a fixed epoch count, without an implicit
validation split or early stopping. Select epochs externally in grouped
training folds. `device` controls CPU/CUDA training; stored models and batched
inference use CPU. The caller's RNG state is restored. Repeatability is tested
on CPU in one environment, not guaranteed across devices/library versions.

## Nested benchmarking and controls

```python
from embpy.tl import benchmark_fusion
from embpy.tl.fusion import ConcatMLPFusion, FusionConfig, GatedFusion

result = benchmark_fusion(
    views, targets, groups=compound_groups,
    fusions={
        "gated": FusionConfig(GatedFusion(64), param_grid={"epochs": [25, 75]}),
        "concat_mlp": FusionConfig(ConcatMLPFusion(64), param_grid={"epochs": [25, 75]}),
        "single_mlp": FusionConfig(
            ConcatMLPFusion(64), views=("morgan",), param_grid={"epochs": [25, 75]},
        ),
    },
    single_n_components=64, cv=3, seeds=(0, 1, 2),
    param_grid={"alpha": [1, 100, 10000]},
)
```

The adapter forwards training targets into fusion. Inner-fold candidates fit
their own preprocessing and representation. `groups` keeps related entities
together in outer and inner folds; outer-test targets cannot choose epochs or
other parameters. Without tuning, a prespecified epoch budget is used.
Manage cache lifetimes; real-data caches belong on the compute cluster.
For an intersected benchmark cohort, set `join="inner"` on `benchmark_fusion`
and leave the neural operator's `join="strict"`: sklearn supplies aligned arrays
inside cross-validation, whereas a direct neural inner join requires indexed targets.

Compare supervised fusion against both single-view and concatenation neural
controls. A gain over PCA alone cannot distinguish the fusion architecture
from extra supervision/capacity. Equal output widths do not imply equal
trainable parameter counts; benchmark counts cover the neural fusion network and its training head, excluding the separate Ridge readout. Linear projection entries are reported separately.
The label-only retrieval benchmark accepts kernels but explicitly rejects
neural operators because it does not provide regression targets.

Pair gains with uncertainty and corrected tests, including negative results.
Nonsignificance is not equivalence; a significant overall comparison does not
identify a winning pair or prove improvement on an untouched dataset.

## References

See the [Nyström feature-map construction](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html)
and [Deep Sets](https://arxiv.org/abs/1703.06114). The gated implementation is
inspired by [gated multimodal fusion](https://arxiv.org/abs/1702.01992), using
softmax view weights; it is not a reproduction of that paper's exact GMU.
