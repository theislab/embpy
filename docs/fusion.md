# Combining embedding views

`embpy.tl.fusion` combines different representations of the **same entities**.
Stage 1 provides fitted concatenation, joint PCA, and late fusion of similarity
scores or ranks. These operators use NumPy/scikit-learn; no neural model is loaded.
See the [executable tutorial](notebooks/07_fusion.ipynb).
The [LINCS cluster validation report](fusion_benchmark_lincs.md) records the
first real-data comparison, input audit, limitations and execution provenance.
Stage 2 adds [CCA, MAXVAR GCCA and Procrustes](fusion_alignment.md), with
per-configuration view selection and nested tuning of fusion parameters.

## Feature fusion

Pass a mapping from meaningful view names to `EmbeddingResult` objects. Row IDs,
entity types and ID schemes are checked. Different row orders are reconciled by
ID. Different ID sets raise by default; `join="inner"` explicitly intersects them
and reports per-view dropped counts in `coverage_` and the output provenance.
Identifier conversion and gene-to-protein relationships must be resolved before
fusion. Equal row counts are not evidence that rows describe the same entities.

```python
from embpy.tl.fusion import PCAFusion, subset_views

train_views = subset_views(views, train_ids)
test_views = subset_views(views, test_ids)
fusion = PCAFusion(n_components=64, normalization="l2")
fusion.fit(train_views)
fused_train = fusion.transform(train_views)
fused_test = fusion.transform(test_views)
```

`ConcatFusion` supports:

| Option | Meaning |
| --- | --- |
| `normalization="none"` | Keep the original feature values. |
| `normalization="zscore"` | Fit per-feature means and standard deviations on training entities. Constant training features use scale one. |
| `normalization="l2"` | Normalize each entity's vector within each view. Zero vectors remain zero. |
| A normalization dictionary | Choose one of the above separately for every named view. |
| `balance_blocks=True` | Divide each block by its training RMS row norm. This controls the contribution of wider blocks. |
| `weights={...}` | Multiply each block by the square root of its fixed, nonnegative weight. At least one weight must be positive. |

`PCAFusion` uses the same preprocessing, concatenates the blocks, and fits one
joint PCA. It requires an explicit feasible output width and never silently
clamps it. Scaling and projection are reused at transform time. The fitted view
names, ordering and input dimensions are fixed; unseen entities are supported,
but an untrained new view is not.

Z-scoring does not bound unseen values. A descriptor that is constant or has a
small range during training can be extreme in a test fold. Inspect error
magnitudes alongside correlations: a reasonable Pearson score can coexist with
catastrophic MSE. These operators do not silently clip descriptors.

Independent PCAs do **not** produce a shared coordinate basis. Neither equal
widths nor L2 normalization make unrelated embedding coordinates safe to average.
This release therefore does not expose arbitrary cross-model vector averaging.

## Input/output and fitted state

- `views_from_anndata(adata, keys)` reads `.obsm` using unique observation IDs.
  Repeated compounds retain distinct observation IDs; pass their compound labels
  as benchmark groups.
- `view_from_store(store, ids)` reads selected canonical IDs from an existing
  `StaticEmbeddingStore`, failing on missing IDs. It does not load other stores.
- Feature outputs are ordinary `EmbeddingResult` objects. Existing `embpy.io`
  exporters write them as tables or AnnData. Inputs are not modified.
- Output provenance records source models, input widths, normalization, weights,
  coverage, a fingerprint of the training IDs and PCA explained variance.
- Standard `joblib.dump` / `joblib.load` preserve the fitted operator. Load only
  trusted serialized Python objects and retain the dependency versions used.

## Late fusion

`fuse_scores` and `fuse_ranks` accept dictionaries of pandas DataFrames whose rows
are query IDs and columns are candidate IDs. Both axes are aligned by label.
Each view must contain the same complete candidate universe, with finite scores.
Batch queries to bound memory for large retrieval tasks.

`fuse_scores` returns the weighted mean score. `fuse_ranks` computes ranks within
each query/view, assigns average ranks to ties, and returns negative weighted
mean rank. **Larger output is always better.** Set `higher_is_better=False` for
distance inputs. Late-fusion weights are normalized to sum to one. Score averaging
requires comparable score scales; rank averaging does not require equal scales.
These are retrieval utilities, not fused feature vectors.

## Paired benchmark integration

```python
from embpy.tl import benchmark_fusion
from embpy.tl.fusion import ConcatFusion, PCAFusion

result = benchmark_fusion(
    views,
    targets,                 # Series/DataFrame indexed by canonical entity ID
    groups=compound_groups,  # Series with the same entity IDs
    fusions={
        "concat": ConcatFusion(normalization="l2"),
        "pca": PCAFusion(n_components=64),
    },
    cv=3,
    seeds=(0, 1, 2),
    param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
)
result.scores       # one row per configuration / seed / outer fold
result.entity_scores
result.splits       # explicit train/test membership for every outer fold
result.coverage
```

All configurations and single-view baselines use identical outer splits. With
`tune=True`, readout parameters are selected inside each outer training fold,
and preprocessing is refitted inside every inner fold. Groups remain intact at
both levels. No operator supplied by the caller is fitted in place.

Use `single_n_components=64` for single-view controls matched to a 64-dimensional
PCAFusion. Full concatenation still has a larger feature budget; output widths
and fit/prediction runtimes are reported. Do not attribute its gain solely to
the fusion rule without considering readout capacity. If selecting a winning
fusion configuration, reserve a separate final test set.

`benchmark_fusion_retrieval` compares cosine retrieval for feature fusions with
score and rank ensembles. Test entities query training entities only. Relevance
is exact equality of categorical labels. It reports precision, recall, hit rate
and the number of queries with any relevant training candidate. Queries without
relevant candidates have zero precision/hit and undefined recall, rather than
being silently dropped from all metrics. Final-score ties use reference order.

Repeated folds/seeds are dependent: their standard deviation is descriptive,
not a confidence interval. Use the retained entity-level results for paired,
group-aware uncertainty estimates. Constant-profile correlations follow the
existing metric convention of zero.

The existing `benchmark_embeddings` also accepts `obsm_keys`, `fusion` and
`groupby`. Its `reduce_dim` now fits PCA within each training fold and no longer
writes a globally fitted PCA to `.obsm`. Rigorous mode now uses nested tuning;
`best_params` contains one dictionary per outer fold. For exploratory full-data
PCA plots, call the standalone reduction utility explicitly.

## Cluster validation protocol

`submission_scripts/run_fusion_lincs.sbatch` runs
`python -m embpy.scripts.benchmark_fusion_lincs` in a Slurm allocation. Set
`EMBPY_PYTHON` to the compute environment's Python and `EMBPY_SOURCE_ROOT` to the
checkout or immutable source snapshot. Pass `--input-h5ad`, `--embedding-root`
and `--output-dir`; `--prepare-only` prepares a cohort and `--reuse-cohort` reuses
it for separate regression/retrieval jobs.

This protocol predicts mean LINCS Level-5 profiles per chemical structure across
observed contexts and evaluates MoA retrieval. Structures connected through a
compound name stay in the same split. Neural features are read from named `e_*`
CSV columns, excluding the erroneous ChemBERTa `row_index` feature found in the
legacy AnnData. Morgan bits (radius 2, 1024 bits) and the installed RDKit version's
descriptor set are regenerated from SMILES because the legacy classical CSVs
were already globally standardized. Descriptor names, dimensions, versions,
source hashes and exclusions are saved with the cohort. Conflicting duplicate
vectors for one SMILES are excluded
from the shared cohort and listed in the audit; an arbitrary vector is not
selected or averaged. Structures with missing views or ambiguous MoA labels
are excluded from all compared methods equally. The original data files
are never changed. These results are not directly comparable to the historical
signature-level notebook or its RDKit-200/Morgan-1019 matrices.

Outputs include fold metrics, per-entity metrics, explicit splits, descriptive
summaries, a training-mean regression baseline, matched-width single-view runs,
and execution/provenance JSON. The benchmark evaluates the existing feature
collection; it does not establish universal superiority of any fusion method.
