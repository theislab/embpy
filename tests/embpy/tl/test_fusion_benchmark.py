"""Small synthetic tests of benchmark isolation, not performance experiments."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.tl.benchmark import _fit_fold, benchmark_embeddings, benchmark_fusion, benchmark_fusion_retrieval
from embpy.tl.fusion import ConcatFusion, PCAFusion
from embpy.tl.fusion._benchmark import MatrixFusion


@pytest.fixture
def cohort():
    rng = np.random.default_rng(2)
    ids = tuple(f"g{i}" for i in range(36))
    a, b = rng.normal(size=(36, 4)), rng.normal(size=(36, 3))
    views = {
        name: EmbeddingResult(matrix, ids, "gene", "id", EmbeddingProvenance(name))
        for name, matrix in [("a", a), ("b", b)]
    }
    y = pd.Series(2 * a[:, 0] + b[:, 1], index=ids)
    groups = pd.Series([str(i // 2) for i in range(36)], index=ids)
    return views, y, groups


def test_grouped_folds_are_paired_and_original_estimators_unfitted(cohort):
    views, y, groups = cohort
    op = PCAFusion(3)
    result = benchmark_fusion(views, y.iloc[::-1], fusions={"pca": op}, groups=groups, cv=3, seeds=(7,), tune=False)
    assert len(result.scores) == 9
    assert not hasattr(op, "pca_")
    for _, fold in result.splits.groupby(["seed", "fold"]):
        assert set(fold.loc[fold.partition == "train", "group"]).isdisjoint(fold.loc[fold.partition == "test", "group"])
    for _, frame in result.entity_scores.groupby("configuration"):
        assert set(frame.entity_id) == set(y.index)
        assert frame.entity_id.is_unique


def test_preprocessing_cannot_see_test_outlier():
    X = np.column_stack([np.arange(18), np.arange(18) ** 2]).astype(float)
    X[-1] = 1e6
    train, test = np.arange(17), np.array([17])
    adapter = MatrixFusion(PCAFusion(1, normalization="zscore"), ("a",), (2,))
    pipeline, _, _ = _fit_fold(X, np.arange(18.0), train, test, model_name="ridge", transformer=adapter)
    scaler = pipeline.named_steps["features"].fusion_.scalers_["a"]
    np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))


def test_nested_tuning_refits_preprocessing_on_inner_training_rows(cohort, monkeypatch):
    views, y, groups = cohort
    seen = []
    original = ConcatFusion.fit

    def record(self, X, y=None):
        seen.append(next(iter(X.values())).n_entities)
        return original(self, X, y)

    monkeypatch.setattr(ConcatFusion, "fit", record)
    result = benchmark_fusion(
        views,
        y,
        groups=groups,
        fusions={"concat": ConcatFusion()},
        include_single=False,
        cv=3,
        tune=True,
        param_grid={"alpha": [0.1, 1.0]},
    )
    assert len(result.scores) == 3
    assert max(seen) == 24  # outer training only, never all 36 rows
    assert min(seen) < 24  # inner folds refit preprocessing rather than reuse the outer fit
    assert all(row in ({"alpha": 0.1}, {"alpha": 1.0}) for row in result.scores.best_params)


def test_anndata_fusion_integration_and_no_global_pca(cohort):
    views, y, groups = cohort
    adata = AnnData(obs=pd.DataFrame({"target": y, "group": groups}, index=y.index))
    for key, value in views.items():
        adata.obsm[key] = value.matrix
    result = benchmark_embeddings(
        adata,
        "obs_names",
        "genetic",
        target="target",
        obsm_keys=["a", "b"],
        fusion=PCAFusion(3),
        groupby="group",
        models=["ridge"],
    )
    assert result.mse.notna().all()
    assert set(adata.obsm) == {"a", "b"}
    assert adata.uns["benchmark_config"]["fit_scope"] == "training_fold"


def test_retrieval_reports_late_fusion_and_missing_relevance(cohort):
    views, y, groups = cohort
    # No positive candidate can exist for labels unique to each held-out entity.
    labels = pd.Series(y.index, index=y.index)
    result = benchmark_fusion_retrieval(views, labels, groups=groups, cv=3, k=4, fusions={"pca": PCAFusion(3)})
    assert set(result.scores.configuration) == {"single:a", "single:b", "pca", "score_mean", "rank_mean"}
    assert (result.scores.precision_at_k == 0).all()
    assert result.scores.recall_at_k.isna().all()
    assert (result.scores.n_queries_with_relevant == 0).all()


def test_invalid_and_missing_targets_are_not_silently_reordered(cohort):
    views, y, groups = cohort
    with pytest.raises(TypeError, match="indexed"):
        benchmark_fusion(views, y.to_numpy())
    with pytest.raises(ValueError, match="missing"):
        benchmark_fusion(views, y.iloc[:-1])
    with pytest.raises(ValueError, match="seeds"):
        benchmark_fusion(views, y, seeds=())
    with pytest.raises(ValueError, match="positive"):
        benchmark_fusion_retrieval(views, y, k=0)


def test_large_finite_descriptors_do_not_overflow_cosine(cohort):
    views, y, groups = cohort
    large = {
        key: EmbeddingResult(
            value.matrix * 1e30, value.entity_ids, value.entity_type, value.id_scheme, value.provenance
        )
        for key, value in views.items()
    }
    labels = pd.Series(np.where(y > 0, "positive", "negative"), index=y.index)
    kwargs = {"fusions": {"raw": ConcatFusion()}, "include_single": False, "groups": groups, "cv": 3, "k": 3}
    ordinary = benchmark_fusion_retrieval(views, labels, **kwargs)
    scaled = benchmark_fusion_retrieval(large, labels, **kwargs)
    np.testing.assert_allclose(ordinary.entity_scores.precision_at_k, scaled.entity_scores.precision_at_k)
