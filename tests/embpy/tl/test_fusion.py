"""Contract and numerical checks for Stage-1 fusion."""

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import cosine_similarity

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.tl.fusion import (
    ConcatFusion,
    PCAFusion,
    align_views,
    fuse_ranks,
    fuse_scores,
    view_from_store,
    views_from_anndata,
)


def view(matrix, ids=None, scheme="test_id"):
    a = np.asarray(matrix, dtype=np.float32)
    return EmbeddingResult(
        a, tuple(ids or [f"e{i}" for i in range(len(a))]), "gene", scheme, EmbeddingProvenance("test")
    )


@pytest.fixture
def views():
    rng = np.random.default_rng(12)
    return {"a": view(rng.normal(size=(24, 5))), "b": view(rng.normal(size=(24, 3)))}


def test_alignment_is_by_id_and_fitted_view_order(views):
    fit = ConcatFusion().fit(views)
    b = views["b"]
    shuffled = {"b": view(b.matrix[::-1], b.entity_ids[::-1]), "a": views["a"]}
    actual = fit.transform(shuffled)
    assert actual.entity_ids == views["a"].entity_ids
    assert_allclose(actual.matrix, np.column_stack([views["a"].matrix, b.matrix]))


def test_explicit_intersection_and_schema_checks(views):
    short = {**views, "b": view(views["b"].matrix[:-1], views["b"].entity_ids[:-1])}
    with pytest.raises(ValueError, match="different entity IDs"):
        ConcatFusion().fit(short)
    fit = ConcatFusion(join="inner").fit(short)
    assert fit.coverage_["dropped_counts"] == {"a": 1, "b": 0}
    assert fit.transform(short).n_entities == 23
    with pytest.raises(ValueError, match="incompatible"):
        align_views({"a": views["a"], "b": view(views["b"].matrix, scheme="symbol")})
    with pytest.raises(ValueError, match="no common"):
        align_views({"a": view([[1]], ["a"]), "b": view([[2]], ["b"])}, join="inner")


def test_standardization_uses_training_statistics():
    fit = ConcatFusion(normalization="zscore").fit({"a": view([[1, 2], [3, 2]], ["a", "b"])})
    held_out = {"a": view([[101, 7]], ["unseen"])}
    result = fit.transform(held_out)
    assert_allclose(result.matrix, [[99, 5]])
    assert_allclose(fit.scalers_["a"].mean_, [2, 2])
    assert result.entity_ids == ("unseen",)


def test_l2_weighting_matches_weighted_cosine_and_preserves_inputs(views):
    before = {k: v.matrix.copy() for k, v in views.items()}
    result = ConcatFusion(normalization="l2", weights={"a": 3.0, "b": 1.0}).fit_transform(views)
    expected = (3 * cosine_similarity(before["a"]) + cosine_similarity(before["b"])) / 4
    assert_allclose(cosine_similarity(result.matrix), expected, atol=2e-7)
    for name in views:
        assert_allclose(views[name].matrix, before[name])


def test_block_balancing_and_zero_norm():
    views = {"a": view([[0, 0], [0, 0]]), "b": view([[1, 2], [3, 4]])}
    fit = ConcatFusion(normalization={"a": "l2", "b": "zscore"}, balance_blocks=True).fit(views)
    result = fit.transform(views)
    assert np.isfinite(result.matrix).all()
    assert_allclose(np.mean(np.sum(result.matrix[:, 2:] ** 2, axis=1)), 1)
    assert_allclose(result.matrix[:, :2], 0)


@pytest.mark.parametrize("weights", [{"a": 1}, {"a": -1, "b": 1}, {"a": 0, "b": 0}, {"a": np.nan, "b": 1}])
def test_invalid_weights(views, weights):
    with pytest.raises(ValueError, match="weights"):
        ConcatFusion(weights=weights).fit(views)


@pytest.mark.parametrize("norm", ["unknown", {"a": "l2"}])
def test_invalid_normalization(views, norm):
    with pytest.raises(ValueError, match="normalization"):
        ConcatFusion(normalization=norm).fit(views)


def test_fitted_schema_cannot_silently_change(views):
    fit = ConcatFusion().fit(views)
    with pytest.raises(ValueError, match="view names"):
        fit.transform({"a": views["a"]})
    with pytest.raises(ValueError, match="dimension"):
        fit.transform({**views, "b": view(np.ones((24, 4)))})
    with pytest.raises(NotFittedError):
        ConcatFusion().transform(views)


@pytest.mark.parametrize("operator", [ConcatFusion(normalization="zscore"), PCAFusion(3)])
def test_roundtrip_fitted_state_provenance_and_clone(views, operator, tmp_path):
    operator.fit(views)
    result = operator.transform(views)
    json.dumps(result.provenance.to_dict())
    assert result.provenance.extra["fusion"]["training_n_entities"] == 24
    path = tmp_path / "fusion.joblib"
    joblib.dump(operator, path)
    assert_allclose(joblib.load(path).transform(views).matrix, result.matrix)
    assert not hasattr(clone(operator), "view_names_")


def test_numpy_scalar_parameters_have_json_provenance(views):
    fit = PCAFusion(np.int64(3), weights={"a": np.float32(1), "b": np.float32(2)})
    result = fit.fit_transform(views)
    assert json.loads(json.dumps(result.provenance.to_dict()))["extra"]["fusion"]["parameters"]["n_components"] == 3


def test_pca_reuses_training_basis(views):
    fit = PCAFusion(3, normalization="none").fit(views)
    mean = fit.pca_.mean_.copy()
    novel = {k: view(v.matrix[:4] + 30, [f"new{i}" for i in range(4)]) for k, v in views.items()}
    result = fit.transform(novel)
    expected = fit.pca_.transform(np.column_stack([v.matrix for v in novel.values()]))
    assert_allclose(result.matrix, expected, atol=1e-5)
    assert_allclose(fit.pca_.mean_, mean)
    assert result.matrix.shape == (4, 3)
    with pytest.raises(ValueError, match="n_components"):
        PCAFusion(9).fit(views)


def test_anndata_adapter_uses_unique_observation_ids(views):
    adata = AnnData(obs=pd.DataFrame(index=views["a"].entity_ids))
    for key, value in views.items():
        adata.obsm[key] = value.matrix
    adapted = views_from_anndata(adata, ["a", "b"])
    assert adapted["a"].entity_ids == views["a"].entity_ids
    with pytest.raises(ValueError, match="distinct"):
        views_from_anndata(adata, ["a", "a"])
    adata.obs_names = ["duplicate"] * len(adata)
    with pytest.raises(ValueError, match="unique"):
        views_from_anndata(adata, ["a"])


def test_static_store_adapter():
    store = SimpleNamespace(key="stored", entity_type="gene", id_type="ensembl", path="store.emstore")
    store.get = lambda ids, **kwargs: np.asarray([[len(key), 1] for key in ids])
    result = view_from_store(store, ["ENSG2", "ENSG1"])
    assert result.entity_ids == ("ENSG2", "ENSG1")
    assert result.id_scheme == "ensembl"


def test_late_fusion_matches_ids_directions_and_ties():
    a = pd.DataFrame([[9, 4, 4], [1, 2, 3]], index=["q1", "q2"], columns=["a", "b", "c"])
    b = pd.DataFrame([[1, 4, 9], [3, 2, 1]], index=a.index, columns=a.columns)
    shuffled = b.loc[["q2", "q1"], ["c", "a", "b"]]
    assert_allclose(fuse_scores({"a": a, "b": shuffled}, weights={"a": 3, "b": 1}), (3 * a + b) / 4)
    assert_allclose(fuse_scores({"a": a}, higher_is_better=False), -a)
    assert_allclose(fuse_ranks({"a": a}).iloc[0], [-1, -2.5, -2.5])
    assert_allclose(fuse_ranks({"a": a}, higher_is_better=False).iloc[0], [-3, -1.5, -1.5])
    with pytest.raises(ValueError, match="identical"):
        fuse_scores({"a": a, "b": b.drop(columns="b")})
    with pytest.raises(ValueError, match="finite"):
        fuse_ranks({"a": a * np.nan})
    with pytest.raises(ValueError, match="unique"):
        fuse_scores({"a": a.rename(columns={"b": "a"})})
