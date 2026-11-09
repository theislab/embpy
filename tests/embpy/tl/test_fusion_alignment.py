"""Numerical and leakage contracts for Stage-2 alignment operators."""

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.tl.benchmark import benchmark_fusion, benchmark_fusion_retrieval
from embpy.tl.fusion import CCAFusion, FusionConfig, GCCAFusion, ProcrustesFusion


def view(matrix, ids=None):
    return EmbeddingResult(
        matrix, tuple(ids or [f"e{i}" for i in range(len(matrix))]), "sample", "test", EmbeddingProvenance("synthetic")
    )


@pytest.fixture
def views():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(48, 3))
    return {
        name: view(z @ rng.normal(size=(3, width)) + rng.normal(size=(48, width)) * 0.1)
        for name, width in [("a", 7), ("b", 5), ("c", 6)]
    }


@pytest.mark.parametrize("cls", [CCAFusion, GCCAFusion, ProcrustesFusion])
def test_unseen_ids_order_serialization_and_clone(cls, views, tmp_path):
    selected = dict(list(views.items())[:2]) if cls is CCAFusion else views
    before = {name: v.matrix.copy() for name, v in selected.items()}
    operator = cls(3, normalization="zscore").fit(selected)
    reversed_views = {name: view(v.matrix[::-1], v.entity_ids[::-1]) for name, v in reversed(list(selected.items()))}
    assert_allclose(operator.transform(reversed_views).matrix[::-1], operator.transform(selected).matrix, atol=1e-6)
    unseen = {name: view(v.matrix[:2] + 100, ["new1", "new2"]) for name, v in selected.items()}
    expected = sum(
        weight
        * (
            (operator.scalers_[name].transform(v.matrix.astype(float)) - operator.means_[name])
            @ operator.projections_[name]
        )
        for (name, v), weight in zip(unseen.items(), operator.weights_, strict=True)
    )
    assert_allclose(operator.transform(unseen).matrix, expected, rtol=1e-5, atol=1e-5)
    path = tmp_path / "fusion.joblib"
    joblib.dump(operator, path)
    assert_allclose(joblib.load(path).transform(unseen).matrix, operator.transform(unseen).matrix)
    assert not hasattr(clone(operator), "projections_")
    assert operator.n_projection_coefficients_ == sum(v.n_dims * 3 for v in selected.values())
    for name in selected:
        assert_allclose(selected[name].matrix, before[name])
    with pytest.raises(ValueError, match="view names"):
        operator.transform({"wrong": next(iter(selected.values()))})
    with pytest.raises(ValueError):
        operator.fit({name: view(np.zeros_like(v.matrix)) for name, v in selected.items()})
    with pytest.raises(NotFittedError):
        operator.transform(selected)


def test_cca_matches_direct_regularized_whitening(views):
    pair = {key: views[key] for key in ("a", "b")}
    fit = CCAFusion(3, normalization="none", regularization=0.2).fit(pair)
    X, Y = [v.matrix.astype(float) - v.matrix.mean(axis=0, dtype=float) for v in pair.values()]
    n = len(X) - 1
    covariance = []
    for name, block in zip(pair, (X, Y), strict=True):
        eigenvalues, eigenvectors = np.linalg.eigh(block.T @ block / n + fit.ridge_[name] * np.eye(block.shape[1]))
        covariance.append((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T)
    singular = np.linalg.svd(covariance[0] @ (X.T @ Y / n) @ covariance[1], compute_uv=False)
    assert_allclose(fit.regularized_correlations_, singular[:3], atol=1e-9)
    for name, block in zip(pair, (X, Y), strict=True):
        assert_allclose((block @ fit.projections_[name]).std(axis=0, ddof=1), 1, atol=1e-10)
    assert np.min(fit.canonical_correlations_) > 0.95


def test_gcca_matches_direct_maxvar_eigenproblem(views):
    fit = GCCAFusion(3, normalization="none", regularization=0.3).fit(views)
    H = np.zeros((48, 48))
    blocks = {}
    for weight, (name, value) in zip(fit.weights_, views.items(), strict=True):
        X = value.matrix.astype(float) - fit.means_[name]
        blocks[name] = X
        H += weight * X @ np.linalg.solve(X.T @ X + 47 * fit.ridge_[name] * np.eye(X.shape[1]), X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    assert_allclose(fit.consensus_eigenvalues_, eigenvalues[-3:][::-1], atol=1e-10)
    actual = fit.transform(views).matrix.astype(float)
    expected = eigenvectors[:, -3:] * np.sqrt(47)
    assert_allclose(actual @ actual.T, expected @ expected.T, atol=2e-6)
    # Transform's inductive mapping must equal direct evaluation, including training rows.
    direct = sum(w * blocks[name] @ fit.projections_[name] for name, w in zip(views, fit.weights_, strict=True))
    assert_allclose(actual, direct, atol=2e-7)


def test_procrustes_recovers_rotated_geometry():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3)) * [3, 2, 1]
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    views = {"a": view(X), "b": view(X @ rotation + 7)}
    fit = ProcrustesFusion(3, normalization="none", reference="a").fit(views)
    for R in fit.rotations_.values():
        assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert fit.alignment_residuals_["b"] < 1e-6
    result = fit.transform(views).matrix
    centered = X - X.mean(axis=0)
    assert_allclose(result @ result.T, centered @ centered.T, atol=1e-5)


@pytest.mark.parametrize("cls", [CCAFusion, GCCAFusion, ProcrustesFusion])
def test_rank_deficiency_is_supported_without_silent_width_change(cls):
    x = np.linspace(-2, 3, 30)
    pair = {"a": view(np.column_stack([x, x, x * 0])), "b": view(np.column_stack([x * 2, x * 3]))}
    fit = cls(1, normalization="none").fit(pair)
    assert fit.transform(pair).matrix.shape == (30, 1)
    with pytest.raises(ValueError, match="rank"):
        cls(2, normalization="none").fit(pair)


def test_gcca_accepts_complementary_low_rank_views():
    rng = np.random.default_rng(7)
    pair = {"a": view(rng.normal(size=(40, 1))), "b": view(rng.normal(size=(40, 1)))}
    assert GCCAFusion(2, normalization="none").fit_transform(pair).n_dims == 2


@pytest.mark.parametrize("cls", [CCAFusion, GCCAFusion])
def test_regularization_and_pre_pca_validation(cls, views):
    pair = {key: views[key] for key in ("a", "b")}
    fit = cls(2, pre_components=3).fit(pair)
    assert fit.transform(pair).n_dims == 2
    for regularization in (-1, np.nan, {"a": 0.1}):
        with pytest.raises(ValueError, match="regularization"):
            cls(2, regularization=regularization).fit(pair)
    with pytest.raises(ValueError, match="pre_components"):
        cls(2, pre_components=100).fit(pair)
    with pytest.raises(ValueError, match="n_components"):
        cls(True).fit(pair)


def test_two_view_contract(views):
    with pytest.raises(ValueError, match="exactly 2"):
        CCAFusion(2).fit(views)
    with pytest.raises(ValueError, match="at least 2"):
        GCCAFusion(2).fit({"a": views["a"]})
    with pytest.raises(ValueError, match="reference"):
        ProcrustesFusion(2, reference="missing").fit(views)


def test_alignment_benchmarks_select_views_and_tune_inside_grouped_folds(views, tmp_path):
    ids = views["a"].entity_ids
    target = pd.DataFrame(views["a"].matrix[:, :3], index=ids)
    groups = pd.Series([str(i // 2) for i in range(len(ids))], index=ids)
    labels = pd.Series([str(i % 3) for i in range(len(ids))], index=ids)
    operator = CCAFusion(2, pre_components=3)
    config = {"pair": FusionConfig(operator, ("a", "b"), {"regularization": [0.01, 1.0]})}
    result = benchmark_fusion(
        views,
        target,
        fusions=config,
        groups=groups,
        include_single=False,
        cv=3,
        param_grid={"alpha": [0.1, 1.0]},
        cache_dir=str(tmp_path),
    )
    assert result.scores.output_dim.eq(2).all()
    assert all(row["fusion__regularization"] in [0.01, 1.0] for row in result.scores.best_params)
    assert result.config["configurations"]["pair"]["views"] == ["a", "b"]
    assert not hasattr(operator, "projections_")
    for _, split in result.splits.groupby(["seed", "fold"]):
        assert not set(split.loc[split.partition == "train", "group"]) & set(
            split.loc[split.partition == "test", "group"]
        )
    retrieval = benchmark_fusion_retrieval(
        views, labels, fusions=config, groups=groups, include_single=False, tune=True, cv=3, k=3
    )
    assert retrieval.splits.equals(result.splits)
    rows = retrieval.scores.query("configuration == 'pair'")
    assert all(row["regularization"] in [0.01, 1.0] for row in rows.best_params)
    assert rows.inner_precision_at_k.notna().all()
    with pytest.raises(ValueError, match="tune=True"):
        benchmark_fusion(views, target, fusions=config, tune=False)
    with pytest.raises(ValueError, match="tune=True"):
        benchmark_fusion_retrieval(views, labels, fusions=config)


def test_outer_test_changes_cannot_affect_nested_selection(views):
    from embpy.tl.benchmark import _fit_fold, _tune_retrieval
    from embpy.tl.fusion._benchmark import MatrixFusion

    ids = views["a"].entity_ids
    pair = {key: views[key] for key in ("a", "b")}
    packed = np.concatenate([v.matrix for v in pair.values()], axis=1).astype(float)
    target = views["c"].matrix[:, :2].copy()
    groups = np.array([str(i // 2) for i in range(48)])
    train, test = np.arange(36), np.arange(36, 48)
    operator = CCAFusion(2, pre_components=3, normalization="zscore")
    adapter = MatrixFusion(operator, tuple(pair), tuple(v.n_dims for v in pair.values()))
    kwargs = {
        "model_name": "ridge",
        "transformer": adapter,
        "groups": groups,
        "tune": True,
        "param_grid": {"alpha": [0.1, 1.0]},
        "fusion_param_grid": {"regularization": [0.01, 1.0]},
    }
    fitted, _, params = _fit_fold(packed, target, train, test, **kwargs)
    changed, different_target = packed.copy(), target.copy()
    changed[test], different_target[test] = 1e6, -1e6
    other, _, other_params = _fit_fold(changed, different_target, train, test, **kwargs)
    assert params == other_params
    for name in pair:
        assert_allclose(
            fitted.named_steps["features"].fusion_.projections_[name],
            other.named_steps["features"].fusion_.projections_[name],
        )
    labels = np.array([str(i % 3) for i in range(48)])
    choice = _tune_retrieval(operator, {"regularization": [0.01, 1.0]}, pair, ids, labels, train, groups, 0, 3)
    labels[test] = "x"
    altered_views = {name: view(np.where(np.arange(48)[:, None] >= 36, 1e6, v.matrix)) for name, v in pair.items()}
    assert choice == _tune_retrieval(
        operator, {"regularization": [0.01, 1.0]}, altered_views, ids, labels, train, groups, 0, 3
    )


def test_inner_retrieval_fits_exclude_outer_entities_and_refit(views, monkeypatch):
    from embpy.tl.benchmark import _tune_retrieval

    ids = views["a"].entity_ids
    seen = []
    original = CCAFusion.fit

    def spy(self, X, y=None):
        seen.append(set(X["a"].entity_ids))
        return original(self, X, y)

    monkeypatch.setattr(CCAFusion, "fit", spy)
    _tune_retrieval(
        CCAFusion(2),
        {"regularization": [0.1, 1.0]},
        {k: views[k] for k in ("a", "b")},
        ids,
        np.array([str(i % 3) for i in range(48)]),
        np.arange(36),
        np.array([str(i // 2) for i in range(48)]),
        0,
        3,
    )
    assert len(seen) == 6
    for fitted_ids in seen:
        assert len(fitted_ids) == 24
        assert fitted_ids < set(ids[:36])
        for i in range(0, 36, 2):
            assert (ids[i] in fitted_ids) == (ids[i + 1] in fitted_ids)


def test_anndata_benchmark_accepts_stage_two(views):
    from anndata import AnnData

    from embpy.tl.benchmark import benchmark_embeddings

    ids = views["a"].entity_ids
    adata = AnnData(views["c"].matrix, obs=pd.DataFrame(index=ids))
    adata.obsm["a"], adata.obsm["b"] = views["a"].matrix, views["b"].matrix
    result = benchmark_embeddings(
        adata, "obs_names", "genetic", obsm_keys=["a", "b"], fusion=CCAFusion(2), models=["ridge"], mode="quick"
    )
    assert result.mse.notna().all()
    assert set(adata.obsm) == {"a", "b"}


def test_randomized_gcca_solve_matches_dense_shared_subspace():
    # Exercise the production-size branch (combined rank > 256) on small synthetic data.
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(300, 5))
    views = {str(i): view(latent @ rng.normal(size=(5, 100)) + 0.05 * rng.normal(size=(300, 100))) for i in range(3)}
    fit = GCCAFusion(5, normalization="none", regularization=0.5, random_state=4).fit(views)
    result = fit.transform(views).matrix.astype(float)
    H = np.zeros((300, 300))
    for name, value in views.items():
        X = value.matrix.astype(float) - fit.means_[name]
        H += X @ np.linalg.solve(X.T @ X + 299 * fit.ridge_[name] * np.eye(100), X.T) / 3
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    expected = eigenvectors[:, -5:] * np.sqrt(299)
    assert_allclose(fit.consensus_eigenvalues_, eigenvalues[-5:][::-1], atol=1e-8)
    assert_allclose(result @ result.T, expected @ expected.T, atol=3e-6)
    repeated = clone(fit).fit_transform(views)
    assert_allclose(result, repeated.matrix, atol=1e-7)
