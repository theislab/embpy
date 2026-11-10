"""Numerical references and supervised leakage contracts for Stage 3."""

import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.tl.benchmark import _fit_fold, benchmark_fusion, benchmark_fusion_retrieval
from embpy.tl.fusion import ConcatMLPFusion, DeepSetsFusion, FusionConfig, GatedFusion, KernelFusion, subset_views
from embpy.tl.fusion._benchmark import MatrixFusion


def view(X, ids=None):
    return EmbeddingResult(
        X, tuple(ids or [str(i) for i in range(len(X))]), "sample", "test", EmbeddingProvenance("test")
    )


@pytest.fixture
def views():
    rng = np.random.default_rng(15)
    return {"a": view(rng.normal(size=(30, 5))), "b": view(rng.normal(size=(30, 4)))}


def test_core_import_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from embpy.tl.fusion import DeepSetsFusion; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_kernel_full_landmarks_matches_exact_weighted_gram(views):
    op = KernelFusion(30, 30, gamma=0.7, weights={"a": 2, "b": 1}, normalization="none", center=False).fit(views)
    Z = op.transform(views).matrix.astype(float)
    exact = (
        2 * rbf_kernel(views["a"].matrix.astype(float), gamma=0.7)
        + rbf_kernel(views["b"].matrix.astype(float), gamma=0.7)
    ) / 3
    assert_allclose(Z @ Z.T, exact, atol=2e-7)
    centered = clone(op).set_params(center=True).fit_transform(views).matrix
    H = np.eye(30) - np.ones((30, 30)) / 30
    assert_allclose(centered @ centered.T, H @ exact @ H, atol=3e-7)


def test_kernel_mixed_tanimoto_unseen_reference():
    bits = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    pair = {"bits": view(bits), "dense": view(bits * [2, 3])}
    op = KernelFusion(
        4, 4, kernels={"bits": "tanimoto", "dense": "rbf"}, normalization="none", gamma=0.5, center=False
    ).fit(pair)
    exact = np.array([[1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0.5, 0.5, 1]])
    exact = (exact + rbf_kernel(bits * [2, 3], gamma=0.5)) / 2
    assert_allclose(op.transform(pair).matrix @ op.transform(pair).matrix.T, exact, atol=2e-7)
    new = {"bits": view([[1, 0]], ["new"]), "dense": view([[4, 5]], ["new"])}
    landmark_bits = op.landmarks_["bits"]
    numerator = np.array([[1, 0]]) @ landmark_bits.T
    tanimoto = numerator / (1 + (landmark_bits * landmark_bits).sum(axis=1) - numerator)
    K = (tanimoto + rbf_kernel([[4, 5]], op.landmarks_["dense"], gamma=0.5)) / 2
    assert_allclose(op.transform(new).matrix, K @ op.projection_, atol=1e-7)
    with pytest.raises(ValueError, match="nonnegative"):
        op.transform({"bits": view([[-1, 0]], ["new"]), "dense": new["dense"]})


def test_kernel_batching_rank_and_fit_invalidation(views):
    op = KernelFusion(8, 15, batch_size=2, normalization="zscore").fit(views)
    assert_allclose(op.transform(views).matrix, clone(op).set_params(batch_size=100).fit_transform(views).matrix)
    assert set(op.landmark_ids_).issubset(views["a"].entity_ids)
    with pytest.raises(ValueError, match="rank"):
        op.fit({"a": view(np.zeros((30, 5))), "b": view(np.zeros((30, 4)))})
    with pytest.raises(NotFittedError):
        op.transform(views)


def test_tanimoto_validates_training_rows_outside_landmarks():
    rows = np.random.default_rng(0).choice(20, 3, replace=False)
    other = next(i for i in range(20) if i not in rows)
    X = np.ones((20, 4))
    X[other, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        KernelFusion(1, 3, kernels="tanimoto", normalization="none", center=False).fit({"bits": view(X)})


@pytest.mark.parametrize("cls", [KernelFusion, DeepSetsFusion, GatedFusion, ConcatMLPFusion])
def test_order_schema_serialization_clone_and_no_mutation(cls, views, tmp_path):
    if cls is not KernelFusion:
        pytest.importorskip("torch")
    kwargs = {} if cls is KernelFusion else {"epochs": 3, "hidden_dim": 8}
    original = {name: v.matrix.copy() for name, v in views.items()}
    target = pd.DataFrame(views["a"].matrix[:, :2], index=views["a"].entity_ids)
    op = cls(3, **kwargs).fit(views, target.iloc[::-1])
    reversed_views = {name: view(v.matrix[::-1], v.entity_ids[::-1]) for name, v in reversed(list(views.items()))}
    assert_allclose(op.transform(reversed_views).matrix[::-1], op.transform(views).matrix, atol=1e-6)
    path = tmp_path / "model.joblib"
    joblib.dump(op, path)
    assert_allclose(joblib.load(path).transform(views).matrix, op.transform(views).matrix)
    assert not hasattr(clone(op), "is_fitted_")
    for name in views:
        assert_allclose(views[name].matrix, original[name])
    with pytest.raises(ValueError, match="view names"):
        op.transform({"a": views["a"]})


@pytest.mark.parametrize("cls", [DeepSetsFusion, GatedFusion, ConcatMLPFusion])
def test_neural_learns_targets_and_restores_rng(cls, views):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    state = torch.random.get_rng_state().clone()
    target = views["a"].matrix[:, :1] * 3 + views["b"].matrix[:, :1]
    op = cls(8, hidden_dim=24, epochs=100, learning_rate=0.01, normalization="zscore").fit(views, target)
    assert torch.equal(state, torch.random.get_rng_state())
    assert op.loss_curve_[-1] < op.loss_curve_[0] * 0.15
    assert np.mean((op.predict(views) - target) ** 2) < 0.3
    assert_allclose(clone(op).fit(views, target).transform(views).matrix, op.transform(views).matrix)
    changed = clone(op).fit(views, target[::-1])
    assert not np.allclose(changed.transform(views).matrix, op.transform(views).matrix)
    if cls is GatedFusion:
        weights = op.view_weights(views)
        assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
        assert (weights.values >= 0).all()
        assert weights.iloc[:, 0].std() > 0.001
    with pytest.raises(ValueError, match="requires numeric"):
        op.fit(views)
    with pytest.raises(NotFittedError):
        op.transform(views)


def test_neural_nested_pipeline_uses_only_training_targets(views, tmp_path):
    pytest.importorskip("torch")
    packed = np.concatenate([v.matrix for v in views.values()], axis=1)
    y = views["a"].matrix[:, :2].copy()
    groups = np.array([str(i // 2) for i in range(30)])
    adapter = MatrixFusion(GatedFusion(3, epochs=2, hidden_dim=8, normalization="zscore"), ("a", "b"), (5, 4))
    train, test = np.arange(24), np.arange(24, 30)
    kwargs = {
        "model_name": "ridge",
        "transformer": adapter,
        "groups": groups,
        "tune": True,
        "param_grid": {"alpha": [0.1, 1]},
        "fusion_param_grid": {"epochs": [2, 3]},
        "memory": str(tmp_path),
    }
    first, _, params = _fit_fold(packed, y, train, test, **kwargs)
    packed[test], y[test] = 1e4, -1e4
    second, _, params2 = _fit_fold(packed, y, train, test, **kwargs)
    assert params == params2
    left, right = [model.named_steps["features"].fusion_ for model in (first, second)]
    assert_allclose(left.target_mean_, right.target_mean_)
    for key, value in left.network_.state_dict().items():
        assert_allclose(value.numpy(), right.network_.state_dict()[key].numpy())


def test_neural_target_alignment_and_retrieval_contract(views):
    pytest.importorskip("torch")
    y = pd.DataFrame(views["a"].matrix[:, :2], index=views["a"].entity_ids)
    op = DeepSetsFusion(3, epochs=2, hidden_dim=8)
    a = clone(op).fit(views, y)
    b = clone(op).fit(views, y.iloc[::-1])
    assert_allclose(a.transform(views).matrix, b.transform(views).matrix)
    with pytest.raises(ValueError, match="unique entity IDs"):
        op.fit(views, y.iloc[:10])
    result = benchmark_fusion(
        views,
        y,
        fusions={"deep": FusionConfig(op, param_grid={"epochs": [1, 2]})},
        include_single=False,
        cv=2,
        param_grid={"alpha": [1.0]},
    )
    assert result.scores.supervised_fusion.all()
    assert (result.scores.trainable_parameters > 0).all()
    with pytest.raises(ValueError, match="label-only"):
        benchmark_fusion_retrieval(views, y.iloc[:, 0], fusions={"deep": op})


def test_neural_inner_join_requires_indexed_targets(views):
    pytest.importorskip("torch")
    pair = {"a": views["a"], "b": subset_views(views, views["a"].entity_ids[:20])["b"]}
    op = DeepSetsFusion(2, hidden_dim=4, epochs=1, join="inner")
    y = pd.Series(np.arange(30), index=views["a"].entity_ids)
    assert op.fit_transform(pair, y).n_entities == 20
    with pytest.raises(ValueError, match="Array targets"):
        op.fit(pair, y.to_numpy())


def test_kernel_retrieval_runs_with_training_only_landmarks(views, monkeypatch):
    seen = []
    original = KernelFusion.fit

    def record(self, X, y=None):
        result = original(self, X, y)
        training_ids = set(next(iter(X.values())).entity_ids)
        assert set(self.landmark_ids_).issubset(training_ids)
        seen.append(training_ids)
        return result

    monkeypatch.setattr(KernelFusion, "fit", record)
    ids = views["a"].entity_ids
    labels = pd.Series([str(i % 3) for i in range(30)], index=ids)
    result = benchmark_fusion_retrieval(
        views,
        labels,
        include_single=False,
        cv=2,
        k=3,
        fusions={"kernel": KernelFusion(3, n_landmarks=8)},
    )
    assert result.scores.query("configuration == 'kernel'").precision_at_k.notna().all()
    assert len(seen) == 2 and all(len(ids) == 15 for ids in seen)
