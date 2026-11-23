"""Tests for representation-alignment metrics (TSI, QSI, linear CKA, mutual kNN).

The two load-bearing tests here are:

* ``TestOutlierRobustness`` -- a pure rotation corrupted with 2% extreme rows keeps
  TSI high while linear CKA collapses. This is the module's reason for existing:
  single-cell data always contains such rows, so a CKA-based layer sweep can be
  driven to a confidently wrong answer by a handful of bad cells.
* ``TestExactMatchesNaiveDefinition`` -- the O(N^2 log N) Kendall-tau path must be
  bit-identical to the naive O(N^3) / O(N^4) definitions, including with ties.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from embpy.tl import (
    alignment_matrix,
    alignment_profile,
    block_to_attention_index,
    block_to_hidden_state_index,
    cca_sharing,
    cknna,
    linear_cka,
    mutual_knn,
    procrustes_similarity,
    qsi,
    rank_layers,
    sample_size_for,
    tsi,
)

# ---------------------------------------------------------------------------
# naive reference implementations (definitions straight from the paper)
# ---------------------------------------------------------------------------


def naive_tsi(DX: np.ndarray, DY: np.ndarray) -> float:
    n = DX.shape[0]
    agree = total = 0
    for i in range(n):
        for j, k in combinations([x for x in range(n) if x != i], 2):
            if np.sign(DX[i, j] - DX[i, k]) == np.sign(DY[i, j] - DY[i, k]):
                agree += 1
            total += 1
    return agree / total


def naive_qsi(DX: np.ndarray, DY: np.ndarray) -> float:
    pairs = list(combinations(range(DX.shape[0]), 2))
    agree = total = 0
    for (i, j), (k, latt) in combinations(pairs, 2):
        if np.sign(DX[i, j] - DX[k, latt]) == np.sign(DY[i, j] - DY[k, latt]):
            agree += 1
        total += 1
    return agree / total


def dmat(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(X), checks=False)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------


class TestExactMatchesNaiveDefinition:
    """The Kendall-tau reduction must reproduce the literal triplet/quadruplet counts."""

    def test_tsi_matches_naive_continuous(self, rng):
        X, Y = rng.normal(size=(25, 6)), rng.normal(size=(25, 6))
        assert tsi(X, Y, method="exact") == pytest.approx(naive_tsi(dmat(X), dmat(Y)), abs=1e-12)

    def test_qsi_matches_naive_continuous(self, rng):
        X, Y = rng.normal(size=(20, 5)), rng.normal(size=(20, 5))
        assert qsi(X, Y, method="exact") == pytest.approx(naive_qsi(dmat(X), dmat(Y)), abs=1e-12)

    def test_tsi_matches_naive_with_many_ties(self, rng):
        """Rounded data creates tied distances; ties must be counted exactly."""
        X, Y = np.round(rng.normal(size=(22, 3))), np.round(rng.normal(size=(22, 3)))
        DX, DY = np.round(dmat(X), 1), np.round(dmat(Y), 1)
        # feed the *same* rounded distances to both paths via a precomputed metric
        got = np.mean([_row_concordance(DX[i][np.arange(22) != i], DY[i][np.arange(22) != i]) for i in range(22)])
        assert got == pytest.approx(naive_tsi(DX, DY), abs=1e-12)

    def test_qsi_matches_naive_with_many_ties(self, rng):
        X, Y = np.round(rng.normal(size=(16, 3))), np.round(rng.normal(size=(16, 3)))
        DX, DY = np.round(dmat(X), 1), np.round(dmat(Y), 1)
        got = _row_concordance(squareform(DX, checks=False), squareform(DY, checks=False))
        assert got == pytest.approx(naive_qsi(DX, DY), abs=1e-12)


def _row_concordance(a, b):
    from embpy.tl.alignment import _concordance

    return _concordance(np.asarray(a), np.asarray(b))


class TestInvariances:
    """Both families are invariant to translation, isotropic scale and rotation."""

    @pytest.fixture
    def X(self, rng):
        return rng.normal(size=(120, 12))

    def test_identical_is_one(self, X):
        assert tsi(X, X) == pytest.approx(1.0)
        assert qsi(X, X) == pytest.approx(1.0)
        assert linear_cka(X, X) == pytest.approx(1.0)

    def test_rotation_invariant(self, X, rng):
        Q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        assert tsi(X, X @ Q) == pytest.approx(1.0, abs=1e-9)
        assert qsi(X, X @ Q) == pytest.approx(1.0, abs=1e-9)
        assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-9)

    def test_isotropic_scale_invariant(self, X):
        assert tsi(X, X * 7.0) == pytest.approx(1.0, abs=1e-9)
        assert linear_cka(X, X * 7.0) == pytest.approx(1.0, abs=1e-9)

    def test_translation_invariant(self, X):
        assert tsi(X, X + 3.5) == pytest.approx(1.0, abs=1e-9)
        assert linear_cka(X, X + 3.5) == pytest.approx(1.0, abs=1e-9)

    def test_unrelated_sits_at_the_null(self, X, rng):
        """TSI's null is a fixed 0.5 -- that is the point of an ordinal metric."""
        Y = rng.normal(size=(120, 12))
        assert tsi(X, Y) == pytest.approx(0.5, abs=0.05)
        assert qsi(X, Y) == pytest.approx(0.5, abs=0.05)


class TestOutlierRobustness:
    """The decisive property for single-cell data.

    Dying cells, doublets and ambient-RNA artefacts produce a few extreme rows. CKA
    is a ratio of Frobenius norms, so those rows dominate it; the ordinal metrics
    only see distance *orderings* and barely move.
    """

    def test_rotation_with_two_percent_outliers(self):
        rng = np.random.default_rng(0)
        n, d = 400, 32
        X = rng.normal(size=(n, d))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y = X @ Q  # true similarity is exactly 1.0

        n_bad = max(1, int(0.02 * n))
        bad = rng.choice(n, size=n_bad, replace=False)
        Y = Y.copy()
        Y[bad] += rng.normal(scale=200.0, size=(n_bad, d))

        tsi_score = tsi(X, Y, method="exact")
        cka_score = linear_cka(X, Y)

        assert tsi_score >= 0.9, f"TSI collapsed to {tsi_score:.3f} on 2% outliers"
        assert cka_score < 0.5, f"CKA unexpectedly survived at {cka_score:.3f}"
        assert tsi_score > cka_score


class TestSamplingEstimator:
    def test_sample_size_formula(self):
        # n = ceil(log(2/delta) / (2 eps^2))
        assert sample_size_for(0.05, 0.05) == 738
        assert sample_size_for(0.02, 0.05) == 4612
        assert sample_size_for(0.01, 0.05) == 18445

    def test_sample_size_is_independent_of_n(self):
        assert sample_size_for(0.05, 0.05) == sample_size_for(0.05, 0.05)

    @pytest.mark.parametrize("epsilon", [0.05, 0.02])
    def test_approx_lands_within_epsilon_of_exact(self, epsilon):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(150, 8))
        Y = 0.5 * X + 0.5 * rng.normal(size=(150, 8))  # partially aligned
        exact = tsi(X, Y, method="exact")
        for seed in range(10):
            approx = tsi(X, Y, method="approx", epsilon=epsilon, delta=0.05, random_state=seed)
            assert abs(approx - exact) <= epsilon + 0.02, (
                f"seed={seed}: |{approx:.4f} - {exact:.4f}| exceeds eps={epsilon}"
            )

    def test_qsi_approx_close_to_exact(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 8))
        Y = 0.5 * X + 0.5 * rng.normal(size=(120, 8))
        exact = qsi(X, Y, method="exact")
        approx = qsi(X, Y, method="approx", n_samples=20000, random_state=0)
        assert abs(approx - exact) < 0.03

    def test_n_samples_and_epsilon_are_mutually_exclusive(self, rng):
        X, Y = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
        with pytest.raises(ValueError, match="not both"):
            tsi(X, Y, method="approx", n_samples=100, epsilon=0.05, delta=0.05)

    def test_sampled_triplet_indices_are_distinct(self):
        """i, j, k must be three different entities or the estimate is biased."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 5))
        # a self-comparison must still be exactly 1.0 under sampling
        assert tsi(X, X, method="approx", n_samples=5000, random_state=0) == pytest.approx(1.0)


class TestMethodSelection:
    def test_auto_uses_exact_for_small_n(self, rng):
        X, Y = rng.normal(size=(60, 5)), rng.normal(size=(60, 5))
        assert tsi(X, Y, method="auto") == pytest.approx(tsi(X, Y, method="exact"))

    def test_invalid_method_raises(self, rng):
        X, Y = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        with pytest.raises(ValueError, match="method must be"):
            tsi(X, Y, method="batched")  # explicitly unsupported: no guarantees


class TestCustomDistance:
    """Metrics must not hardcode Euclidean -- cosine/correlation are routine here."""

    @pytest.mark.parametrize("metric", ["cosine", "correlation", "cityblock"])
    def test_string_metrics_accepted(self, metric, rng):
        X, Y = rng.normal(size=(40, 6)), rng.normal(size=(40, 6))
        val = tsi(X, Y, metric=metric)
        assert 0.0 <= val <= 1.0

    def test_callable_metric_accepted(self, rng):
        X, Y = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))

        def chebyshev(u, v):
            return float(np.max(np.abs(u - v)))

        val = tsi(X, Y, metric=chebyshev)
        assert val == pytest.approx(tsi(X, Y, metric="chebyshev"), abs=1e-12)

    def test_metric_choice_changes_the_answer(self, rng):
        X, Y = rng.normal(size=(50, 6)), rng.normal(size=(50, 6))
        assert tsi(X, Y, metric="euclidean") != tsi(X, Y, metric="cosine")


class TestMutualKnn:
    def test_identical_is_one(self, rng):
        X = rng.normal(size=(60, 5))
        assert mutual_knn(X, X, k=5) == pytest.approx(1.0)

    def test_unrelated_is_near_chance(self, rng):
        X, Y = rng.normal(size=(200, 5)), rng.normal(size=(200, 5))
        assert mutual_knn(X, Y, k=10) < 0.2  # chance ~ k/(n-1) = 0.05

    def test_invalid_k_raises(self, rng):
        X, Y = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        with pytest.raises(ValueError, match="k must be in"):
            mutual_knn(X, Y, k=50)


class TestAlignmentMatrix:
    def test_shape_labels_and_diagonal(self, rng):
        embs = {"a": rng.normal(size=(40, 5)), "b": rng.normal(size=(40, 7)), "c": rng.normal(size=(40, 3))}
        df = alignment_matrix(embs)
        assert list(df.index) == ["a", "b", "c"]
        assert list(df.columns) == ["a", "b", "c"]
        assert np.allclose(np.diag(df.values), 1.0)

    def test_symmetric(self, rng):
        embs = {"a": rng.normal(size=(30, 4)), "b": rng.normal(size=(30, 4))}
        df = alignment_matrix(embs)
        assert df.loc["a", "b"] == pytest.approx(df.loc["b", "a"])

    def test_translation_invariance_shows_as_one(self, rng):
        X = rng.normal(size=(50, 8))
        df = alignment_matrix({"a": X, "b": X + 1.0})
        assert df.loc["a", "b"] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("metric", ["tsi", "qsi", "cka", "linear_cka", "mutual_knn", "procrustes", "cca", "cknna"])
    def test_all_metrics_supported(self, metric, rng):
        embs = {"a": rng.normal(size=(40, 5)), "b": rng.normal(size=(40, 5))}
        df = alignment_matrix(embs, metric=metric)
        assert df.shape == (2, 2)

    def test_distance_is_forwarded_to_the_metric(self, rng):
        embs = {"a": rng.normal(size=(40, 5)), "b": rng.normal(size=(40, 5))}
        euc = alignment_matrix(embs, metric="tsi", distance="euclidean").loc["a", "b"]
        cos = alignment_matrix(embs, metric="tsi", distance="cosine").loc["a", "b"]
        assert euc != cos
        assert euc == pytest.approx(tsi(embs["a"], embs["b"], metric="euclidean"))
        assert cos == pytest.approx(tsi(embs["a"], embs["b"], metric="cosine"))

    def test_cka_ignores_distance_without_erroring(self, rng):
        """linear_cka takes no distance argument; it must not receive one."""
        embs = {"a": rng.normal(size=(40, 5)), "b": rng.normal(size=(40, 5))}
        df = alignment_matrix(embs, metric="cka", distance="cosine")
        assert df.loc["a", "b"] == pytest.approx(linear_cka(embs["a"], embs["b"]))

    def test_keys_subset_and_order(self, rng):
        embs = {"a": rng.normal(size=(30, 4)), "b": rng.normal(size=(30, 4)), "c": rng.normal(size=(30, 4))}
        df = alignment_matrix(embs, keys=["c", "a"])
        assert list(df.index) == ["c", "a"]

    def test_mismatched_row_counts_raise(self, rng):
        embs = {"a": rng.normal(size=(30, 4)), "b": rng.normal(size=(29, 4))}
        with pytest.raises(ValueError, match="same entities"):
            alignment_matrix(embs)

    def test_unknown_metric_raises(self, rng):
        embs = {"a": rng.normal(size=(20, 4)), "b": rng.normal(size=(20, 4))}
        with pytest.raises(ValueError, match="Unknown metric"):
            alignment_matrix(embs, metric="svcca")


class TestInputValidation:
    def test_row_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="same entities"):
            tsi(rng.normal(size=(20, 4)), rng.normal(size=(19, 4)))

    def test_non_2d_raises(self, rng):
        with pytest.raises(ValueError, match="must be 2-D"):
            tsi(rng.normal(size=20), rng.normal(size=(20, 4)))

    def test_nan_raises(self, rng):
        X = rng.normal(size=(20, 4))
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            tsi(X, rng.normal(size=(20, 4)))

    def test_too_few_entities_raises(self, rng):
        with pytest.raises(ValueError, match="At least 3"):
            tsi(rng.normal(size=(2, 4)), rng.normal(size=(2, 4)))


class TestLayerIndexConventions:
    """extract_hidden_states and extract_attention do NOT share an index convention."""

    def test_block_to_hidden_state_is_offset_by_one(self):
        # hidden states: index 0 is the embedding layer, so block b is at b+1
        assert block_to_hidden_state_index(0) == 1
        assert block_to_hidden_state_index(5) == 6

    def test_block_to_attention_is_identity(self):
        # attention: one entry per block, no embedding entry
        assert block_to_attention_index(0) == 0
        assert block_to_attention_index(5) == 5

    def test_the_two_conventions_differ(self):
        """The off-by-one this guards against."""
        assert block_to_hidden_state_index(3) != block_to_attention_index(3)

    @pytest.mark.parametrize("fn", [block_to_hidden_state_index, block_to_attention_index])
    def test_negative_block_raises(self, fn):
        with pytest.raises(ValueError, match="non-negative"):
            fn(-1)


class TestRankLayers:
    @pytest.fixture
    def layers(self):
        """4 layers that progressively rotate away from layer 0."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=(80, 6))
        out = {0: base}
        cur = base
        for i in range(1, 4):
            cur = cur + 0.6 * rng.normal(size=base.shape)
            out[i] = cur
        return out

    def test_returns_row_per_layer_indexed_by_layer(self, layers):
        df = rank_layers(layers)
        assert list(df.index) == [0, 1, 2, 3]
        assert df.index.name == "layer"

    def test_structural_columns_without_target(self, layers):
        df = rank_layers(layers)
        assert list(df.columns) == ["to_final", "to_previous"]

    def test_probe_and_target_columns_with_target(self, layers):
        rng = np.random.default_rng(1)
        y = layers[3] @ rng.normal(size=(6,))  # predictable from the last layer
        df = rank_layers(layers, target=y)
        assert {"probe_score", "to_final", "to_target", "to_previous"} <= set(df.columns)

    def test_final_layer_aligns_perfectly_with_itself(self, layers):
        df = rank_layers(layers)
        assert df.loc[3, "to_final"] == pytest.approx(1.0)

    def test_first_layer_has_no_previous(self, layers):
        df = rank_layers(layers)
        assert np.isnan(df.loc[0, "to_previous"])

    def test_alignment_to_final_increases_with_depth(self, layers):
        """Later layers are progressively closer to the final representation."""
        df = rank_layers(layers)
        assert df.loc[0, "to_final"] < df.loc[2, "to_final"]

    def test_probe_recovers_a_learnable_target(self, layers):
        rng = np.random.default_rng(2)
        y = layers[0] @ rng.normal(size=(6,))  # linear in layer 0
        df = rank_layers(layers, target=y)
        assert df.loc[0, "probe_score"] > 0.9

    def test_classification_target_uses_accuracy(self, layers):
        y = np.array(["a", "b"] * 40)
        df = rank_layers(layers, target=y)
        assert (df["probe_score"] >= 0.0).all() and (df["probe_score"] <= 1.0).all()

    def test_layers_subset(self, layers):
        df = rank_layers(layers, layers=[0, 2])
        assert list(df.index) == [0, 2]

    def test_metric_is_configurable(self, layers):
        a = rank_layers(layers, metric="tsi").loc[0, "to_final"]
        b = rank_layers(layers, metric="cka").loc[0, "to_final"]
        assert a != b

    def test_target_length_mismatch_raises(self, layers):
        with pytest.raises(ValueError, match="target has"):
            rank_layers(layers, target=np.zeros(5))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="nothing to rank"):
            rank_layers({})

    def test_missing_layer_raises(self, layers):
        with pytest.raises(KeyError, match="not present"):
            rank_layers(layers, layers=[0, 99])

    def test_wrapper_form_requires_inputs(self):
        class FakeWrapper:
            def embed_all_layers(self, ids):
                return {0: np.zeros((5, 3))}

        with pytest.raises(ValueError, match="also pass inputs"):
            rank_layers(FakeWrapper())

    def test_wrapper_form_calls_embed_all_layers(self):
        rng = np.random.default_rng(3)
        made = {i: rng.normal(size=(40, 5)) for i in range(3)}

        class FakeWrapper:
            def embed_all_layers(self, ids):
                assert ids == "TOKENS"
                return made

        df = rank_layers(FakeWrapper(), inputs="TOKENS")
        assert list(df.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# scales: TSI / QSI / Procrustes at neighbourhood size k
# ---------------------------------------------------------------------------


def _naive_neighbours(D: np.ndarray, i: int, k: int) -> list[int]:
    """The k nearest of i, ties broken by index -- the documented convention."""
    others = [x for x in range(D.shape[0]) if x != i]
    return sorted(others, key=lambda x: (D[i, x], x))[:k]


def naive_tsi_k(DX: np.ndarray, DY: np.ndarray, k: int, source: str) -> float:
    D = DX if source == "x" else DY
    per_anchor = []
    for i in range(D.shape[0]):
        agree = total = 0
        for j, m in combinations(_naive_neighbours(D, i, k), 2):
            agree += np.sign(DX[i, j] - DX[i, m]) == np.sign(DY[i, j] - DY[i, m])
            total += 1
        per_anchor.append(agree / total)
    return float(np.mean(per_anchor))


def naive_qsi_k(DX: np.ndarray, DY: np.ndarray, k: int, source: str) -> float:
    D = DX if source == "x" else DY
    pairs = sorted({tuple(sorted((i, j))) for i in range(D.shape[0]) for j in _naive_neighbours(D, i, k)})
    agree = total = 0
    for (i, j), (a, b) in combinations(pairs, 2):
        agree += np.sign(DX[i, j] - DX[a, b]) == np.sign(DY[i, j] - DY[a, b])
        total += 1
    return agree / total


class TestScales:
    """``k`` restricts every comparison to close neighbourhoods; ``k = n - 1`` is global."""

    @pytest.mark.parametrize("source", ["x", "y"])
    def test_tsi_at_k_matches_naive(self, source, rng):
        X, Y = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        got = tsi(X, Y, k=6, neighbors=source, method="exact")
        assert got == pytest.approx(naive_tsi_k(dmat(X), dmat(Y), 6, source), abs=1e-12)

    @pytest.mark.parametrize("source", ["x", "y"])
    def test_qsi_at_k_matches_naive(self, source, rng):
        X, Y = rng.normal(size=(16, 3)), rng.normal(size=(16, 3))
        got = qsi(X, Y, k=4, neighbors=source, method="exact")
        assert got == pytest.approx(naive_qsi_k(dmat(X), dmat(Y), 4, source), abs=1e-12)

    @pytest.mark.parametrize("fn", [tsi, qsi])
    def test_both_is_the_mean_of_the_two_sources(self, fn, rng):
        X, Y = rng.normal(size=(40, 5)), rng.normal(size=(40, 5))
        x, y = fn(X, Y, k=8, neighbors="x"), fn(X, Y, k=8, neighbors="y")
        assert fn(X, Y, k=8, neighbors="both") == pytest.approx((x + y) / 2)

    @pytest.mark.parametrize("fn", [tsi, qsi])
    @pytest.mark.parametrize("source", ["x", "y", "both"])
    def test_k_all_is_the_global_value(self, fn, source, rng):
        X, Y = rng.normal(size=(30, 5)), rng.normal(size=(30, 5))
        assert fn(X, Y, k=29, neighbors=source, method="exact") == pytest.approx(fn(X, Y, method="exact"), abs=1e-12)

    @pytest.mark.parametrize("fn", [tsi, qsi, procrustes_similarity])
    def test_rotation_invariant_at_small_k(self, fn, rng):
        X = rng.normal(size=(60, 6))
        Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        assert fn(X, 2.0 * X @ Q + 1.0, k=7) == pytest.approx(1.0, abs=1e-9)

    def test_unrelated_sits_at_the_null_at_small_k(self, rng):
        X, Y = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
        assert abs(tsi(X, Y, k=10) - 0.5) < 0.03
        assert abs(qsi(X, Y, k=10) - 0.5) < 0.03

    def test_sampled_local_tsi_lands_near_exact(self, rng):
        X = rng.normal(size=(200, 5))
        Y = X + rng.normal(scale=0.7, size=(200, 5))
        exact = tsi(X, Y, k=20, method="exact")
        approx = tsi(X, Y, k=20, method="approx", n_samples=40_000, random_state=1)
        assert abs(exact - approx) < 0.015

    def test_sampled_local_qsi_lands_near_exact(self, rng):
        X = rng.normal(size=(150, 5))
        Y = X + rng.normal(scale=0.7, size=(150, 5))
        exact = qsi(X, Y, k=15, method="exact")
        approx = qsi(X, Y, k=15, method="approx", n_samples=40_000, random_state=1)
        assert abs(exact - approx) < 0.015

    def test_local_structure_survives_a_warp_that_breaks_global_order(self, rng):
        """A fold keeps close neighbourhoods but flips which far pairs are farther."""
        t = np.sort(rng.uniform(0, 1, size=200))
        X = np.c_[t, np.zeros_like(t)]
        Y = np.c_[np.cos(1.8 * np.pi * t), np.sin(1.8 * np.pi * t)]  # 1-D line bent into an arc
        assert tsi(X, Y, k=5) > tsi(X, Y) + 0.05

    @pytest.mark.parametrize("fn", [tsi, procrustes_similarity])
    def test_k_below_two_raises(self, fn, rng):
        X, Y = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
        with pytest.raises(ValueError, match="k must be in"):
            fn(X, Y, k=1)

    @pytest.mark.parametrize("fn", [tsi, qsi, procrustes_similarity])
    def test_k_above_n_minus_one_raises(self, fn, rng):
        X, Y = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
        with pytest.raises(ValueError, match="k must be in"):
            fn(X, Y, k=20)

    def test_invalid_neighbors_raises(self, rng):
        X, Y = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
        with pytest.raises(ValueError, match="neighbors must be"):
            tsi(X, Y, k=5, neighbors="z")


class TestProcrustes:
    def test_rotation_scale_and_translation_give_one(self, rng):
        X = rng.normal(size=(80, 7))
        Q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        assert procrustes_similarity(X, 0.3 * X @ Q - 2.0) == pytest.approx(1.0, abs=1e-12)

    def test_unrelated_is_low(self, rng):
        X, Y = rng.normal(size=(400, 5)), rng.normal(size=(400, 5))
        assert procrustes_similarity(X, Y) < 0.2

    def test_bounded_with_different_dimensions(self, rng):
        X, Y = rng.normal(size=(50, 3)), rng.normal(size=(50, 11))
        assert 0.0 <= procrustes_similarity(X, Y) <= 1.0

    def test_a_shared_subspace_scores_its_variance_share(self, rng):
        """Shared coordinates carry half the variance, so the similarity is about 0.5."""
        z = rng.normal(size=(4000, 4))
        X = np.hstack([z, rng.normal(size=(4000, 4))])
        Y = np.hstack([z, rng.normal(size=(4000, 4))])
        assert procrustes_similarity(X, Y) == pytest.approx(0.5, abs=0.03)

    def test_row_space_path_matches_the_direct_svd(self, rng):
        from embpy.tl.alignment import _nuclear_cross

        A, B = rng.normal(size=(12, 40)), rng.normal(size=(12, 30))  # features >> rows
        direct = np.linalg.svd(A.T @ B, compute_uv=False).sum()
        assert _nuclear_cross(A, B) == pytest.approx(direct, rel=1e-10)

    def test_local_at_k_all_is_global(self, rng):
        X, Y = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
        assert procrustes_similarity(X, Y, k=29) == pytest.approx(procrustes_similarity(X, Y))

    def test_anchor_sampling_is_reproducible(self, rng):
        X, Y = rng.normal(size=(100, 4)), rng.normal(size=(100, 4))
        a = procrustes_similarity(X, Y, k=10, n_anchors=20, random_state=3)
        b = procrustes_similarity(X, Y, k=10, n_anchors=20, random_state=3)
        assert a == b


class TestCCASharing:
    @pytest.fixture
    def planted(self, rng):
        """Three shared directions, five private ones per view, each view rotated."""
        n = 400
        z = rng.normal(size=(n, 3))
        X = np.hstack([z, rng.normal(size=(n, 5))]) @ np.linalg.qr(rng.normal(size=(8, 8)))[0]
        Y = np.hstack([z, rng.normal(size=(n, 5))]) @ np.linalg.qr(rng.normal(size=(8, 8)))[0]
        return X + 0.05 * rng.normal(size=X.shape), Y + 0.05 * rng.normal(size=Y.shape)

    def test_recovers_the_planted_shared_directions(self, planted):
        df = cca_sharing(*planted, pre_components=8, n_permutations=19)
        assert (df["r"].iloc[:3] > 0.9).all()
        assert (df["r"].iloc[3:] < 0.3).all()
        assert df.attrs["n_significant"] == 3
        assert df.attrs["sum_r2"] == pytest.approx(3.0, abs=0.3)

    def test_unrelated_views_share_nothing(self, rng):
        X, Y = rng.normal(size=(300, 6)), rng.normal(size=(300, 6))
        df = cca_sharing(X, Y, pre_components=6)
        assert df.attrs["sum_r2"] < 0.3

    def test_held_out_is_below_in_sample_when_overfitting(self, rng):
        """With many dimensions per row, training correlations inflate; held-out ones do not."""
        X, Y = rng.normal(size=(80, 30)), rng.normal(size=(80, 30))
        df = cca_sharing(X, Y, pre_components=30, regularization=0.0)
        assert df["r_train"].iloc[0] > 0.8
        assert df["r"].iloc[0] < 0.4

    def test_groups_keep_each_group_in_one_fold(self, planted):
        groups = np.repeat(np.arange(40), 10)
        df = cca_sharing(*planted, pre_components=8, groups=groups)
        assert df.attrs["n_folds"] == 5
        assert (df["r"].iloc[:3] > 0.9).all()

    def test_n_components_truncates(self, planted):
        assert len(cca_sharing(*planted, pre_components=8, n_components=4)) == 4

    def test_no_permutations_leaves_significance_unset(self, planted):
        df = cca_sharing(*planted, pre_components=8)
        assert df.attrs["n_significant"] is None
        assert "significant" not in df.columns

    def test_invalid_arguments_raise(self, planted):
        with pytest.raises(ValueError, match="cv must be"):
            cca_sharing(*planted, cv=1)
        with pytest.raises(ValueError, match="groups has"):
            cca_sharing(*planted, groups=np.zeros(3))


class TestAlignmentProfile:
    @pytest.fixture
    def pair(self, rng):
        X = rng.normal(size=(120, 6))
        return X, X + rng.normal(scale=0.8, size=X.shape)

    def test_default_grid_ends_at_global(self, pair):
        df = alignment_profile(*pair, cca=False, cka=False)
        ks = sorted(df["k"].dropna().unique())
        assert ks == [5, 10, 20, 50, 100, 119]
        assert set(df.columns) == {"metric", "k", "k_frac", "neighbors", "value"}
        assert set(df["metric"]) == {"tsi", "qsi", "mutual_knn", "cknna", "procrustes"}

    def test_rows_match_the_single_calls(self, pair):
        X, Y = pair
        df = alignment_profile(X, Y, ks=[10, 119], cca=False, cka=False, method="exact").set_index(
            ["metric", "k", "neighbors"]
        )["value"]
        assert df[("tsi", 119, "all")] == pytest.approx(tsi(X, Y, method="exact"))
        assert df[("qsi", 119, "all")] == pytest.approx(qsi(X, Y, method="exact"))
        assert df[("tsi", 10, "x")] == pytest.approx(tsi(X, Y, k=10, neighbors="x", method="exact"))
        assert df[("tsi", 10, "mean")] == pytest.approx(tsi(X, Y, k=10, method="exact"))
        assert df[("mutual_knn", 10, "x,y")] == pytest.approx(mutual_knn(X, Y, k=10))
        assert df[("procrustes", 119, "all")] == pytest.approx(procrustes_similarity(X, Y))

    def test_global_rows_for_cka_and_cca(self, pair):
        df = alignment_profile(*pair, ks=[10], cca_kwargs={"pre_components": 6})
        metrics = set(df["metric"])
        assert {"linear_cka", "cca_sum_r2", "cca_mean_r2", "cca_r1"} <= metrics
        assert df.loc[df["metric"] == "linear_cka", "k"].isna().all()

    def test_null_column_reads_near_chance_for_tsi(self, pair):
        df = alignment_profile(*pair, ks=[10], cca=False, null=True)
        tsi_null = df.loc[(df["metric"] == "tsi") & (df["neighbors"] == "mean"), "null"].iloc[0]
        assert abs(tsi_null - 0.5) < 0.05
        assert (df["value"] > df["null"]).all()

    def test_single_source_has_no_mean_rows(self, pair):
        df = alignment_profile(*pair, ks=[10], neighbors="x", cca=False, cka=False)
        assert set(df["neighbors"]) == {"x", "x,y"}

    def test_unknown_metric_raises(self, pair):
        with pytest.raises(ValueError, match="Unknown scaled metrics"):
            alignment_profile(*pair, metrics=["tsi", "cka"])

    def test_invalid_k_raises(self, pair):
        with pytest.raises(ValueError, match="k must be in"):
            alignment_profile(*pair, ks=[1])


# ---------------------------------------------------------------------------
# CKNNA (Huh et al. 2024)
# ---------------------------------------------------------------------------


def reference_cknna(A: np.ndarray, B: np.ndarray, topk: int) -> float:
    """Line-by-line numpy port of ``AlignmentMetrics.cknna`` from platonic-rep ``metrics.py``.

    The only change is the ``+ 1e-6`` in the final denominator, dropped here as in embpy.
    """
    n = A.shape[0]
    K, L = A @ A.T, B @ B.T

    def hsic_unbiased(K, L):
        m = K.shape[0]
        Kt, Lt = K.copy(), L.copy()
        np.fill_diagonal(Kt, 0)
        np.fill_diagonal(Lt, 0)
        v = np.sum(Kt * Lt.T) + np.sum(Kt) * np.sum(Lt) / ((m - 1) * (m - 2)) - 2 * np.sum(Kt @ Lt) / (m - 2)
        return v / (m * (m - 3))

    def similarity(K, L):
        K_hat, L_hat = K.copy(), L.copy()
        np.fill_diagonal(K_hat, -np.inf)
        np.fill_diagonal(L_hat, -np.inf)
        iK = np.argsort(-K_hat, axis=1, kind="stable")[:, :topk]
        iL = np.argsort(-L_hat, axis=1, kind="stable")[:, :topk]
        mK, mL = np.zeros((n, n)), np.zeros((n, n))
        mK[np.arange(n)[:, None], iK] = 1
        mL[np.arange(n)[:, None], iL] = 1
        mask = mK * mL
        return hsic_unbiased(mask * K, mask * L)

    return similarity(K, L) / np.sqrt(similarity(K, K) * similarity(L, L))


def brute_force_hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """The U-statistic definition: average of h over all ordered 4-tuples of distinct indices."""
    from itertools import permutations

    total, count = 0.0, 0
    for t, u, v, w in permutations(range(K.shape[0]), 4):
        total += K[t, u] * L[t, u] + K[t, u] * L[v, w] - 2 * K[t, u] * L[t, v]
        count += 1
    return total / count


class TestCKNNA:
    def _unit(self, M):
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    @pytest.mark.parametrize("k", [3, 8, 29])
    def test_matches_the_reference_implementation(self, k, rng):
        A, B = self._unit(rng.normal(size=(30, 5))), self._unit(rng.normal(size=(30, 7)))
        B = 0.6 * self._unit(A @ rng.normal(size=(5, 7))) + 0.4 * B  # partly related
        assert cknna(A, B, k=k, normalize=False) == pytest.approx(reference_cknna(A, B, k), abs=1e-10)

    def test_unbiased_hsic_matches_the_u_statistic(self, rng):
        from embpy.tl.alignment import _hsic_unbiased

        X, Y = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        K, L = X @ X.T, Y @ Y.T
        assert _hsic_unbiased(K, L) == pytest.approx(brute_force_hsic_unbiased(K, L), rel=1e-10)

    def test_rotation_and_row_scale_give_one(self, rng):
        X = rng.normal(size=(80, 6))
        Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        Y = (X @ Q) * rng.uniform(0.5, 2.0, size=(80, 1))  # normalize=True removes row scale
        assert cknna(X, Y, k=10) == pytest.approx(1.0, abs=1e-9)

    def test_unrelated_is_near_zero(self, rng):
        X, Y = rng.normal(size=(300, 8)), rng.normal(size=(300, 8))
        assert abs(cknna(X, Y, k=10)) < 0.2
        assert abs(cknna(X, Y)) < 0.05

    def test_close_to_symmetric(self, rng):
        """Row-wise kNN masks are asymmetric, so the unbiased HSIC term 1'KL1 makes the
        reference definition only approximately symmetric; k = n - 1 is exact."""
        X, Y = rng.normal(size=(60, 5)), rng.normal(size=(60, 4))
        assert cknna(X, Y, k=7) == pytest.approx(cknna(Y, X, k=7), abs=0.05)
        assert cknna(X, Y) == pytest.approx(cknna(Y, X), abs=1e-12)

    def test_shared_signal_scores_above_unrelated(self, rng):
        z = rng.normal(size=(300, 4))
        X = np.hstack([z, rng.normal(size=(300, 4))])
        Y = np.hstack([z, rng.normal(size=(300, 4))])
        assert cknna(X, Y, k=20) > cknna(X, rng.normal(size=(300, 8)), k=20) + 0.1

    def test_profile_rows_match_the_single_call(self, rng):
        X = rng.normal(size=(60, 5))
        Y = X + rng.normal(scale=0.8, size=X.shape)
        df = alignment_profile(X, Y, ks=[8, 59], metrics=["cknna"], cca=False, cka=False)
        df = df.set_index(["k", "neighbors"])["value"]
        assert df[(8, "x,y")] == pytest.approx(cknna(X, Y, k=8))
        assert df[(59, "all")] == pytest.approx(cknna(X, Y))

    def test_invalid_inputs_raise(self, rng):
        X = rng.normal(size=(20, 3))
        with pytest.raises(ValueError, match="k must be in"):
            cknna(X, X, k=1)
        with pytest.raises(ValueError, match="at least 4"):
            cknna(X[:3], X[:3])
        Z = X.copy()
        Z[0] = 0.0
        with pytest.raises(ValueError, match="zero norm"):
            cknna(Z, X)
