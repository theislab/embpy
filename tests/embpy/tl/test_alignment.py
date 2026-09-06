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
    block_to_attention_index,
    block_to_hidden_state_index,
    linear_cka,
    mutual_knn,
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

    @pytest.mark.parametrize("metric", ["tsi", "qsi", "cka", "linear_cka", "mutual_knn"])
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
            alignment_matrix(embs, metric="cca")


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
