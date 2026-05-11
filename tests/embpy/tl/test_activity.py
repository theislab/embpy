"""Tests for embpy.tl.activity -- phenotypic activity scoring."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from embpy.tl.activity import (
    _ap_from_ranked_relevance,
    _chunked_cosine_ap_cpu,
    phenotypic_activity,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_activity_adata(
    n_perts: int = 10,
    n_reps: int = 5,
    d: int = 32,
    noise: float = 0.1,
    seed: int = 42,
) -> AnnData:
    """Create synthetic AnnData where each perturbation has a distinct centroid.

    With low noise, replicates cluster tightly and mAP should be high.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_perts, d)).astype(np.float32)

    X_parts, labels = [], []
    for i in range(n_perts):
        reps = centroids[i] + noise * rng.standard_normal((n_reps, d)).astype(np.float32)
        X_parts.append(reps)
        labels.extend([f"pert_{i}"] * n_reps)

    X = np.vstack(X_parts)
    adata = AnnData(
        obs={"perturbation": labels},
    )
    adata.obsm["X_emb"] = X
    return adata


def _make_random_adata(n: int = 50, d: int = 16, n_perts: int = 10, seed: int = 0) -> AnnData:
    """AnnData with purely random features -- mAP should be near chance."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    labels = [f"pert_{i % n_perts}" for i in range(n)]
    adata = AnnData(obs={"perturbation": labels})
    adata.obsm["X_emb"] = X
    return adata


# ------------------------------------------------------------------
# Tests: _ap_from_ranked_relevance
# ------------------------------------------------------------------

class TestAPFromRankedRelevance:
    def test_perfect_ranking(self):
        rel = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
        assert _ap_from_ranked_relevance(rel) == pytest.approx(1.0)

    def test_worst_ranking(self):
        rel = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        ap = _ap_from_ranked_relevance(rel)
        assert 0.0 < ap < 1.0

    def test_no_positives(self):
        rel = np.array([0, 0, 0, 0], dtype=np.float64)
        assert _ap_from_ranked_relevance(rel) == 0.0

    def test_single_positive_at_top(self):
        rel = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        assert _ap_from_ranked_relevance(rel) == pytest.approx(1.0)

    def test_single_positive_at_bottom(self):
        rel = np.array([0, 0, 0, 0, 1], dtype=np.float64)
        assert _ap_from_ranked_relevance(rel) == pytest.approx(1 / 5)

    def test_known_value(self):
        # positives at ranks 1, 3, 5 -> precisions 1/1, 2/3, 3/5
        rel = np.array([1, 0, 1, 0, 1, 0], dtype=np.float64)
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert _ap_from_ranked_relevance(rel) == pytest.approx(expected)


# ------------------------------------------------------------------
# Tests: _chunked_cosine_ap_cpu
# ------------------------------------------------------------------

class TestChunkedCosineAPCPU:
    def test_perfect_clusters(self):
        rng = np.random.default_rng(42)
        d = 32
        centroids = np.eye(3, d, dtype=np.float32) * 10
        X_parts = []
        labels = []
        for i in range(3):
            reps = centroids[i] + 0.01 * rng.standard_normal((5, d)).astype(np.float32)
            X_parts.append(reps)
            labels.extend([i] * 5)
        X = np.vstack(X_parts)
        from sklearn.preprocessing import normalize
        X_norm = normalize(X, norm="l2", axis=1).astype(np.float32)
        labels_arr = np.array(labels)

        ap = _chunked_cosine_ap_cpu(X_norm, labels_arr, chunk_size=5)
        assert ap.shape == (15,)
        assert ap.mean() > 0.95

    def test_chunk_size_does_not_change_result(self):
        rng = np.random.default_rng(7)
        n, d = 30, 8
        X = rng.standard_normal((n, d)).astype(np.float32)
        from sklearn.preprocessing import normalize
        X_norm = normalize(X, norm="l2", axis=1).astype(np.float32)
        labels = np.array([i // 3 for i in range(n)])

        ap_small = _chunked_cosine_ap_cpu(X_norm, labels, chunk_size=4)
        ap_big = _chunked_cosine_ap_cpu(X_norm, labels, chunk_size=100)
        np.testing.assert_allclose(ap_small, ap_big, atol=1e-10)

    def test_single_chunk(self):
        rng = np.random.default_rng(1)
        n, d = 10, 4
        X = rng.standard_normal((n, d)).astype(np.float32)
        from sklearn.preprocessing import normalize
        X_norm = normalize(X, norm="l2", axis=1).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        ap = _chunked_cosine_ap_cpu(X_norm, labels, chunk_size=100)
        assert ap.shape == (10,)
        assert np.all(ap >= 0.0)
        assert np.all(ap <= 1.0)


# ------------------------------------------------------------------
# Tests: phenotypic_activity (public API)
# ------------------------------------------------------------------

class TestPhenotypicActivity:
    def test_basic_output_shape(self):
        adata = _make_activity_adata(n_perts=5, n_reps=4)
        result = phenotypic_activity(adata, "X_emb", "perturbation")
        assert isinstance(result, type(result))
        assert set(result.columns) == {
            "perturbation", "mean_ap", "normalized_mean_ap", "n_wells",
        }
        assert len(result) == 5
        assert (result["n_wells"] == 4).all()

    def test_high_signal_has_high_map(self):
        adata = _make_activity_adata(n_perts=8, n_reps=6, noise=0.01)
        result = phenotypic_activity(adata, "X_emb", "perturbation")
        assert result["mean_ap"].mean() > 0.9

    def test_random_data_near_chance(self):
        adata = _make_random_adata(n=100, d=16, n_perts=20, seed=123)
        result = phenotypic_activity(adata, "X_emb", "perturbation")
        assert result["mean_ap"].mean() < 0.5

    def test_normalized_ap_range(self):
        adata = _make_activity_adata(n_perts=6, n_reps=5)
        result = phenotypic_activity(adata, "X_emb", "perturbation")
        assert (result["normalized_mean_ap"] <= 1.0 + 1e-6).all()

    def test_control_filtering(self):
        adata = _make_activity_adata(n_perts=5, n_reps=4)
        adata.obs["is_ctrl"] = [True] * 4 + [False] * 16
        result = phenotypic_activity(
            adata, "X_emb", "perturbation",
            control_col="is_ctrl", control_ids={True},
        )
        total_wells = result["n_wells"].sum()
        assert total_wells == 16

    def test_sorted_descending(self):
        adata = _make_activity_adata(n_perts=10, n_reps=5)
        result = phenotypic_activity(adata, "X_emb", "perturbation")
        ap_values = result["mean_ap"].values
        assert np.all(ap_values[:-1] >= ap_values[1:])

    def test_missing_obsm_key(self):
        adata = _make_activity_adata()
        with pytest.raises(KeyError, match="not found"):
            phenotypic_activity(adata, "nonexistent", "perturbation")

    def test_missing_perturbation_col(self):
        adata = _make_activity_adata()
        with pytest.raises(KeyError, match="not in adata.obs"):
            phenotypic_activity(adata, "X_emb", "nonexistent_col")

    def test_unsupported_metric(self):
        adata = _make_activity_adata()
        with pytest.raises(ValueError, match="Unsupported metric"):
            phenotypic_activity(adata, "X_emb", "perturbation", metric="euclidean")

    def test_small_chunk_size(self):
        adata = _make_activity_adata(n_perts=4, n_reps=3)
        r1 = phenotypic_activity(adata, "X_emb", "perturbation", chunk_size=1)
        r2 = phenotypic_activity(adata, "X_emb", "perturbation", chunk_size=100)
        np.testing.assert_allclose(
            r1.set_index("perturbation")["mean_ap"].sort_index().values,
            r2.set_index("perturbation")["mean_ap"].sort_index().values,
            atol=1e-10,
        )

    def test_single_replicate_perturbation(self):
        adata = _make_activity_adata(n_perts=3, n_reps=5)
        extra = AnnData(obs={"perturbation": ["singleton"]})
        rng = np.random.default_rng(99)
        extra.obsm["X_emb"] = rng.standard_normal((1, 32)).astype(np.float32)
        import anndata as ad
        adata = ad.concat([adata, extra], join="outer")
        result = phenotypic_activity(adata, "X_emb", "perturbation")
        singleton_row = result[result["perturbation"] == "singleton"]
        assert len(singleton_row) == 1
        assert singleton_row["mean_ap"].values[0] == 0.0
