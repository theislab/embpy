"""Tests for embpy.tl.similarity."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from embpy.tl.similarity import (
    aggregate_embedding_table,
    compare_embedding_matrices,
    compute_distance_matrix,
    compute_knn_overlap,
    compute_similarity,
    embedding_similarity_matrix,
    knn_jaccard,
    nearest_neighbors,
    nearest_neighbors_table,
    rank_perturbations,
    similarity_correlation,
)


def _make_adata(n: int = 20, d: int = 10) -> AnnData:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float32)
    adata = AnnData(obs={"name": [f"pert_{i}" for i in range(n)]})
    adata.obs_names = [f"pert_{i}" for i in range(n)]
    adata.obsm["X_emb"] = X
    return adata


class TestComputeSimilarity:
    def test_cosine(self):
        adata = _make_adata()
        sim = compute_similarity(adata, "X_emb", metric="cosine")
        assert sim.shape == (20, 20)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_pearson(self):
        adata = _make_adata()
        sim = compute_similarity(adata, "X_emb", metric="pearson")
        assert sim.shape == (20, 20)

    def test_spearman(self):
        adata = _make_adata(n=5)
        sim = compute_similarity(adata, "X_emb", metric="spearman")
        assert sim.shape == (5, 5)

    def test_invalid_metric(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="Unknown"):
            compute_similarity(adata, "X_emb", metric="invalid")

    def test_missing_key(self):
        adata = _make_adata()
        with pytest.raises(KeyError):
            compute_similarity(adata, "nonexistent_key")


class TestComputeDistanceMatrix:
    def test_euclidean(self):
        adata = _make_adata()
        dist = compute_distance_matrix(adata, "X_emb", metric="euclidean")
        assert dist.shape == (20, 20)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)

    def test_cosine_distance(self):
        adata = _make_adata()
        dist = compute_distance_matrix(adata, "X_emb", metric="cosine")
        assert dist.shape == (20, 20)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)

    def test_wasserstein(self):
        adata = _make_adata(n=5)
        dist = compute_distance_matrix(adata, "X_emb", metric="wasserstein")
        assert dist.shape == (5, 5)


class TestKNNOverlap:
    def test_same_embedding(self):
        adata = _make_adata()
        adata.obsm["X_emb2"] = adata.obsm["X_emb"].copy()
        jaccard, mean_j = compute_knn_overlap(adata, "X_emb", "X_emb2", k=5)
        assert jaccard.shape == (20,)
        assert mean_j == 1.0

    def test_different_embeddings(self):
        adata = _make_adata()
        rng = np.random.default_rng(99)
        adata.obsm["X_rand"] = rng.standard_normal((20, 10)).astype(np.float32)
        jaccard, mean_j = compute_knn_overlap(adata, "X_emb", "X_rand", k=5)
        assert 0.0 <= mean_j <= 1.0


class TestRankPerturbations:
    def test_rank_by_index(self):
        adata = _make_adata()
        results = rank_perturbations(adata, "pert_0", "X_emb", top_k=5)
        assert len(results) == 5
        assert results[0][0] == "pert_0"
        assert results[0][1] >= results[1][1]

    def test_rank_by_vector(self):
        adata = _make_adata()
        q = adata.obsm["X_emb"][0]
        results = rank_perturbations(adata, q, "X_emb", top_k=3)
        assert len(results) == 3

    def test_invalid_query(self):
        adata = _make_adata()
        with pytest.raises(KeyError):
            rank_perturbations(adata, "nonexistent", "X_emb")


class TestNearestNeighborsTable:
    def test_query_skips_self(self):
        matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        out = nearest_neighbors_table(matrix, ["a", "b", "c"], query="a", k=2)
        assert out.iloc[0]["neighbor_id"] == "b"
        assert out.iloc[0]["rank"] == 1

    def test_anndata_wrapper(self):
        adata = _make_adata(n=5, d=3)
        out = nearest_neighbors(adata, "X_emb", query="pert_0", k=2)
        assert len(out) == 2
        assert set(out.columns) >= {"query_id", "neighbor_id", "rank", "distance"}


class TestGenericEmbeddingUtilities:
    def test_embedding_similarity_matrix_matches_compute_similarity(self):
        adata = _make_adata(n=5, d=3)
        direct = embedding_similarity_matrix(adata.obsm["X_emb"], metric="cosine")
        from_adata = compute_similarity(adata, "X_emb", metric="cosine")
        np.testing.assert_allclose(direct, from_adata)

    def test_aggregate_embedding_table(self):
        matrix = np.array([[1, 0], [3, 2], [0, 2]], dtype=np.float32)
        out = aggregate_embedding_table(matrix, ["a", "a", "b"])
        assert out.loc["a"].to_numpy().tolist() == pytest.approx([2.0, 1.0])
        assert out.loc["b"].to_numpy().tolist() == pytest.approx([0.0, 2.0])

    def test_similarity_correlation_and_compare(self):
        a = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        b = a.copy()
        corr = similarity_correlation(a, matrix_b=b, label_a="a", target="b")
        comp = compare_embedding_matrices({"a": a, "b": b})
        assert corr.loc[0, "pearson"] == pytest.approx(1.0)
        assert comp.loc[0, "knn_jaccard"] == pytest.approx(1.0)

    def test_knn_jaccard(self):
        a = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        per_row, mean_j = knn_jaccard(a, a, k=1)
        assert per_row.shape == (3,)
        assert mean_j == pytest.approx(1.0)
