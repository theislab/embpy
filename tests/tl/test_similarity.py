"""Tests for embpy.tl.similarity."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from embpy.tl.similarity import (
    compute_distance_matrix,
    compute_knn_overlap,
    compute_similarity,
    rank_perturbations,
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
