"""Tests for embpy.tl (analysis tools) and embpy.pl (plotting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_adata() -> AnnData:
    """Small synthetic AnnData with two embedding spaces and metadata."""
    rng = np.random.default_rng(42)
    n_obs = 30
    emb_a = rng.standard_normal((n_obs, 64)).astype(np.float32)
    emb_b = rng.standard_normal((n_obs, 128)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "identifier": [f"gene_{i}" for i in range(n_obs)],
            "perturbation_type": rng.choice(["genetic", "drug"], size=n_obs),
        },
        index=[str(i) for i in range(n_obs)],
    )
    adata = AnnData(obs=obs)
    adata.obsm["X_emb_a"] = emb_a
    adata.obsm["X_emb_b"] = emb_b
    return adata


@pytest.fixture()
def tiny_adata() -> AnnData:
    """Very small AnnData for fast deterministic tests."""
    rng = np.random.default_rng(0)
    n = 10
    obs = pd.DataFrame(
        {"identifier": [f"p{i}" for i in range(n)]},
        index=[str(i) for i in range(n)],
    )
    adata = AnnData(obs=obs)
    adata.obsm["X_test"] = rng.standard_normal((n, 16)).astype(np.float32)
    return adata


# ===================================================================
# tl/ tests (non-scanpy)
# ===================================================================

class TestComputeSimilarity:
    def test_cosine(self, tiny_adata):
        from embpy.tl import compute_similarity

        sim = compute_similarity(tiny_adata, obsm_key="X_test", metric="cosine")
        assert sim.shape == (10, 10)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_pearson(self, tiny_adata):
        from embpy.tl import compute_similarity

        sim = compute_similarity(tiny_adata, obsm_key="X_test", metric="pearson")
        assert sim.shape == (10, 10)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_spearman(self, tiny_adata):
        from embpy.tl import compute_similarity

        sim = compute_similarity(tiny_adata, obsm_key="X_test", metric="spearman")
        assert sim.shape == (10, 10)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_invalid_metric(self, tiny_adata):
        from embpy.tl import compute_similarity

        with pytest.raises(ValueError, match="Unknown similarity metric"):
            compute_similarity(tiny_adata, obsm_key="X_test", metric="invalid")

    def test_missing_key(self, tiny_adata):
        from embpy.tl import compute_similarity

        with pytest.raises(KeyError):
            compute_similarity(tiny_adata, obsm_key="X_missing")


class TestComputeDistanceMatrix:
    def test_euclidean(self, tiny_adata):
        from embpy.tl import compute_distance_matrix

        D = compute_distance_matrix(tiny_adata, obsm_key="X_test", metric="euclidean")
        assert D.shape == (10, 10)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
        assert (D >= 0).all()

    def test_cosine_distance(self, tiny_adata):
        from embpy.tl import compute_distance_matrix

        D = compute_distance_matrix(tiny_adata, obsm_key="X_test", metric="cosine")
        assert D.shape == (10, 10)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_wasserstein(self, tiny_adata):
        from embpy.tl import compute_distance_matrix

        D = compute_distance_matrix(tiny_adata, obsm_key="X_test", metric="wasserstein")
        assert D.shape == (10, 10)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
        assert np.allclose(D, D.T)

    def test_invalid_metric(self, tiny_adata):
        from embpy.tl import compute_distance_matrix

        with pytest.raises(ValueError, match="Unknown distance metric"):
            compute_distance_matrix(tiny_adata, obsm_key="X_test", metric="invalid")


class TestComputeKnnOverlap:
    def test_same_embedding(self, tiny_adata):
        from embpy.tl import compute_knn_overlap

        jaccard, mean_j = compute_knn_overlap(tiny_adata, "X_test", "X_test", k=5)
        assert jaccard.shape == (10,)
        np.testing.assert_allclose(jaccard, 1.0)
        assert mean_j == pytest.approx(1.0)

    def test_different_embeddings(self, synthetic_adata):
        from embpy.tl import compute_knn_overlap

        jaccard, mean_j = compute_knn_overlap(synthetic_adata, "X_emb_a", "X_emb_b", k=5)
        assert jaccard.shape == (30,)
        assert 0.0 <= mean_j <= 1.0


class TestRankPerturbations:
    def test_rank_by_name(self, tiny_adata):
        from embpy.tl import rank_perturbations

        results = rank_perturbations(tiny_adata, query="0", obsm_key="X_test", top_k=5)
        assert len(results) == 5
        assert results[0][0] == "0"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_rank_by_vector(self, tiny_adata):
        from embpy.tl import rank_perturbations

        vec = tiny_adata.obsm["X_test"][0]
        results = rank_perturbations(tiny_adata, query=vec, obsm_key="X_test", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r[1], float) for r in results)

    def test_invalid_query(self, tiny_adata):
        from embpy.tl import rank_perturbations

        with pytest.raises(KeyError, match="not found"):
            rank_perturbations(tiny_adata, query="nonexistent", obsm_key="X_test")


class TestClusterEmbeddings:
    def test_kmeans(self, synthetic_adata):
        from embpy.tl import cluster_embeddings

        result = cluster_embeddings(synthetic_adata, obsm_key="X_emb_a", method="kmeans", n_clusters=3)
        assert "cluster" in result.obs.columns
        assert result.obs["cluster"].nunique() == 3

    def test_invalid_method(self, synthetic_adata):
        from embpy.tl import cluster_embeddings

        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_embeddings(synthetic_adata, obsm_key="X_emb_a", method="unknown")


# ===================================================================
# tl/ tests (scanpy-dependent)
# ===================================================================

class TestScanpyDependentTl:
    @pytest.fixture(autouse=True)
    def _skip_without_scanpy(self):
        pytest.importorskip("scanpy")

    def test_find_nearest_neighbors(self, synthetic_adata):
        from embpy.tl import find_nearest_neighbors

        result = find_nearest_neighbors(synthetic_adata, obsm_key="X_emb_a", n_neighbors=5)
        assert "connectivities" in result.obsp
        assert "distances" in result.obsp

    def test_compute_umap(self, synthetic_adata):
        from embpy.tl import compute_umap

        result = compute_umap(synthetic_adata, obsm_key="X_emb_a")
        assert "X_umap_X_emb_a" in result.obsm
        assert result.obsm["X_umap_X_emb_a"].shape == (30, 2)

    def test_compute_tsne(self, synthetic_adata):
        from embpy.tl import compute_tsne

        result = compute_tsne(synthetic_adata, obsm_key="X_emb_a", perplexity=5)
        assert "X_tsne_X_emb_a" in result.obsm
        assert result.obsm["X_tsne_X_emb_a"].shape == (30, 2)

    def test_leiden(self, synthetic_adata):
        from embpy.tl import leiden

        result = leiden(synthetic_adata, obsm_key="X_emb_a", resolution=0.5, key_added="test_leiden")
        assert "test_leiden" in result.obs.columns
        assert result.obs["test_leiden"].nunique() >= 1


# ===================================================================
# pl/ helper tests
# ===================================================================

class TestGetEmbeddingKeys:
    def test_auto_discovery(self, synthetic_adata):
        from embpy.pl.basic import _get_embedding_keys

        keys = _get_embedding_keys(synthetic_adata)
        assert "X_emb_a" in keys
        assert "X_emb_b" in keys

    def test_explicit_keys(self, synthetic_adata):
        from embpy.pl.basic import _get_embedding_keys

        keys = _get_embedding_keys(synthetic_adata, obsm_keys=["X_emb_a"])
        assert keys == ["X_emb_a"]

    def test_skip_umap_tsne(self, synthetic_adata):
        from embpy.pl.basic import _get_embedding_keys

        synthetic_adata.obsm["X_umap"] = np.zeros((30, 2))
        synthetic_adata.obsm["X_tsne"] = np.zeros((30, 2))
        keys = _get_embedding_keys(synthetic_adata)
        assert "X_umap" not in keys
        assert "X_tsne" not in keys

    def test_invalid_key_raises(self, synthetic_adata):
        from embpy.pl.basic import _get_embedding_keys

        with pytest.raises(KeyError):
            _get_embedding_keys(synthetic_adata, obsm_keys=["X_nonexistent"])


# ===================================================================
# pl/ plotting tests (non-scanpy)
# ===================================================================

class TestPlotSimilarityHeatmap:
    def test_from_matrix(self, tiny_adata):
        from embpy.pl import plot_similarity_heatmap
        from embpy.tl import compute_similarity

        sim = compute_similarity(tiny_adata, obsm_key="X_test", metric="cosine")
        fig = plot_similarity_heatmap(similarity_matrix=sim)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_from_adata(self, tiny_adata):
        from embpy.pl import plot_similarity_heatmap

        fig = plot_similarity_heatmap(adata=tiny_adata, obsm_key="X_test", metric="cosine")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestDistanceHeatmap:
    def test_euclidean(self, tiny_adata):
        from embpy.pl import distance_heatmap

        fig = distance_heatmap(tiny_adata, obsm_key="X_test", metric="euclidean")
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_wasserstein(self, tiny_adata):
        from embpy.pl import distance_heatmap

        fig = distance_heatmap(tiny_adata, obsm_key="X_test", metric="wasserstein")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestCorrelationMatrix:
    def test_pearson(self, tiny_adata):
        from embpy.pl import correlation_matrix

        fig = correlation_matrix(tiny_adata, obsm_key="X_test", method="pearson")
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_spearman(self, tiny_adata):
        from embpy.pl import correlation_matrix

        fig = correlation_matrix(tiny_adata, obsm_key="X_test", method="spearman")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestCrossEmbeddingCorrelation:
    def test_basic(self, synthetic_adata):
        from embpy.pl import cross_embedding_correlation

        fig = cross_embedding_correlation(
            synthetic_adata, obsm_key_a="X_emb_a", obsm_key_b="X_emb_b", method="pearson"
        )
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPerturbationRanking:
    def test_from_rankings(self):
        from embpy.pl import plot_perturbation_ranking

        rankings = [(f"gene_{i}", 1.0 - i * 0.05) for i in range(10)]
        fig = plot_perturbation_ranking(rankings=rankings, top_k=5)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_from_adata(self, tiny_adata):
        from embpy.pl import plot_perturbation_ranking

        fig = plot_perturbation_ranking(adata=tiny_adata, query="0", obsm_key="X_test", top_k=5)
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestDendrogram:
    def test_cosine(self, tiny_adata):
        from embpy.pl import dendrogram

        fig = dendrogram(tiny_adata, obsm_key="X_test", metric="cosine")
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_euclidean(self, tiny_adata):
        from embpy.pl import dendrogram

        fig = dendrogram(tiny_adata, obsm_key="X_test", metric="euclidean", linkage_method="ward")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestEmbeddingDistributions:
    def test_basic(self, tiny_adata):
        from embpy.pl import embedding_distributions

        fig = embedding_distributions(tiny_adata, n_dims=5)
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestEmbeddingNorms:
    def test_basic(self, synthetic_adata):
        from embpy.pl import embedding_norms

        fig = embedding_norms(synthetic_adata)
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestClusterComposition:
    def test_basic(self, synthetic_adata):
        from embpy.pl import plot_cluster_composition
        from embpy.tl import cluster_embeddings

        cluster_embeddings(synthetic_adata, obsm_key="X_emb_a", method="kmeans", n_clusters=3)
        fig = plot_cluster_composition(synthetic_adata, cluster_key="cluster", color_by="perturbation_type")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestKnnOverlap:
    def test_basic(self, synthetic_adata):
        from embpy.pl import knn_overlap

        fig = knn_overlap(synthetic_adata, k=5)
        assert isinstance(fig, Figure)
        plt_close(fig)


# ===================================================================
# pl/ plotting tests (scanpy-dependent)
# ===================================================================

class TestScanpyDependentPl:
    @pytest.fixture(autouse=True)
    def _skip_without_scanpy(self):
        pytest.importorskip("scanpy")

    def test_plot_embedding_space_umap(self, synthetic_adata):
        from embpy.pl import plot_embedding_space

        fig = plot_embedding_space(
            synthetic_adata, obsm_key="X_emb_a", color="perturbation_type", method="umap"
        )
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_plot_embedding_space_tsne(self, synthetic_adata):
        from embpy.pl import plot_embedding_space

        fig = plot_embedding_space(
            synthetic_adata, obsm_key="X_emb_a", method="tsne"
        )
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_all_embeddings(self, synthetic_adata):
        from embpy.pl import all_embeddings

        fig = all_embeddings(synthetic_adata, method="umap", color="perturbation_type")
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_leiden_overview(self, synthetic_adata):
        from embpy.pl import leiden_overview

        fig = leiden_overview(
            synthetic_adata, obsm_key="X_emb_a", resolution=0.5, color_by="perturbation_type"
        )
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_leiden_overview_selected_plots(self, synthetic_adata):
        from embpy.pl import leiden_overview

        fig = leiden_overview(
            synthetic_adata, obsm_key="X_emb_a", resolution=0.5,
            plots=["umap_cluster", "cluster_sizes"]
        )
        assert isinstance(fig, Figure)
        plt_close(fig)


# ===================================================================
# Utility
# ===================================================================

def plt_close(fig: Figure) -> None:
    """Close a figure to avoid resource warnings in tests."""
    import matplotlib.pyplot as _plt
    _plt.close(fig)
