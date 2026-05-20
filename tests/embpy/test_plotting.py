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
        from embpy.pl._helpers import _get_embedding_keys

        keys = _get_embedding_keys(synthetic_adata)
        assert "X_emb_a" in keys
        assert "X_emb_b" in keys

    def test_explicit_keys(self, synthetic_adata):
        from embpy.pl._helpers import _get_embedding_keys

        keys = _get_embedding_keys(synthetic_adata, obsm_keys=["X_emb_a"])
        assert keys == ["X_emb_a"]

    def test_skip_umap_tsne(self, synthetic_adata):
        from embpy.pl._helpers import _get_embedding_keys

        synthetic_adata.obsm["X_umap"] = np.zeros((30, 2))
        synthetic_adata.obsm["X_tsne"] = np.zeros((30, 2))
        keys = _get_embedding_keys(synthetic_adata)
        assert "X_umap" not in keys
        assert "X_tsne" not in keys

    def test_invalid_key_raises(self, synthetic_adata):
        from embpy.pl._helpers import _get_embedding_keys

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

    def test_into_external_axes(self, tiny_adata):
        """plot_similarity_heatmap accepts ax= and draws into it without creating a fig."""
        import matplotlib.pyplot as plt

        from embpy.pl import plot_similarity_heatmap

        fig, ax = plt.subplots()
        out = plot_similarity_heatmap(
            adata=tiny_adata, obsm_key="X_test", metric="cosine", ax=ax,
        )
        assert out is fig
        assert len(ax.collections) > 0
        plt_close(fig)

    def test_label_col_used_for_ticks(self, synthetic_adata):
        """label_col= picks tick labels from a chosen obs column."""
        from embpy.pl import plot_similarity_heatmap

        fig = plot_similarity_heatmap(
            adata=synthetic_adata, obsm_key="X_emb_a", metric="cosine",
            label_col="identifier",
        )
        ax = fig.axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("gene_" in lbl for lbl in ytick_labels)
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
# pl/ embedding-space additions (PCA + annotate, panel, highlight)
# ===================================================================


class TestPlotEmbeddingSpacePCA:
    """compute_pca-backed branch of plot_embedding_space (no scanpy needed)."""

    def test_pca_method_creates_obsm_and_axes(self, synthetic_adata):
        from embpy.pl import plot_embedding_space

        fig = plot_embedding_space(
            synthetic_adata, obsm_key="X_emb_a", method="pca",
            color="perturbation_type",
        )
        assert isinstance(fig, Figure)
        assert "X_pca_X_emb_a" in synthetic_adata.obsm
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel()
        assert "%" in ax.get_xlabel()
        plt_close(fig)

    def test_annotate_draws_text_per_observation(self, tiny_adata):
        from embpy.pl import plot_embedding_space

        fig = plot_embedding_space(
            tiny_adata, obsm_key="X_test", method="pca", annotate=True,
        )
        ax = fig.axes[0]
        text_count = sum(1 for t in ax.texts if t.get_text())
        assert text_count >= tiny_adata.n_obs
        plt_close(fig)

    def test_annotate_col_uses_obs_column(self, synthetic_adata):
        from embpy.pl import plot_embedding_space

        fig = plot_embedding_space(
            synthetic_adata, obsm_key="X_emb_a", method="pca",
            annotate=True, annotate_col="identifier",
        )
        ax = fig.axes[0]
        labels = {t.get_text() for t in ax.texts}
        assert any("gene_" in lbl for lbl in labels)
        plt_close(fig)

    def test_into_external_ax(self, tiny_adata):
        import matplotlib.pyplot as plt

        from embpy.pl import plot_embedding_space

        fig, ax = plt.subplots()
        out = plot_embedding_space(
            tiny_adata, obsm_key="X_test", method="pca", ax=ax,
        )
        assert out is fig
        plt_close(fig)

    def test_invalid_method_raises(self, tiny_adata):
        from embpy.pl import plot_embedding_space

        with pytest.raises(ValueError, match="Unknown method"):
            plot_embedding_space(
                tiny_adata, obsm_key="X_test",
                method="not_a_method",  # type: ignore[arg-type]
            )


class TestEmbeddingColorPanel:
    def test_panel_count_matches_color_keys(self, synthetic_adata):
        from embpy.pl import embedding_color_panel

        synthetic_adata.obs["category2"] = ["cat_a"] * 15 + ["cat_b"] * 15
        fig = embedding_color_panel(
            synthetic_adata,
            color_keys=["perturbation_type", "category2"],
            obsm_key="X_emb_a",
            method="pca",
            ncols=2,
        )
        assert isinstance(fig, Figure)
        visible_axes = [a for a in fig.axes if a.get_visible()]
        assert len([a for a in visible_axes if a.collections]) >= 2
        plt_close(fig)

    def test_extra_grid_axes_hidden(self, synthetic_adata):
        from embpy.pl import embedding_color_panel

        fig = embedding_color_panel(
            synthetic_adata,
            color_keys=["perturbation_type"],
            obsm_key="X_emb_a",
            method="pca",
            ncols=2,
        )
        hidden = [a for a in fig.axes if not a.get_visible()]
        assert len(hidden) >= 1
        plt_close(fig)

    def test_no_obsm_keys_raises(self):
        from embpy.pl import embedding_color_panel

        adata = AnnData(
            obs=pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index(list("abc"))),
        )
        with pytest.raises(ValueError):
            embedding_color_panel(adata, color_keys=["x"])


class TestHighlightGeneSets:
    def test_highlights_set_members(self, synthetic_adata):
        from embpy.pl import highlight_gene_sets

        gene_ids = synthetic_adata.obs["identifier"].tolist()
        sets = {
            "first_three": gene_ids[:3],
            "next_three": gene_ids[3:6],
        }
        fig = highlight_gene_sets(
            synthetic_adata, gene_sets=sets, obsm_key="X_emb_a",
            method="pca", label_col="identifier", ncols=2, annotate=True,
        )
        assert isinstance(fig, Figure)
        visible_axes = [a for a in fig.axes if a.get_visible()]
        assert len(visible_axes) == 2
        for ax in visible_axes:
            assert len(ax.collections) >= 2
            assert "n=3" in ax.get_title()
        plt_close(fig)

    def test_handles_unknown_members_gracefully(self, synthetic_adata):
        from embpy.pl import highlight_gene_sets

        sets = {"only_unknown": ["nonexistent_gene_1", "nonexistent_gene_2"]}
        fig = highlight_gene_sets(
            synthetic_adata, gene_sets=sets, obsm_key="X_emb_a",
            method="pca", label_col="identifier",
        )
        ax = next(a for a in fig.axes if a.get_visible())
        assert "n=0" in ax.get_title()
        plt_close(fig)

    def test_invalid_method_raises(self, tiny_adata):
        from embpy.pl import highlight_gene_sets

        with pytest.raises(ValueError, match="Unknown"):
            highlight_gene_sets(
                tiny_adata, gene_sets={"a": ["0"]}, obsm_key="X_test",
                method="bogus",  # type: ignore[arg-type]
            )


# ===================================================================
# Utility
# ===================================================================

def plt_close(fig: Figure) -> None:
    """Close a figure to avoid resource warnings in tests."""
    import matplotlib.pyplot as _plt
    _plt.close(fig)
