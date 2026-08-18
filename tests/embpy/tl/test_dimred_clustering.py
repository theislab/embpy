"""Tests for embpy.tl.dimred and embpy.tl.clustering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.tl.clustering import (
    cluster_annotation_enrichment,
    cluster_embeddings,
    find_nearest_neighbors,
    leiden,
)
from embpy.tl.dimred import compute_pca, compute_tsne, compute_umap


def _make_adata(n: int = 30, d: int = 10) -> AnnData:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float32)
    names = [f"cell_{i}" for i in range(n)]
    adata = AnnData(
        X=np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(names)),
    )
    adata.obsm["X_emb"] = X
    return adata


def _labelled_adata(n_per_group: int = 6, d: int = 16, seed: int = 0) -> AnnData:
    """Adata with well-separated clusters tagged by 'group' in obs."""
    rng = np.random.default_rng(seed)
    groups = ["A", "B", "C", "D"]
    centers = rng.normal(0, 4, size=(len(groups), d))
    X = np.vstack(
        [centers[i] + rng.normal(0, 1, (n_per_group, d)) for i in range(len(groups))]
    ).astype(np.float32)
    names = [f"{g}_{i}" for g in groups for i in range(n_per_group)]
    labels = [g for g in groups for _ in range(n_per_group)]
    adata = AnnData(
        X=np.zeros((len(names), 1), dtype=np.float32),
        obs=pd.DataFrame({"group": labels}, index=pd.Index(names)),
    )
    adata.obsm["X_emb"] = X
    return adata


# =====================================================================
# Dimensionality reduction
# =====================================================================


class TestComputeUMAP:
    @patch("embpy.tl.dimred._require_scanpy")
    def test_umap_cpu(self, mock_sc):
        sc = MagicMock()
        mock_sc.return_value = sc

        adata = _make_adata()
        sc.tl.umap = MagicMock()
        sc.pp.neighbors = MagicMock()
        adata.obsm["X_umap"] = np.random.randn(30, 2).astype(np.float32)

        result = compute_umap(adata, "X_emb", backend="cpu")
        sc.pp.neighbors.assert_called_once()
        sc.tl.umap.assert_called_once()
        assert "X_umap_X_emb" in result.obsm

    def test_umap_custom_output_key(self):
        with patch("embpy.tl.dimred._require_scanpy") as mock_sc:
            sc = MagicMock()
            mock_sc.return_value = sc
            adata = _make_adata()
            adata.obsm["X_umap"] = np.random.randn(30, 2).astype(np.float32)

            result = compute_umap(adata, "X_emb", output_key="my_umap")
            assert "my_umap" in result.obsm

    @patch("embpy.tl.dimred._require_scanpy")
    def test_spectral_init_on_a_normal_sized_input(self, mock_sc):
        sc = MagicMock()
        mock_sc.return_value = sc
        adata = _make_adata(n=30)
        adata.obsm["X_umap"] = np.random.randn(30, 2).astype(np.float32)

        compute_umap(adata, "X_emb")
        assert sc.tl.umap.call_args[1]["init_pos"] == "spectral"

    @patch("embpy.tl.dimred._require_scanpy")
    def test_tiny_input_falls_back_to_random_init(self, mock_sc):
        """Regression: spectral init needs more rows than components.

        UMAP solves for ``n_components + 1`` eigenvectors of the neighbour
        graph, so at three observations scipy raised "Cannot use
        scipy.linalg.eigh for sparse A with k >= N" from inside the solver --
        which is what ``pl.plot_species_umap`` hit whenever an ortholog lookup
        returned only a couple of species.
        """
        sc = MagicMock()
        mock_sc.return_value = sc
        adata = _make_adata(n=3)
        adata.obsm["X_umap"] = np.random.randn(3, 2).astype(np.float32)

        compute_umap(adata, "X_emb")
        assert sc.tl.umap.call_args[1]["init_pos"] == "random"
        # n_neighbors must also fit inside the graph.
        assert sc.pp.neighbors.call_args[1]["n_neighbors"] <= 2

    @patch("embpy.tl.dimred._require_scanpy")
    def test_n_neighbors_never_exceeds_the_row_count(self, mock_sc):
        sc = MagicMock()
        mock_sc.return_value = sc
        adata = _make_adata(n=6)
        adata.obsm["X_umap"] = np.random.randn(6, 2).astype(np.float32)

        compute_umap(adata, "X_emb", n_neighbors=15)   # more than there are rows
        assert sc.pp.neighbors.call_args[1]["n_neighbors"] == 5


class TestComputeTSNE:
    @patch("embpy.tl.dimred._require_scanpy")
    def test_tsne(self, mock_sc):
        sc = MagicMock()
        mock_sc.return_value = sc

        adata = _make_adata()
        adata.obsm["X_tsne"] = np.random.randn(30, 2).astype(np.float32)

        result = compute_tsne(adata, "X_emb")
        sc.tl.tsne.assert_called_once()
        assert "X_tsne_X_emb" in result.obsm


class TestComputePCA:
    def test_default_output_key(self):
        adata = _make_adata()
        result = compute_pca(adata, "X_emb")
        assert "X_pca_X_emb" in result.obsm
        assert result.obsm["X_pca_X_emb"].shape == (30, 2)
        assert "X_pca_X_emb_variance_ratio" in result.uns

    def test_custom_output_key(self):
        adata = _make_adata()
        result = compute_pca(adata, "X_emb", output_key="my_pca", n_components=3)
        assert "my_pca" in result.obsm
        assert result.obsm["my_pca"].shape == (30, 3)
        assert "my_pca_variance_ratio" in result.uns

    def test_variance_ratio_sums_to_at_most_one(self):
        adata = _make_adata()
        result = compute_pca(adata, "X_emb", n_components=5)
        ratios = result.uns["X_pca_X_emb_variance_ratio"]
        assert ratios.shape == (5,)
        assert ratios.sum() <= 1.0 + 1e-6
        assert (ratios >= 0).all()

    def test_n_components_clamped(self):
        adata = _make_adata(n=4, d=10)
        result = compute_pca(adata, "X_emb", n_components=20)
        assert result.obsm["X_pca_X_emb"].shape[1] <= 4

    def test_missing_key_raises(self):
        adata = _make_adata()
        with pytest.raises(KeyError):
            compute_pca(adata, "X_missing")

    def test_separates_clusters(self):
        adata = _labelled_adata()
        compute_pca(adata, "X_emb")
        coords = adata.obsm["X_pca_X_emb"]
        groups = adata.obs["group"].to_numpy()
        intra = []
        for g in np.unique(groups):
            sub = coords[groups == g]
            intra.append(np.linalg.norm(sub - sub.mean(axis=0), axis=1).mean())
        centroids = np.stack(
            [coords[groups == g].mean(axis=0) for g in np.unique(groups)]
        )
        inter = np.linalg.norm(
            centroids[:, None, :] - centroids[None, :, :], axis=-1
        )
        inter_mean = inter[np.triu_indices(len(centroids), k=1)].mean()
        assert inter_mean > np.mean(intra)


# =====================================================================
# Clustering
# =====================================================================


class TestFindNearestNeighbors:
    @patch("embpy.tl.clustering._require_scanpy")
    def test_neighbors_cpu(self, mock_sc):
        sc = MagicMock()
        mock_sc.return_value = sc

        adata = _make_adata()
        result = find_nearest_neighbors(adata, "X_emb", n_neighbors=5, backend="cpu")
        sc.pp.neighbors.assert_called_once()

    @patch("embpy.tl.clustering._require_rapids")
    def test_neighbors_gpu(self, mock_rsc):
        rsc = MagicMock()
        mock_rsc.return_value = rsc

        adata = _make_adata()
        result = find_nearest_neighbors(adata, "X_emb", backend="gpu")
        rsc.pp.neighbors.assert_called_once()


class TestLeiden:
    @patch("embpy.tl.clustering._require_scanpy")
    def test_leiden_cpu(self, mock_sc):
        sc = MagicMock()
        mock_sc.return_value = sc

        adata = _make_adata()
        adata.obs["leiden"] = ["0"] * 15 + ["1"] * 15
        import pandas as pd
        adata.obs["leiden"] = pd.Categorical(adata.obs["leiden"])

        result = leiden(adata, "X_emb", resolution=1.0, backend="cpu")
        sc.pp.neighbors.assert_called_once()
        sc.tl.leiden.assert_called_once()


class TestClusterEmbeddings:
    def test_kmeans(self):
        adata = _make_adata()
        result = cluster_embeddings(
            adata, "X_emb", method="kmeans", n_clusters=3,
        )
        assert "cluster" in result.obs.columns
        assert result.obs["cluster"].nunique() == 3

    def test_spectral(self):
        adata = _make_adata()
        result = cluster_embeddings(
            adata, "X_emb", method="spectral", n_clusters=3,
        )
        assert "cluster" in result.obs.columns

    def test_invalid_method(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="Unknown"):
            cluster_embeddings(adata, "X_emb", method="invalid")


class TestClusterAnnotationEnrichment:
    def test_single_label_enrichment(self):
        adata = _labelled_adata()
        adata.obs["cluster"] = pd.Categorical(adata.obs["group"])
        df = cluster_annotation_enrichment(
            adata, cluster_key="cluster", annotation_key="group", top_k=3,
        )
        assert not df.empty
        expected_cols = {"cluster", "term", "in_cluster", "overall",
                         "in_freq", "overall_freq", "enrichment"}
        assert expected_cols.issubset(df.columns)
        for c, sub in df.groupby("cluster"):
            top = sub.iloc[0]
            assert top["term"] == c
            assert top["enrichment"] >= 1.0

    def test_multi_label_enrichment(self):
        adata = _labelled_adata()
        adata.obs["cluster"] = pd.Categorical(adata.obs["group"])
        terms = []
        for g in adata.obs["group"]:
            base = [f"GO:{g}_term"]
            if g in {"A", "B"}:
                base.append("GO:cancer")
            terms.append(base)
        adata.obs["go_terms"] = terms
        df = cluster_annotation_enrichment(
            adata, cluster_key="cluster", annotation_key="go_terms", top_k=4,
        )
        assert not df.empty
        for c, sub in df.groupby("cluster"):
            top_term = sub.iloc[0]["term"]
            assert top_term == f"GO:{c}_term"

    def test_top_k_respected(self):
        adata = _labelled_adata()
        adata.obs["cluster"] = pd.Categorical(adata.obs["group"])
        df = cluster_annotation_enrichment(
            adata, cluster_key="cluster", annotation_key="group", top_k=1,
        )
        for _, sub in df.groupby("cluster"):
            assert len(sub) == 1

    def test_handles_nan_annotation(self):
        adata = _labelled_adata()
        adata.obs["cluster"] = pd.Categorical(adata.obs["group"])
        ann = adata.obs["group"].astype(object).copy()
        ann.iloc[0] = np.nan
        adata.obs["maybe_label"] = ann
        df = cluster_annotation_enrichment(
            adata, cluster_key="cluster", annotation_key="maybe_label",
        )
        assert not df.empty

    def test_missing_keys_raise(self):
        adata = _labelled_adata()
        adata.obs["cluster"] = pd.Categorical(adata.obs["group"])
        with pytest.raises(KeyError):
            cluster_annotation_enrichment(
                adata, cluster_key="missing", annotation_key="group",
            )
        with pytest.raises(KeyError):
            cluster_annotation_enrichment(
                adata, cluster_key="cluster", annotation_key="missing",
            )
