"""Tests for embpy.tl.dimred and embpy.tl.clustering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from anndata import AnnData

from embpy.tl.clustering import cluster_embeddings, find_nearest_neighbors, leiden
from embpy.tl.dimred import compute_tsne, compute_umap


def _make_adata(n: int = 30, d: int = 10) -> AnnData:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float32)
    adata = AnnData()
    adata.obs_names = [f"cell_{i}" for i in range(n)]
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
