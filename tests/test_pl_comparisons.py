"""Tests for embpy.pl.comparisons (parallel coordinates, radar chart,
star coordinates, t-SNE feature panel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

sc = pytest.importorskip("scanpy")


@pytest.fixture
def tiny_adata():
    n_obs = 30
    rng = np.random.default_rng(42)
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical([str(i % 3) for i in range(n_obs)]),
            "molecular_weight": rng.normal(300, 50, n_obs),
            "logp": rng.normal(2, 1, n_obs),
            "tpsa": rng.normal(80, 20, n_obs),
            "hbd": rng.integers(0, 5, n_obs).astype(float),
            "species": pd.Categorical(["human"] * 15 + ["mouse"] * 15),
        },
        index=pd.Index([f"mol_{i}" for i in range(n_obs)]),
    )
    adata = AnnData(obs=obs)
    adata.obsm["X_model_a"] = rng.normal(size=(n_obs, 50)).astype(np.float32)
    adata.obsm["X_model_b"] = rng.normal(size=(n_obs, 100)).astype(np.float32)
    adata.obsm["X_model_a_pca"] = rng.normal(size=(n_obs, 20)).astype(np.float32)
    return adata


class TestParallelCoordinates:
    def test_returns_figure(self, tiny_adata):
        from embpy.pl.comparisons import parallel_coordinates

        fig = parallel_coordinates(
            tiny_adata, obsm_key="X_model_a_pca",
            obs_features=["molecular_weight", "logp"],
            n_dims=3, color_by="cluster",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_dims_only(self, tiny_adata):
        from embpy.pl.comparisons import parallel_coordinates

        fig = parallel_coordinates(
            tiny_adata, obsm_key="X_model_a_pca",
            obs_features=[], n_dims=5, color_by="cluster",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_obs_features_only(self, tiny_adata):
        from embpy.pl.comparisons import parallel_coordinates

        fig = parallel_coordinates(
            tiny_adata, obsm_key="X_model_a_pca",
            obs_features=["molecular_weight", "logp", "tpsa"],
            n_dims=0, color_by="cluster",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_raises_missing_obsm(self, tiny_adata):
        from embpy.pl.comparisons import parallel_coordinates

        with pytest.raises(KeyError):
            parallel_coordinates(
                tiny_adata, obsm_key="X_nonexistent",
                obs_features=["logp"], color_by="cluster",
            )

    def test_raises_missing_color_by(self, tiny_adata):
        from embpy.pl.comparisons import parallel_coordinates

        with pytest.raises(KeyError):
            parallel_coordinates(
                tiny_adata, obsm_key="X_model_a_pca",
                obs_features=["logp"], color_by="nonexistent",
            )


class TestRadarChart:
    def test_returns_figure(self, tiny_adata):
        from embpy.pl.comparisons import radar_chart

        fig = radar_chart(
            tiny_adata,
            properties=["molecular_weight", "logp", "tpsa", "hbd"],
            group_by="cluster",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_z_score_false(self, tiny_adata):
        from embpy.pl.comparisons import radar_chart

        fig = radar_chart(
            tiny_adata,
            properties=["molecular_weight", "logp", "tpsa"],
            group_by="cluster", z_score=False,
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_groups_subset(self, tiny_adata):
        from embpy.pl.comparisons import radar_chart

        fig = radar_chart(
            tiny_adata,
            properties=["molecular_weight", "logp", "tpsa"],
            group_by="cluster", groups=["0", "1"],
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_raises_too_few_properties(self, tiny_adata):
        from embpy.pl.comparisons import radar_chart

        with pytest.raises(ValueError, match="at least 3"):
            radar_chart(
                tiny_adata,
                properties=["molecular_weight", "logp"],
                group_by="cluster",
            )


class TestStarCoordinates:
    def test_returns_figure(self, tiny_adata):
        from embpy.pl.comparisons import star_coordinates

        fig = star_coordinates(tiny_adata, obsm_key="X_model_a_pca", n_dims=5)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_categorical_color(self, tiny_adata):
        from embpy.pl.comparisons import star_coordinates

        fig = star_coordinates(
            tiny_adata, obsm_key="X_model_a_pca",
            n_dims=5, color_by="cluster",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_continuous_color(self, tiny_adata):
        from embpy.pl.comparisons import star_coordinates

        fig = star_coordinates(
            tiny_adata, obsm_key="X_model_a_pca",
            n_dims=5, color_by="logp",
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_n_dims_parameter(self, tiny_adata):
        from embpy.pl.comparisons import star_coordinates

        fig = star_coordinates(tiny_adata, obsm_key="X_model_a_pca", n_dims=3)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")


class TestTsneFeaturePanel:
    def test_returns_figure(self, tiny_adata):
        from embpy.pl.comparisons import tsne_feature_panel

        sc.pp.neighbors(tiny_adata, use_rep="X_model_a_pca", n_neighbors=5)
        sc.tl.tsne(tiny_adata, use_rep="X_model_a_pca")
        tiny_adata.obsm["X_tsne_X_model_a_pca"] = tiny_adata.obsm["X_tsne"].copy()

        fig = tsne_feature_panel(
            tiny_adata, obsm_key="X_model_a_pca",
            features=["molecular_weight", "logp"],
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_missing_feature_handled(self, tiny_adata):
        from embpy.pl.comparisons import tsne_feature_panel

        tiny_adata.obsm["X_tsne_X_model_a_pca"] = np.random.default_rng(0).normal(
            size=(tiny_adata.n_obs, 2)
        ).astype(np.float32)

        fig = tsne_feature_panel(
            tiny_adata, obsm_key="X_model_a_pca",
            features=["logp", "nonexistent_feature"],
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close("all")
