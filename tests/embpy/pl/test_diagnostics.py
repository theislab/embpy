"""Tests for embpy.pl.diagnostics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

from embpy.pl.diagnostics import (
    category_centroid_similarity,
    knn_label_purity,
    within_vs_between_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _well_separated_adata(n_per_group: int = 8, d: int = 16, seed: int = 0) -> AnnData:
    """Synthetic AnnData with clearly separated functional groups."""
    rng = np.random.default_rng(seed)
    groups = ["alpha", "beta", "gamma", "delta"]
    centers = rng.normal(0, 5, size=(len(groups), d))
    X = np.vstack(
        [centers[i] + rng.normal(0, 1, (n_per_group, d)) for i in range(len(groups))]
    ).astype(np.float32)
    names = [f"{g}_{i}" for g in groups for i in range(n_per_group)]
    labels = [g for g in groups for _ in range(n_per_group)]

    adata = AnnData(
        X=np.zeros((len(names), 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"group": labels, "tissue": rng.choice(["liver", "brain"], len(names))},
            index=pd.Index(names),
        ),
    )
    adata.obsm["X_emb"] = X
    return adata


def _random_adata(n: int = 30, d: int = 16, n_groups: int = 4, seed: int = 1) -> AnnData:
    """Synthetic AnnData with random embeddings and random group labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    labels = rng.choice([f"g{i}" for i in range(n_groups)], n)
    names = [f"obs_{i}" for i in range(n)]
    adata = AnnData(
        X=np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame({"group": labels}, index=pd.Index(names)),
    )
    adata.obsm["X_emb"] = X
    return adata


@pytest.fixture()
def labelled_adata():
    return _well_separated_adata()


@pytest.fixture()
def random_adata():
    return _random_adata()


def _close(fig: Figure) -> None:
    plt.close(fig)


# ---------------------------------------------------------------------------
# within_vs_between_similarity
# ---------------------------------------------------------------------------


class TestWithinVsBetweenSimilarity:
    def test_returns_expected_keys(self, labelled_adata):
        res = within_vs_between_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        for key in ("mean_within", "mean_between", "n_within", "n_between", "p_value"):
            assert key in res
        plt.close("all")

    def test_within_higher_than_between_for_clusters(self, labelled_adata):
        res = within_vs_between_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        assert res["mean_within"] > res["mean_between"]
        assert res["p_value"] < 0.05
        plt.close("all")

    def test_within_close_to_between_for_random(self, random_adata):
        res = within_vs_between_similarity(
            random_adata, label_key="group", obsm_key="X_emb",
        )
        assert abs(res["mean_within"] - res["mean_between"]) < 0.5
        plt.close("all")

    def test_counts_match_pair_combinatorics(self, labelled_adata):
        res = within_vs_between_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        n = labelled_adata.n_obs
        assert res["n_within"] + res["n_between"] == n * (n - 1) // 2
        plt.close("all")

    def test_uses_provided_axes(self, labelled_adata):
        fig, ax = plt.subplots()
        within_vs_between_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb", ax=ax,
        )
        assert ax.get_title()
        plt.close(fig)

    def test_invalid_metric_raises(self, labelled_adata):
        with pytest.raises(ValueError, match="cosine"):
            within_vs_between_similarity(
                labelled_adata, label_key="group", obsm_key="X_emb", metric="euclidean",
            )

    def test_handles_nan_labels(self, labelled_adata):
        labels = labelled_adata.obs["group"].astype(object).copy()
        labels.iloc[:3] = np.nan
        labelled_adata.obs["group"] = labels
        res = within_vs_between_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        assert res["n_within"] > 0
        plt.close("all")

    def test_missing_label_raises(self, labelled_adata):
        with pytest.raises(KeyError):
            within_vs_between_similarity(
                labelled_adata, label_key="missing", obsm_key="X_emb",
            )

    def test_missing_obsm_raises(self, labelled_adata):
        with pytest.raises(KeyError):
            within_vs_between_similarity(
                labelled_adata, label_key="group", obsm_key="X_missing",
            )


# ---------------------------------------------------------------------------
# category_centroid_similarity
# ---------------------------------------------------------------------------


class TestCategoryCentroidSimilarity:
    def test_returns_dataframe(self, labelled_adata):
        df = category_centroid_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        assert isinstance(df, pd.DataFrame)
        n_cats = labelled_adata.obs["group"].nunique()
        assert df.shape == (n_cats, n_cats)
        plt.close("all")

    def test_diagonal_is_one(self, labelled_adata):
        df = category_centroid_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        np.testing.assert_allclose(np.diag(df.values), 1.0, atol=1e-5)
        plt.close("all")

    def test_symmetric(self, labelled_adata):
        df = category_centroid_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb",
        )
        np.testing.assert_allclose(df.values, df.values.T, atol=1e-6)
        plt.close("all")

    def test_uses_provided_axes(self, labelled_adata):
        fig, ax = plt.subplots()
        category_centroid_similarity(
            labelled_adata, label_key="group", obsm_key="X_emb", ax=ax,
        )
        plt.close(fig)

    def test_no_labels_raises(self):
        names = ["a", "b", "c"]
        adata = AnnData(
            X=np.zeros((3, 1), dtype=np.float32),
            obs=pd.DataFrame({"group": [None, None, None]}, index=pd.Index(names)),
        )
        adata.obsm["X_emb"] = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            category_centroid_similarity(adata, label_key="group", obsm_key="X_emb")


# ---------------------------------------------------------------------------
# knn_label_purity
# ---------------------------------------------------------------------------


class TestKnnLabelPurity:
    def test_returns_summary_keys(self, labelled_adata):
        out = knn_label_purity(
            labelled_adata, label_key="group", obsm_key="X_emb", k=3,
        )
        assert "__overall__" in out
        assert "__baseline__" in out
        plt.close("all")

    def test_high_purity_for_separated_clusters(self, labelled_adata):
        out = knn_label_purity(
            labelled_adata, label_key="group", obsm_key="X_emb", k=3,
        )
        assert out["__overall__"] > 0.9
        assert out["__overall__"] > out["__baseline__"]
        plt.close("all")

    def test_purity_close_to_baseline_for_random(self, random_adata):
        out = knn_label_purity(
            random_adata, label_key="group", obsm_key="X_emb", k=3,
        )
        assert abs(out["__overall__"] - out["__baseline__"]) < 0.25
        plt.close("all")

    def test_per_category_keys_present(self, labelled_adata):
        out = knn_label_purity(
            labelled_adata, label_key="group", obsm_key="X_emb", k=3,
        )
        for cat in labelled_adata.obs["group"].unique():
            assert cat in out

    def test_too_few_rows_raises(self):
        names = ["a", "b"]
        adata = AnnData(
            X=np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame({"group": ["x", "y"]}, index=pd.Index(names)),
        )
        adata.obsm["X_emb"] = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            knn_label_purity(adata, label_key="group", obsm_key="X_emb", k=5)

    def test_uses_provided_axes(self, labelled_adata):
        fig, ax = plt.subplots()
        knn_label_purity(
            labelled_adata, label_key="group", obsm_key="X_emb", k=3, ax=ax,
        )
        plt.close(fig)
