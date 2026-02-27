"""Tests for embpy.tl.benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anndata import AnnData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalar_adata() -> AnnData:
    """AnnData with pre-computed embeddings and a scalar target (IC50)."""
    rng = np.random.default_rng(42)
    n = 60

    X_expr = rng.standard_normal((n, 20)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "drug_name": [f"drug_{i}" for i in range(n)],
            "ic50": rng.uniform(0, 10, size=n),
            "perturbation_type": ["chemical"] * n,
        },
        index=[str(i) for i in range(n)],
    )

    adata = AnnData(X=X_expr, obs=obs)
    adata.obsm["X_emb"] = rng.standard_normal((n, 64)).astype(np.float32)
    return adata


@pytest.fixture()
def expression_adata() -> AnnData:
    """AnnData with embeddings, expression matrix, and controls."""
    rng = np.random.default_rng(0)
    n_pert = 50
    n_ctrl = 10
    n_total = n_pert + n_ctrl
    n_genes = 30

    X_expr = rng.standard_normal((n_total, n_genes)).astype(np.float32)
    X_expr[:n_ctrl] *= 0.1  # controls have lower variance

    conditions = ["perturbation"] * n_pert + ["DMSO"] * n_ctrl
    obs = pd.DataFrame(
        {
            "gene_name": [f"gene_{i}" for i in range(n_pert)] + [f"ctrl_{i}" for i in range(n_ctrl)],
            "condition": conditions,
        },
        index=[str(i) for i in range(n_total)],
    )

    adata = AnnData(X=X_expr, obs=obs)
    adata.obsm["X_emb"] = rng.standard_normal((n_total, 64)).astype(np.float32)
    adata.obsm["X_emb_alt"] = rng.standard_normal((n_total, 128)).astype(np.float32)
    return adata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarkScalarTarget:
    """Benchmark predicting a scalar .obs column (e.g. IC50)."""

    def test_all_models(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
        )
        assert isinstance(results, pd.DataFrame)
        assert list(results.index) == ["linear", "ridge", "knn", "random_forest"]
        assert "mse" in results.columns
        assert "r2" in results.columns
        assert "pearson" in results.columns
        assert "spearman" in results.columns
        assert results["mse"].notna().all()

    def test_stored_in_uns(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
        )
        assert "benchmark_results" in scalar_adata.uns

    def test_no_delta_l2_for_scalar(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
        )
        assert results["delta_l2_mae"].isna().all()
        assert results["delta_l2_pearson"].isna().all()


class TestBenchmarkExpressionTarget:
    """Benchmark predicting expression (.X)."""

    def test_expression_metrics(self, expression_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            expression_adata,
            perturbation_column="gene_name",
            perturbation_type="genetic",
            target="X",
            obsm_key="X_emb",
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4
        assert "mse" in results.columns
        assert "r2" in results.columns
        assert "pearson" in results.columns
        assert "spearman" in results.columns


class TestBenchmarkWithControls:
    """Benchmark with controls for delta L2."""

    def test_delta_l2_computed(self, expression_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            expression_adata,
            perturbation_column="gene_name",
            perturbation_type="genetic",
            target="X",
            obsm_key="X_emb",
            control_column="condition",
            control_value="DMSO",
        )
        assert results["delta_l2_mae"].notna().all()
        assert results["delta_l2_pearson"].notna().all()
        assert (results["delta_l2_mae"] >= 0).all()


class TestBenchmarkNoControlsNoDelta:
    """Without controls, delta L2 should be NaN."""

    def test_no_controls(self, expression_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            expression_adata,
            perturbation_column="gene_name",
            perturbation_type="genetic",
            target="X",
            obsm_key="X_emb",
        )
        assert results["delta_l2_mae"].isna().all()
        assert results["delta_l2_pearson"].isna().all()


class TestBenchmarkWithDimReduction:
    """Verify PCA is applied before benchmarking."""

    def test_pca_applied(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            reduce_dim=16,
        )
        assert isinstance(results, pd.DataFrame)
        assert "X_emb_pca16" in scalar_adata.obsm
        assert scalar_adata.obsm["X_emb_pca16"].shape[1] == 16


class TestBenchmarkSelectedModels:
    """Only selected models should appear in the results."""

    def test_subset(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge", "knn"],
        )
        assert list(results.index) == ["ridge", "knn"]
        assert len(results) == 2


class TestBenchmarkInvalidTarget:
    """Non-existent target column should raise KeyError."""

    def test_bad_target(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        with pytest.raises(KeyError, match="Target"):
            benchmark_embeddings(
                scalar_adata,
                perturbation_column="drug_name",
                perturbation_type="chemical",
                target="nonexistent_column",
                obsm_key="X_emb",
            )


class TestBenchmarkInvalidModel:
    """Unknown model name should raise ValueError."""

    def test_bad_model(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        with pytest.raises(ValueError, match="Unknown model"):
            benchmark_embeddings(
                scalar_adata,
                perturbation_column="drug_name",
                perturbation_type="chemical",
                target="ic50",
                obsm_key="X_emb",
                models=["xgboost"],
            )


class TestBenchmarkObsNames:
    """perturbation_column='obs_names' should work."""

    def test_obs_names(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="obs_names",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge"],
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Rigorous mode tests
# ---------------------------------------------------------------------------


class TestRigorousMode:
    """Tests for mode='rigorous' (5-fold CV + grid search)."""

    def test_rigorous_columns(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge", "knn"],
            mode="rigorous",
        )
        assert isinstance(results, pd.DataFrame)
        assert list(results.index) == ["ridge", "knn"]
        assert "r2_mean" in results.columns
        assert "r2_std" in results.columns
        assert "pearson_mean" in results.columns
        assert "pearson_std" in results.columns
        assert "mse_mean" in results.columns
        assert "mse_std" in results.columns
        assert "best_params" in results.columns

    def test_rigorous_std_positive(self, scalar_adata):
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge"],
            mode="rigorous",
        )
        std_cols = [c for c in results.columns if c.endswith("_std")]
        for col in std_cols:
            vals = results[col].dropna()
            assert (vals >= 0).all(), f"{col} has negative values"


# ---------------------------------------------------------------------------
# Plotting tests
# ---------------------------------------------------------------------------


class TestPlotBenchmark:
    """Tests for pl.plot_benchmark and pl.plot_benchmark_comparison."""

    def test_plot_quick(self, scalar_adata):
        from matplotlib.figure import Figure

        from embpy.pl.benchmark import plot_benchmark
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge", "knn"],
            mode="quick",
        )
        fig = plot_benchmark(results)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_plot_rigorous(self, scalar_adata):
        from matplotlib.figure import Figure

        from embpy.pl.benchmark import plot_benchmark
        from embpy.tl.benchmark import benchmark_embeddings

        results = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge", "knn"],
            mode="rigorous",
        )
        fig = plot_benchmark(results)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_plot_comparison(self, scalar_adata):
        from matplotlib.figure import Figure

        from embpy.pl.benchmark import plot_benchmark_comparison
        from embpy.tl.benchmark import benchmark_embeddings

        results_a = benchmark_embeddings(
            scalar_adata,
            perturbation_column="drug_name",
            perturbation_type="chemical",
            target="ic50",
            obsm_key="X_emb",
            models=["ridge"],
            mode="quick",
        )
        results_b = results_a.copy()
        fig = plot_benchmark_comparison({"Emb A": results_a, "Emb B": results_b})
        assert isinstance(fig, Figure)
        plt_close(fig)


def plt_close(fig) -> None:
    """Close a figure to free resources."""
    import matplotlib.pyplot as _plt

    _plt.close(fig)
