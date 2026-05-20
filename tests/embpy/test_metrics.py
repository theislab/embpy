"""Tests for embpy.tl.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anndata import AnnData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def expression_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted expression matrices (n_perturbations, n_genes)."""
    n_pert, n_genes = 40, 100
    y_true = rng.standard_normal((n_pert, n_genes))
    y_pred = y_true + rng.standard_normal((n_pert, n_genes)) * 0.3
    return y_true, y_pred


@pytest.fixture()
def scalar_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted scalar targets (1-D)."""
    n = 50
    y_true = rng.uniform(0, 10, size=n)
    y_pred = y_true + rng.standard_normal(n) * 0.5
    return y_true, y_pred


@pytest.fixture()
def mean_control(rng: np.random.Generator) -> np.ndarray:
    """Mean control expression vector (n_genes,)."""
    return rng.standard_normal(100) * 0.1


@pytest.fixture()
def deg_adata(rng: np.random.Generator) -> AnnData:
    """AnnData suitable for DEG analysis (control + two perturbation groups)."""
    n_ctrl, n_a, n_b = 30, 25, 25
    n_genes = 50
    n_total = n_ctrl + n_a + n_b

    X = rng.standard_normal((n_total, n_genes)).astype(np.float32)
    X[n_ctrl : n_ctrl + n_a, :5] += 3.0
    X[n_ctrl + n_a :, 5:10] -= 3.0

    conditions = (
        ["control"] * n_ctrl + ["drug_A"] * n_a + ["drug_B"] * n_b
    )
    obs = pd.DataFrame(
        {"condition": pd.Categorical(conditions)},
        index=pd.Index([str(i) for i in range(n_total)]),
    )
    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(n_genes)]))
    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


class TestMSE:
    def test_perfect(self):
        from embpy.tl.metrics import mse

        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == 0.0

    def test_positive(self, scalar_pair):
        from embpy.tl.metrics import mse

        y_true, y_pred = scalar_pair
        assert mse(y_true, y_pred) > 0.0

    def test_2d(self, expression_pair):
        from embpy.tl.metrics import mse

        y_true, y_pred = expression_pair
        assert mse(y_true, y_pred) > 0.0


class TestR2:
    def test_perfect(self):
        from embpy.tl.metrics import r2

        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_1d(self, scalar_pair):
        from embpy.tl.metrics import r2

        y_true, y_pred = scalar_pair
        result = r2(y_true, y_pred)
        assert -1.0 <= result <= 1.0

    def test_2d_averages_per_gene(self, expression_pair):
        from embpy.tl.metrics import r2

        y_true, y_pred = expression_pair
        result = r2(y_true, y_pred)
        assert isinstance(result, float)

    def test_constant_column_skipped(self):
        from embpy.tl.metrics import r2

        y_true = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        y_pred = np.array([[1.1, 5.1], [1.9, 5.0], [3.2, 4.9]])
        result = r2(y_true, y_pred)
        assert isinstance(result, float)


class TestMeanCorrelation:
    def test_perfect_pearson(self):
        from embpy.tl.metrics import mean_correlation

        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert mean_correlation(y, y, method="pearson") == pytest.approx(1.0)

    def test_perfect_spearman(self):
        from embpy.tl.metrics import mean_correlation

        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert mean_correlation(y, y, method="spearman") == pytest.approx(1.0)

    def test_1d(self, scalar_pair):
        from embpy.tl.metrics import mean_correlation

        y_true, y_pred = scalar_pair
        assert -1.0 <= mean_correlation(y_true, y_pred) <= 1.0

    def test_2d(self, expression_pair):
        from embpy.tl.metrics import mean_correlation

        y_true, y_pred = expression_pair
        assert -1.0 <= mean_correlation(y_true, y_pred) <= 1.0

    def test_spearman_2d(self, expression_pair):
        from embpy.tl.metrics import mean_correlation

        y_true, y_pred = expression_pair
        assert -1.0 <= mean_correlation(y_true, y_pred, method="spearman") <= 1.0

    def test_constant_returns_zero(self):
        from embpy.tl.metrics import mean_correlation

        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mean_correlation(y_true, y_pred) == 0.0

    def test_close_predictions_high_corr(self):
        from embpy.tl.metrics import mean_correlation

        rng = np.random.default_rng(0)
        y_true = rng.standard_normal((10, 50))
        y_pred = y_true + rng.standard_normal((10, 50)) * 0.01
        assert mean_correlation(y_true, y_pred) > 0.95


class TestDeltaL2:
    def test_2d(self, expression_pair, mean_control):
        from embpy.tl.metrics import delta_l2

        y_true, y_pred = expression_pair
        result = delta_l2(y_true, y_pred, mean_control)
        assert "delta_l2_mae" in result
        assert "delta_l2_pearson" in result
        assert result["delta_l2_mae"] >= 0.0
        assert -1.0 <= result["delta_l2_pearson"] <= 1.0

    def test_1d_returns_nan(self, scalar_pair, mean_control):
        from embpy.tl.metrics import delta_l2

        y_true, y_pred = scalar_pair
        result = delta_l2(y_true, y_pred, mean_control)
        assert np.isnan(result["delta_l2_mae"])
        assert np.isnan(result["delta_l2_pearson"])


class TestComputeMetrics:
    def test_all_keys_present_1d(self, scalar_pair):
        from embpy.tl.metrics import compute_metrics

        y_true, y_pred = scalar_pair
        m = compute_metrics(y_true, y_pred)
        assert set(m.keys()) == {
            "mse", "r2", "pearson", "spearman",
            "delta_l2_mae", "delta_l2_pearson",
        }
        assert m["mse"] >= 0.0
        assert np.isnan(m["delta_l2_mae"])

    def test_all_keys_present_2d(self, expression_pair, mean_control):
        from embpy.tl.metrics import compute_metrics

        y_true, y_pred = expression_pair
        m = compute_metrics(y_true, y_pred, mean_control=mean_control)
        assert m["mse"] >= 0.0
        assert not np.isnan(m["delta_l2_mae"])
        assert not np.isnan(m["delta_l2_pearson"])


# ---------------------------------------------------------------------------
# Biological metrics
# ---------------------------------------------------------------------------


class TestGeneR2:
    def test_requires_2d(self):
        from embpy.tl.metrics import gene_r2

        with pytest.raises(ValueError, match="2-D"):
            gene_r2(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_returns_float(self, expression_pair):
        from embpy.tl.metrics import gene_r2

        y_true, y_pred = expression_pair
        result = gene_r2(y_true, y_pred)
        assert isinstance(result, float)


class TestFracCorrectDirection:
    def test_requires_2d(self):
        from embpy.tl.metrics import frac_correct_direction

        ctrl = np.zeros(5)
        with pytest.raises(ValueError, match="2-D"):
            frac_correct_direction(np.zeros(5), np.zeros(5), ctrl)

    def test_perfect_direction(self):
        from embpy.tl.metrics import frac_correct_direction

        ctrl = np.zeros(4)
        y_true = np.array([[1.0, -1.0, 2.0, -2.0]])
        y_pred = np.array([[0.5, -0.5, 1.0, -1.0]])
        assert frac_correct_direction(y_true, y_pred, ctrl) == 1.0

    def test_wrong_direction(self):
        from embpy.tl.metrics import frac_correct_direction

        ctrl = np.zeros(4)
        y_true = np.array([[1.0, -1.0, 1.0, -1.0]])
        y_pred = np.array([[-1.0, 1.0, -1.0, 1.0]])
        assert frac_correct_direction(y_true, y_pred, ctrl) == 0.0

    def test_with_threshold(self, expression_pair, mean_control):
        from embpy.tl.metrics import frac_correct_direction

        y_true, y_pred = expression_pair
        result = frac_correct_direction(y_true, y_pred, mean_control, threshold=0.5)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# DEG overlap & direction (unit-level, no scanpy needed)
# ---------------------------------------------------------------------------


class TestDegOverlap:
    def test_perfect_overlap(self):
        from embpy.tl.metrics import deg_overlap

        genes = ["A", "B", "C", "D"]
        result = deg_overlap(genes, genes)
        assert result["jaccard"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["n_shared"] == 4.0

    def test_no_overlap(self):
        from embpy.tl.metrics import deg_overlap

        result = deg_overlap(["A", "B"], ["C", "D"])
        assert result["jaccard"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["n_shared"] == 0.0

    def test_partial_overlap(self):
        from embpy.tl.metrics import deg_overlap

        result = deg_overlap(["A", "B", "C"], ["B", "C", "D"])
        assert result["jaccard"] == pytest.approx(2.0 / 4.0)
        assert result["precision"] == pytest.approx(2.0 / 3.0)
        assert result["recall"] == pytest.approx(2.0 / 3.0)
        assert result["n_shared"] == 2.0

    def test_empty_lists(self):
        from embpy.tl.metrics import deg_overlap

        result = deg_overlap([], [])
        assert result["jaccard"] == 0.0


class TestDegDirectionAgreement:
    def test_all_agree(self):
        from embpy.tl.metrics import deg_direction_agreement

        df_true = pd.DataFrame({"gene": ["A", "B", "C"], "logfoldchange": [1.0, -2.0, 0.5]})
        df_pred = pd.DataFrame({"gene": ["A", "B", "C"], "logfoldchange": [0.5, -1.0, 0.1]})
        result = deg_direction_agreement(df_true, df_pred)
        assert result["direction_agreement"] == pytest.approx(1.0)
        assert result["n_shared"] == 3.0

    def test_none_agree(self):
        from embpy.tl.metrics import deg_direction_agreement

        df_true = pd.DataFrame({"gene": ["A", "B"], "logfoldchange": [1.0, -1.0]})
        df_pred = pd.DataFrame({"gene": ["A", "B"], "logfoldchange": [-1.0, 1.0]})
        result = deg_direction_agreement(df_true, df_pred)
        assert result["direction_agreement"] == 0.0

    def test_no_shared_genes(self):
        from embpy.tl.metrics import deg_direction_agreement

        df_true = pd.DataFrame({"gene": ["A"], "logfoldchange": [1.0]})
        df_pred = pd.DataFrame({"gene": ["B"], "logfoldchange": [1.0]})
        result = deg_direction_agreement(df_true, df_pred)
        assert np.isnan(result["direction_agreement"])
        assert result["n_shared"] == 0.0


# ---------------------------------------------------------------------------
# Scanpy-dependent tests (rank_genes_groups, get_deg_dataframe, compare_deg)
# ---------------------------------------------------------------------------

scanpy = pytest.importorskip("scanpy")


class TestRankGenesGroups:
    def test_stores_results(self, deg_adata):
        from embpy.tl.metrics import rank_genes_groups

        rank_genes_groups(deg_adata, groupby="condition", reference="control")
        assert "rank_genes_groups" in deg_adata.uns

    def test_custom_key(self, deg_adata):
        from embpy.tl.metrics import rank_genes_groups

        rank_genes_groups(
            deg_adata, groupby="condition", reference="control",
            key_added="my_degs",
        )
        assert "my_degs" in deg_adata.uns

    def test_bad_groupby(self, deg_adata):
        from embpy.tl.metrics import rank_genes_groups

        with pytest.raises(KeyError, match="not found"):
            rank_genes_groups(deg_adata, groupby="nonexistent")


class TestGetDegDataframe:
    def test_returns_dataframe(self, deg_adata):
        from embpy.tl.metrics import get_deg_dataframe, rank_genes_groups

        rank_genes_groups(deg_adata, groupby="condition", reference="control")
        df = get_deg_dataframe(deg_adata, group="drug_A")
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns
        assert "score" in df.columns
        assert "pval_adj" in df.columns

    def test_n_top(self, deg_adata):
        from embpy.tl.metrics import get_deg_dataframe, rank_genes_groups

        rank_genes_groups(deg_adata, groupby="condition", reference="control")
        df = get_deg_dataframe(deg_adata, group="drug_A", n_top=5, pval_cutoff=None)
        assert len(df) <= 5

    def test_missing_key_raises(self, deg_adata):
        from embpy.tl.metrics import get_deg_dataframe

        with pytest.raises(KeyError, match="not found"):
            get_deg_dataframe(deg_adata, group="drug_A", key="nonexistent")


class TestCompareDeg:
    def test_identical_gives_perfect_overlap(self, deg_adata):
        from embpy.tl.metrics import compare_deg

        result = compare_deg(
            deg_adata, deg_adata,
            groupby="condition", reference="control",
            n_top_genes=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert "jaccard" in result.columns
        assert "direction_agreement" in result.columns
        assert (result["jaccard"] == 1.0).all()
        assert (result["direction_agreement"] >= 0.0).all()

    def test_cleans_up_temp_keys(self, deg_adata):
        from embpy.tl.metrics import compare_deg

        compare_deg(
            deg_adata, deg_adata.copy(),
            groupby="condition", reference="control",
        )
        assert "_embpy_deg_true" not in deg_adata.uns
        assert "_embpy_deg_pred" not in deg_adata.uns

    def test_subset_groups(self, deg_adata):
        from embpy.tl.metrics import compare_deg

        result = compare_deg(
            deg_adata, deg_adata,
            groupby="condition", reference="control",
            groups=["drug_A"],
        )
        assert len(result) == 1
        assert result["group"].iloc[0] == "drug_A"
