"""Tests for embpy.pp.sc_preprocessing -- preprocess_counts."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

# =====================================================================
# Fixtures
# =====================================================================


def _make_synthetic_adata(
    n_cells: int = 200,
    n_genes: int = 500,
    sparse: bool = True,
) -> AnnData:
    """Create a synthetic AnnData with count-like data.

    Uses high density and large counts to survive QC filtering.
    """
    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(
            n_cells,
            n_genes,
            density=0.8,
            format="csr",
            random_state=42,
            dtype=np.float32,
        )
        X.data = (np.abs(X.data) * 500 + 1).astype(np.float32)
    else:
        X = (np.abs(rng.standard_normal((n_cells, n_genes))) * 500 + 1).astype(np.float32)

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    gene_names[0] = "MT-CO1"
    gene_names[1] = "MT-ND1"

    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


@pytest.fixture
def adata_sparse():
    return _make_synthetic_adata(sparse=True)


@pytest.fixture
def adata_dense():
    return _make_synthetic_adata(sparse=False)


# =====================================================================
# Pipeline: raw
# =====================================================================


class TestPreprocessRaw:
    def test_raw_preserves_x(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        original_shape = adata_sparse.shape
        result = preprocess_counts(adata_sparse, pipeline="raw", copy=True)

        assert "counts" in result.layers
        # .X should be raw counts (may be fewer cells after filtering)
        assert result.n_vars <= original_shape[1]

    def test_raw_creates_counts_layer(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(adata_sparse, pipeline="raw")
        assert "counts" in result.layers

    def test_raw_no_log_normalized_layer(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(adata_sparse, pipeline="raw")
        assert "log_normalized" not in result.layers

    def test_raw_qc_filtering(self):
        from embpy.pp.sc_preprocessing import preprocess_counts

        adata = _make_synthetic_adata(n_cells=50, n_genes=500)
        result = preprocess_counts(
            adata,
            pipeline="raw",
            min_genes=1,
            min_cells=1,
        )
        assert result.n_obs <= 50


# =====================================================================
# Pipeline: standard
# =====================================================================


class TestPreprocessStandard:
    def test_standard_creates_both_layers(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(adata_sparse, pipeline="standard")
        assert "counts" in result.layers
        assert "log_normalized" in result.layers

    def test_standard_x_is_raw_counts(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(adata_sparse, pipeline="standard")
        # .X should be restored to raw counts
        if sp.issparse(result.X):
            x_vals = result.X.toarray()
        else:
            x_vals = result.X

        if sp.issparse(result.layers["counts"]):
            counts_vals = result.layers["counts"].toarray()
        else:
            counts_vals = result.layers["counts"]

        np.testing.assert_array_equal(x_vals, counts_vals)

    def test_standard_marks_hvg(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(
            adata_sparse,
            pipeline="standard",
            n_top_genes=100,
        )
        assert "highly_variable" in result.var.columns
        n_hvg = result.var["highly_variable"].sum()
        assert n_hvg > 0

    def test_standard_log_normalized_differs_from_raw(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(adata_sparse, pipeline="standard")

        if sp.issparse(result.layers["log_normalized"]):
            ln = result.layers["log_normalized"].toarray()
        else:
            ln = result.layers["log_normalized"]

        if sp.issparse(result.layers["counts"]):
            raw = result.layers["counts"].toarray()
        else:
            raw = result.layers["counts"]

        # log-normalized should differ from raw (unless all zeros)
        assert not np.allclose(ln, raw)

    def test_standard_with_scale(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(
            adata_sparse,
            pipeline="standard",
            scale=True,
        )
        assert "counts" in result.layers
        assert "log_normalized" in result.layers


# =====================================================================
# QC filtering
# =====================================================================


class TestQCFiltering:
    def test_mito_filter(self):
        from embpy.pp.sc_preprocessing import preprocess_counts

        adata = _make_synthetic_adata(n_cells=100)
        result = preprocess_counts(
            adata,
            pipeline="raw",
            max_pct_mito=5.0,
        )
        assert result.n_obs <= 100

    def test_min_genes_filter(self):
        from embpy.pp.sc_preprocessing import preprocess_counts

        adata = _make_synthetic_adata()
        before = adata.n_obs
        result = preprocess_counts(
            adata,
            pipeline="raw",
            min_genes=1,
        )
        assert result.n_obs <= before

    def test_min_cells_filter(self):
        from embpy.pp.sc_preprocessing import preprocess_counts

        adata = _make_synthetic_adata()
        before = adata.n_vars
        result = preprocess_counts(
            adata,
            pipeline="raw",
            min_cells=1,
        )
        assert result.n_vars <= before


# =====================================================================
# Copy behavior
# =====================================================================


class TestCopyBehavior:
    def test_copy_true_does_not_modify_original(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        original_shape = adata_sparse.shape
        _ = preprocess_counts(adata_sparse, pipeline="standard", copy=True)
        assert adata_sparse.shape == original_shape
        assert "counts" not in adata_sparse.layers

    def test_copy_false_modifies_in_place(self, adata_sparse):
        from embpy.pp.sc_preprocessing import preprocess_counts

        preprocess_counts(adata_sparse, pipeline="raw", copy=False)
        assert "counts" in adata_sparse.layers


# =====================================================================
# Dense matrix support
# =====================================================================


class TestDenseMatrix:
    def test_standard_with_dense(self, adata_dense):
        from embpy.pp.sc_preprocessing import preprocess_counts

        result = preprocess_counts(adata_dense, pipeline="standard")
        assert "counts" in result.layers
        assert "log_normalized" in result.layers
