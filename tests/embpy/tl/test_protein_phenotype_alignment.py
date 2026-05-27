"""Tests for tl.protein.phenotype_alignment and tl.similarity.cross_modal_mantel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.tl.protein.phenotype_alignment import (
    _cross_modal_phenocopy,
    _pseudobulk_dual,
    alignment_summary,
    protein_phenotype_alignment,
)
from embpy.tl.similarity import cross_modal_mantel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_aligned_adata(
    n_perts: int = 20,
    d_sc: int = 32,
    d_prot: int = 64,
    noise: float = 0.05,
    seed: int = 0,
) -> AnnData:
    """Perturbation-level AnnData where both spaces share the same latent structure.

    Protein and scRNA-seq embeddings are both generated from the same
    underlying centroids, so they should be strongly cross-modally aligned.
    """
    rng = np.random.default_rng(seed)
    # shared latent centroids in a low-d space
    centroids = rng.standard_normal((n_perts, 8))

    # project into each modality with a random matrix + noise
    W_sc = rng.standard_normal((8, d_sc))
    W_prot = rng.standard_normal((8, d_prot))

    emb_sc = centroids @ W_sc + noise * rng.standard_normal((n_perts, d_sc))
    emb_prot = centroids @ W_prot + noise * rng.standard_normal((n_perts, d_prot))

    adata = AnnData(obs=pd.DataFrame(index=[f"gene_{i}" for i in range(n_perts)]))
    adata.obsm["X_sc"] = emb_sc.astype(np.float32)
    adata.obsm["X_prot"] = emb_prot.astype(np.float32)
    return adata


def _make_random_adata(
    n_perts: int = 20,
    d_sc: int = 32,
    d_prot: int = 64,
    seed: int = 1,
) -> AnnData:
    """Perturbation-level AnnData with independent random embeddings (null model)."""
    rng = np.random.default_rng(seed)
    adata = AnnData(obs=pd.DataFrame(index=[f"gene_{i}" for i in range(n_perts)]))
    adata.obsm["X_sc"] = rng.standard_normal((n_perts, d_sc)).astype(np.float32)
    adata.obsm["X_prot"] = rng.standard_normal((n_perts, d_prot)).astype(np.float32)
    return adata


def _make_cell_level_adata(
    n_perts: int = 10,
    n_reps: int = 5,
    d_sc: int = 16,
    d_prot: int = 32,
    seed: int = 2,
) -> AnnData:
    """Cell-level AnnData with replicate cells per perturbation."""
    rng = np.random.default_rng(seed)
    n_cells = n_perts * n_reps
    perts = [f"gene_{i}" for i in range(n_perts) for _ in range(n_reps)]

    X_sc = rng.standard_normal((n_cells, d_sc)).astype(np.float32)
    # protein: broadcast same vector per perturbation (realistic scenario)
    prot_centroids = rng.standard_normal((n_perts, d_prot)).astype(np.float32)
    X_prot = np.repeat(prot_centroids, n_reps, axis=0)

    adata = AnnData(obs=pd.DataFrame({"perturbation": perts}))
    adata.obsm["X_sc"] = X_sc
    adata.obsm["X_prot"] = X_prot
    return adata


# ---------------------------------------------------------------------------
# Tests: cross_modal_mantel
# ---------------------------------------------------------------------------

class TestCrossModalMantel:
    def test_identical_spaces_gives_rho_one(self):
        adata = _make_aligned_adata(n_perts=15, noise=0.0)
        # copy sc embedding to prot slot → distance matrices are identical
        adata.obsm["X_prot"] = adata.obsm["X_sc"].copy()
        rho, p = cross_modal_mantel(adata, "X_sc", "X_prot")
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_aligned_spaces_have_positive_rho(self):
        adata = _make_aligned_adata(n_perts=30, noise=0.1)
        rho, p = cross_modal_mantel(adata, "X_sc", "X_prot")
        assert rho > 0.3, f"Expected strong positive correlation, got {rho:.4f}"

    def test_random_spaces_near_zero_rho(self):
        # independent spaces → rho should be close to 0
        rng = np.random.default_rng(99)
        adata = AnnData(obs=pd.DataFrame(index=list(range(50))))
        adata.obsm["X_sc"] = rng.standard_normal((50, 20)).astype(np.float32)
        adata.obsm["X_prot"] = rng.standard_normal((50, 16)).astype(np.float32)
        rho, p = cross_modal_mantel(adata, "X_sc", "X_prot")
        assert abs(rho) < 0.4, f"Expected near-zero correlation, got {rho:.4f}"

    def test_returns_float_tuple(self):
        adata = _make_aligned_adata(n_perts=10)
        rho, p = cross_modal_mantel(adata, "X_sc", "X_prot")
        assert isinstance(rho, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_euclidean_metric(self):
        adata = _make_aligned_adata(n_perts=15)
        rho, p = cross_modal_mantel(adata, "X_sc", "X_prot", metric="euclidean")
        assert isinstance(rho, float)

    def test_invalid_metric_raises(self):
        adata = _make_aligned_adata()
        with pytest.raises(ValueError, match="Unknown metric"):
            cross_modal_mantel(adata, "X_sc", "X_prot", metric="manhattan")

    def test_missing_key_raises(self):
        adata = _make_aligned_adata()
        with pytest.raises(KeyError):
            cross_modal_mantel(adata, "X_sc", "nonexistent")

    def test_different_n_obs_raises(self):
        from unittest.mock import patch

        adata = _make_aligned_adata(n_perts=10)
        rng = np.random.default_rng(0)
        wrong = rng.standard_normal((5, 8)).astype(np.float64)
        # AnnData validates obsm shapes, so mock _get_embedding to simulate mismatch
        side_effects = [adata.obsm["X_sc"].astype(np.float64), wrong]
        with patch("embpy.tl.similarity._get_embedding", side_effect=side_effects):
            with pytest.raises(ValueError, match="same number of observations"):
                cross_modal_mantel(adata, "X_sc", "X_prot")

    def test_too_few_observations_raises(self):
        rng = np.random.default_rng(0)
        adata = AnnData(obs=pd.DataFrame(index=list(range(2))))
        adata.obsm["X_sc"] = rng.standard_normal((2, 4)).astype(np.float32)
        adata.obsm["X_prot"] = rng.standard_normal((2, 4)).astype(np.float32)
        with pytest.raises(ValueError, match="at least 3"):
            cross_modal_mantel(adata, "X_sc", "X_prot")


# ---------------------------------------------------------------------------
# Tests: _cross_modal_phenocopy
# ---------------------------------------------------------------------------

class TestCrossModalPhenocopy:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        emb_sc = rng.standard_normal((20, 32))
        emb_prot = rng.standard_normal((20, 64))
        out = _cross_modal_phenocopy(emb_sc, emb_prot, 10, (1, 2), (5, 10))
        assert "cross_modal_auroc_mad_1" in out
        assert "cross_modal_auroc_mad_2" in out
        assert "cross_modal_recall_5" in out
        assert "cross_modal_recall_10" in out

    def test_aligned_spaces_have_high_auroc(self):
        adata = _make_aligned_adata(n_perts=30, noise=0.05)
        emb_sc = adata.obsm["X_sc"].astype(np.float64)
        emb_prot = adata.obsm["X_prot"].astype(np.float64)
        out = _cross_modal_phenocopy(emb_sc, emb_prot, 20, (1,), (10,))
        assert out["cross_modal_auroc_mad_1"] > 0.6

    def test_identical_spaces_gives_perfect_recall(self):
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((15, 20))
        out = _cross_modal_phenocopy(emb, emb.copy(), 10, (1,), (5,))
        assert out["cross_modal_recall_5"] == pytest.approx(1.0)

    def test_recall_nan_when_k_exceeds_n(self):
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((5, 8))
        out = _cross_modal_phenocopy(emb, emb.copy(), None, (1,), (10,))
        assert np.isnan(out["cross_modal_recall_10"])

    def test_no_pca(self):
        rng = np.random.default_rng(0)
        emb_sc = rng.standard_normal((15, 8))
        emb_prot = rng.standard_normal((15, 8))
        out = _cross_modal_phenocopy(emb_sc, emb_prot, None, (1,), (5,))
        assert isinstance(out["cross_modal_auroc_mad_1"], float)

    def test_different_dims_handled(self):
        rng = np.random.default_rng(0)
        emb_sc = rng.standard_normal((20, 512))
        emb_prot = rng.standard_normal((20, 1280))
        # should not raise even though dimensions differ
        out = _cross_modal_phenocopy(emb_sc, emb_prot, 50, (1,), (5,))
        assert not np.isnan(out["cross_modal_auroc_mad_1"])


# ---------------------------------------------------------------------------
# Tests: _pseudobulk_dual
# ---------------------------------------------------------------------------

class TestPseudobulkDual:
    def test_reduces_to_perturbation_level(self):
        adata = _make_cell_level_adata(n_perts=8, n_reps=4)
        adata_pb = _pseudobulk_dual(adata, "perturbation", "X_sc", "X_prot", None, None)
        assert adata_pb.n_obs == 8

    def test_both_keys_present(self):
        adata = _make_cell_level_adata()
        adata_pb = _pseudobulk_dual(adata, "perturbation", "X_sc", "X_prot", None, None)
        assert "X_sc" in adata_pb.obsm
        assert "X_prot" in adata_pb.obsm

    def test_mean_is_correct(self):
        adata = _make_cell_level_adata(n_perts=3, n_reps=5)
        adata_pb = _pseudobulk_dual(adata, "perturbation", "X_sc", "X_prot", None, None)
        # gene_0 mean in sc space
        g0_cells = adata.obs["perturbation"] == "gene_0"
        expected_mean = adata.obsm["X_sc"][g0_cells].mean(axis=0)
        actual_mean = adata_pb.obsm["X_sc"][adata_pb.obs_names == "gene_0"][0]
        np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-5)

    def test_control_filtering(self):
        adata = _make_cell_level_adata(n_perts=5, n_reps=4)
        adata.obs["is_ctrl"] = (adata.obs["perturbation"] == "gene_0").values
        adata_pb = _pseudobulk_dual(
            adata, "perturbation", "X_sc", "X_prot",
            control_col="is_ctrl", control_ids={True},
        )
        assert "gene_0" not in list(adata_pb.obs_names)
        assert adata_pb.n_obs == 4  # 5 total - 1 control gene

    def test_missing_perturbation_col_raises(self):
        adata = _make_cell_level_adata()
        with pytest.raises(KeyError):
            _pseudobulk_dual(adata, "nonexistent", "X_sc", "X_prot", None, None)

    def test_missing_control_col_raises(self):
        adata = _make_cell_level_adata()
        with pytest.raises(KeyError, match="control_col"):
            _pseudobulk_dual(
                adata, "perturbation", "X_sc", "X_prot",
                control_col="nonexistent", control_ids={True},
            )


# ---------------------------------------------------------------------------
# Tests: protein_phenotype_alignment (full pipeline)
# ---------------------------------------------------------------------------

class TestProteinPhenotypeAlignment:
    def test_output_keys_complete(self):
        adata = _make_aligned_adata(n_perts=15)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        assert "mantel_rho" in out
        assert "mantel_pval" in out
        assert "knn_jaccard_mean" in out
        assert "cross_modal_auroc_mad_1" in out
        assert "cross_modal_recall_5" in out
        assert "n_perturbations" in out
        assert "knn_jaccard_per_perturbation" in out
        assert "perturbation_names" in out

    def test_n_perturbations_matches(self):
        n = 18
        adata = _make_aligned_adata(n_perts=n)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        assert out["n_perturbations"] == n

    def test_perturbation_names_length(self):
        adata = _make_aligned_adata(n_perts=12)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        assert len(out["perturbation_names"]) == 12
        assert len(out["knn_jaccard_per_perturbation"]) == 12

    def test_aligned_scores_higher_than_random(self):
        adata_aligned = _make_aligned_adata(n_perts=30, noise=0.05, seed=0)
        adata_random = _make_random_adata(n_perts=30, seed=1)

        out_aligned = protein_phenotype_alignment(adata_aligned, "X_prot", "X_sc")
        out_random = protein_phenotype_alignment(adata_random, "X_prot", "X_sc")

        assert out_aligned["mantel_rho"] > out_random["mantel_rho"]
        assert out_aligned["knn_jaccard_mean"] > out_random["knn_jaccard_mean"]

    def test_cell_level_adata_with_perturbation_col(self):
        adata = _make_cell_level_adata(n_perts=8, n_reps=5)
        out = protein_phenotype_alignment(
            adata, "X_prot", "X_sc", perturbation_col="perturbation"
        )
        assert out["n_perturbations"] == 8

    def test_cell_level_control_filtering(self):
        adata = _make_cell_level_adata(n_perts=6, n_reps=4)
        adata.obs["ctrl"] = (adata.obs["perturbation"] == "gene_0").values
        out = protein_phenotype_alignment(
            adata, "X_prot", "X_sc",
            perturbation_col="perturbation",
            control_col="ctrl",
            control_ids={True},
        )
        assert out["n_perturbations"] == 5  # 6 - 1 control
        assert "gene_0" not in out["perturbation_names"]

    def test_custom_k(self):
        adata = _make_aligned_adata(n_perts=20)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc", k=5)
        assert isinstance(out["knn_jaccard_mean"], float)

    def test_no_pca(self):
        adata = _make_aligned_adata(n_perts=15)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc", n_pca_components=None)
        assert isinstance(out["cross_modal_auroc_mad_1"], float)

    def test_missing_protein_key_raises(self):
        adata = _make_aligned_adata()
        with pytest.raises(KeyError):
            protein_phenotype_alignment(adata, "X_missing", "X_sc")

    def test_missing_sc_key_raises(self):
        adata = _make_aligned_adata()
        with pytest.raises(KeyError):
            protein_phenotype_alignment(adata, "X_prot", "X_missing")

    def test_missing_perturbation_col_raises(self):
        adata = _make_cell_level_adata()
        with pytest.raises(KeyError):
            protein_phenotype_alignment(
                adata, "X_prot", "X_sc", perturbation_col="nonexistent"
            )

    def test_too_few_perturbations_raises(self):
        rng = np.random.default_rng(0)
        adata = AnnData(obs=pd.DataFrame(index=["a", "b"]))
        adata.obsm["X_prot"] = rng.standard_normal((2, 8)).astype(np.float32)
        adata.obsm["X_sc"] = rng.standard_normal((2, 4)).astype(np.float32)
        with pytest.raises(ValueError, match="at least 3"):
            protein_phenotype_alignment(adata, "X_prot", "X_sc")

    def test_protein_sc_keys_in_metadata(self):
        adata = _make_aligned_adata(n_perts=10)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        assert out["protein_obsm_key"] == "X_prot"
        assert out["sc_obsm_key"] == "X_sc"


# ---------------------------------------------------------------------------
# Tests: alignment_summary
# ---------------------------------------------------------------------------

class TestAlignmentSummary:
    def test_single_result_returns_dataframe(self):
        adata = _make_aligned_adata(n_perts=10)
        out = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        df = alignment_summary(out)
        assert isinstance(df, pd.DataFrame)
        assert "mantel_rho" in df.index

    def test_multi_model_returns_wide_dataframe(self):
        adata = _make_aligned_adata(n_perts=10)
        out1 = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        out2 = protein_phenotype_alignment(adata, "X_prot", "X_sc")
        df = alignment_summary({"model_a": out1, "model_b": out2})
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"model_a", "model_b"}
        assert "mantel_rho" in df.columns
