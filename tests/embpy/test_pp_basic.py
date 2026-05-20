"""Tests for embpy.pp.basic – PerturbationProcessor and reduce_embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.pp.basic import PerturbationProcessor, reduce_embeddings


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_resolver():
    """A GeneResolver mock so we never hit the network."""
    r = MagicMock()
    r.symbol_to_ensembl = MagicMock(return_value="ENSG00000141510")
    return r


@pytest.fixture
def processor(mock_resolver):
    return PerturbationProcessor(gene_resolver=mock_resolver)


@pytest.fixture
def mock_embedder(mock_resolver):
    """A BioEmbedder mock that returns deterministic embeddings."""
    embedder = MagicMock()
    embedder.gene_resolver = mock_resolver

    def _embed_gene(identifier, model, id_type="symbol", organism="human", pooling_strategy="mean", **kw):
        rng = np.random.RandomState(hash(identifier) % 2**31)
        return rng.randn(16).astype(np.float32)

    embedder.embed_gene = MagicMock(side_effect=_embed_gene)
    return embedder


@pytest.fixture
def processor_with_embedder(mock_embedder):
    return PerturbationProcessor(embedder=mock_embedder)


# =====================================================================
# normalize_gene_names
# =====================================================================


class TestNormalizeGeneNames:
    def test_ensembl_id_passes_through(self, processor):
        result = processor.normalize_gene_names(["ENSG00000141510"])
        assert result["ENSG00000141510"] == "ENSG00000141510"

    def test_ensembl_strips_version(self, processor):
        result = processor.normalize_gene_names(["ENSG00000141510.12"])
        assert result["ENSG00000141510.12"] == "ENSG00000141510"

    def test_symbol_resolved(self, processor, mock_resolver):
        mock_resolver.symbol_to_ensembl.return_value = "ENSG00000141510"
        result = processor.normalize_gene_names(["TP53"])
        assert result["TP53"] == "ENSG00000141510"

    def test_dna_sequence_returns_none(self, processor):
        result = processor.normalize_gene_names(["ACGTACGTACGTACGTACGTACGT"])
        assert result["ACGTACGTACGTACGTACGTACGT"] is None


# =====================================================================
# resolve_identifiers
# =====================================================================


class TestResolveIdentifiers:
    def test_auto_detect_mixed(self, processor, mock_resolver):
        mock_resolver.symbol_to_ensembl.return_value = "ENSG00000141510"
        ids = ["TP53", "ENSG00000141510", "CC(=O)O"]
        df = processor.resolve_identifiers(ids)
        assert len(df) == 3
        assert list(df.columns) == ["original_id", "canonical_id", "id_type", "resolved"]
        assert df.iloc[0]["id_type"] == "symbol"
        assert df.iloc[0]["resolved"] == True  # noqa: E712
        assert df.iloc[1]["id_type"] == "ensembl_id"
        assert df.iloc[1]["resolved"] == True  # noqa: E712
        assert df.iloc[2]["id_type"] == "smiles"
        assert df.iloc[2]["resolved"] == True  # noqa: E712

    def test_explicit_id_type(self, processor, mock_resolver):
        mock_resolver.symbol_to_ensembl.return_value = "ENSG001"
        df = processor.resolve_identifiers(["TP53"], id_type="symbol")
        assert df.iloc[0]["id_type"] == "symbol"
        assert df.iloc[0]["resolved"] == True  # noqa: E712

    def test_unresolvable_symbol(self, processor, mock_resolver):
        mock_resolver.symbol_to_ensembl.return_value = None
        df = processor.resolve_identifiers(["FAKEGENE"])
        assert df.iloc[0]["resolved"] == False  # noqa: E712


# =====================================================================
# build_embedding_matrix
# =====================================================================


class TestBuildEmbeddingMatrix:
    def test_requires_embedder(self, processor):
        with pytest.raises(ValueError, match="BioEmbedder"):
            processor.build_embedding_matrix(["TP53"], model="esm2_650M")

    def test_builds_adata_with_obsm(self, processor_with_embedder):
        adata = processor_with_embedder.build_embedding_matrix(
            ["TP53", "BRCA1"], model="esm2_650M"
        )
        assert isinstance(adata, AnnData)
        assert adata.n_obs == 2
        assert "X_esm2_650M" in adata.obsm
        assert adata.obsm["X_esm2_650M"].shape == (2, 16)
        assert "embedded" in adata.obs.columns
        assert all(adata.obs["embedded"])

    def test_custom_obsm_key(self, processor_with_embedder):
        adata = processor_with_embedder.build_embedding_matrix(
            ["TP53"], model="esm2_650M", obsm_key="X_custom"
        )
        assert "X_custom" in adata.obsm

    def test_handles_failed_embeddings(self):
        embedder = MagicMock()
        embedder.gene_resolver = MagicMock()
        call_count = 0

        def _embed_gene(identifier, **kw):
            nonlocal call_count
            call_count += 1
            if identifier == "FAKEGENE":
                raise RuntimeError("Not found")
            return np.ones(8, dtype=np.float32)

        embedder.embed_gene = MagicMock(side_effect=_embed_gene)
        proc = PerturbationProcessor(embedder=embedder)

        adata = proc.build_embedding_matrix(["TP53", "FAKEGENE"], model="esm2_650M")
        assert adata.n_obs == 2
        assert adata.obs["embedded"].iloc[0] is True or adata.obs["embedded"].iloc[0] == True  # noqa: E712
        assert adata.obs["embedded"].iloc[1] is False or adata.obs["embedded"].iloc[1] == False  # noqa: E712
        # Row for FAKEGENE should be all zeros
        assert np.allclose(adata.obsm["X_esm2_650M"][1], 0.0)

    def test_obs_metadata(self, processor_with_embedder):
        adata = processor_with_embedder.build_embedding_matrix(
            ["TP53"], model="esm2_650M", id_type="symbol", pooling_strategy="max"
        )
        assert adata.obs["model"].iloc[0] == "esm2_650M"
        assert adata.obs["pooling"].iloc[0] == "max"


# =====================================================================
# filter_failed_embeddings
# =====================================================================


class TestFilterFailedEmbeddings:
    def test_filters_correctly(self, processor):
        obs = pd.DataFrame({"identifier": ["A", "B", "C"], "embedded": [True, False, True]})
        obs.index = obs.index.astype(str)
        adata = AnnData(obs=obs)
        adata.obsm["X_emb"] = np.random.randn(3, 8).astype(np.float32)

        filtered = processor.filter_failed_embeddings(adata)
        assert filtered.n_obs == 2
        assert list(filtered.obs["identifier"]) == ["A", "C"]

    def test_raises_without_embedded_col(self, processor):
        adata = AnnData(obs=pd.DataFrame({"foo": [1]}))
        with pytest.raises(ValueError, match="embedded"):
            processor.filter_failed_embeddings(adata)

    def test_all_pass(self, processor):
        obs = pd.DataFrame({"identifier": ["A", "B"], "embedded": [True, True]})
        obs.index = obs.index.astype(str)
        adata = AnnData(obs=obs)
        adata.obsm["X_emb"] = np.random.randn(2, 8).astype(np.float32)
        filtered = processor.filter_failed_embeddings(adata)
        assert filtered.n_obs == 2


# =====================================================================
# combine_perturbation_spaces
# =====================================================================


class TestCombinePerturbationSpaces:
    def _make_adata(self, n, key="X_emb", dim=8):
        obs = pd.DataFrame({"identifier": [f"id_{i}" for i in range(n)]})
        obs.index = obs.index.astype(str)
        adata = AnnData(obs=obs)
        adata.obsm[key] = np.random.randn(n, dim).astype(np.float32)
        return adata

    def test_basic_combine(self, processor):
        a1 = self._make_adata(3)
        a2 = self._make_adata(2)
        combined = processor.combine_perturbation_spaces(a1, a2, labels=["genetic", "molecular"])
        assert combined.n_obs == 5
        assert "perturbation_type" in combined.obs.columns
        assert list(combined.obs["perturbation_type"].unique()) == ["genetic", "molecular"]

    def test_default_labels(self, processor):
        a1 = self._make_adata(2)
        a2 = self._make_adata(2)
        combined = processor.combine_perturbation_spaces(a1, a2)
        assert set(combined.obs["perturbation_type"].unique()) == {"set_0", "set_1"}

    def test_raises_on_empty(self, processor):
        with pytest.raises(ValueError, match="At least one"):
            processor.combine_perturbation_spaces()

    def test_raises_on_label_mismatch(self, processor):
        a1 = self._make_adata(2)
        with pytest.raises(ValueError, match="labels"):
            processor.combine_perturbation_spaces(a1, labels=["a", "b"])


# =====================================================================
# reduce_embeddings / PCA
# =====================================================================


class TestReduceEmbeddings:
    def _make_adata(self, n=20, dim=64, key="X_emb"):
        obs = pd.DataFrame({"identifier": [f"id_{i}" for i in range(n)]})
        obs.index = obs.index.astype(str)
        rng = np.random.RandomState(42)
        adata = AnnData(obs=obs)
        adata.obsm[key] = rng.randn(n, dim).astype(np.float32)
        return adata

    def test_basic_pca(self):
        adata = self._make_adata(n=30, dim=64, key="X_emb")
        result = PerturbationProcessor.reduce_embeddings(adata, obsm_key="X_emb", n_components=10)
        assert "X_emb_pca" in result.obsm
        assert result.obsm["X_emb_pca"].shape == (30, 10)

    def test_custom_output_key(self):
        adata = self._make_adata()
        PerturbationProcessor.reduce_embeddings(adata, "X_emb", n_components=5, output_key="X_reduced")
        assert "X_reduced" in adata.obsm
        assert adata.obsm["X_reduced"].shape[1] == 5

    def test_stores_params_in_uns(self):
        adata = self._make_adata()
        PerturbationProcessor.reduce_embeddings(adata, "X_emb", n_components=5)
        assert "X_emb_pca_params" in adata.uns
        params = adata.uns["X_emb_pca_params"]
        assert params["n_components"] == 5
        assert params["scaled"] is True
        assert 0.0 < params["total_variance_explained"] <= 1.0

    def test_no_scaling(self):
        adata = self._make_adata()
        PerturbationProcessor.reduce_embeddings(adata, "X_emb", n_components=5, scale=False)
        assert adata.uns["X_emb_pca_params"]["scaled"] is False

    def test_clamps_to_min_samples_features(self):
        adata = self._make_adata(n=5, dim=10)
        PerturbationProcessor.reduce_embeddings(adata, "X_emb", n_components=100)
        assert adata.obsm["X_emb_pca"].shape[1] == 5  # min(100, 5, 10)

    def test_missing_key_raises(self):
        adata = self._make_adata()
        with pytest.raises(KeyError, match="X_missing"):
            PerturbationProcessor.reduce_embeddings(adata, "X_missing")

    def test_convenience_function(self):
        adata = self._make_adata()
        result = reduce_embeddings(adata, "X_emb", n_components=3)
        assert "X_emb_pca" in result.obsm
        assert result.obsm["X_emb_pca"].shape[1] == 3

    def test_variance_explained_sums_correctly(self):
        adata = self._make_adata(n=50, dim=20)
        PerturbationProcessor.reduce_embeddings(adata, "X_emb", n_components=20)
        params = adata.uns["X_emb_pca_params"]
        # All components → should explain ~100% of variance
        assert params["total_variance_explained"] > 0.99

    def test_output_dtype_is_float32(self):
        adata = self._make_adata()
        PerturbationProcessor.reduce_embeddings(adata, "X_emb", n_components=5)
        assert adata.obsm["X_emb_pca"].dtype == np.float32
