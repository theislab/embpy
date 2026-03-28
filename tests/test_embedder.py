"""Tests for the BioEmbedder central class (using mocked model wrappers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.errors import ConfigError, IdentifierError, ModelNotFoundError


class TestBioEmbedderInit:
    """Tests for BioEmbedder initialization."""

    @patch("embpy.embedder.GeneResolver")
    def test_init_auto_device(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="auto")
        assert embedder.device is not None
        assert isinstance(embedder.device, torch.device)

    @patch("embpy.embedder.GeneResolver")
    def test_init_cpu_device(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        assert embedder.device == torch.device("cpu")

    @patch("embpy.embedder.GeneResolver")
    def test_init_torch_device(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        dev = torch.device("cpu")
        embedder = BioEmbedder(device=dev)
        assert embedder.device == dev

    @patch("embpy.embedder.GeneResolver")
    def test_init_invalid_device_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        with pytest.raises(ConfigError, match="Invalid device"):
            BioEmbedder(device=12345)

    @patch("embpy.embedder.GeneResolver")
    def test_init_local_backend_requires_paths(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        with pytest.raises(ConfigError, match="mart_file"):
            BioEmbedder(device="cpu", resolver_backend="local")

    @patch("embpy.embedder.GeneResolver")
    def test_list_available_models(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        models = embedder.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "enformer_human_rough" in models
        assert "esm2_650M" in models
        assert "chemberta2MTR" in models


class TestBioEmbedderEmbedGene:
    """Tests for embed_gene routing logic."""

    @patch("embpy.embedder.GeneResolver")
    def test_embed_gene_dna_model(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "dna"
        mock_wrapper.embed.return_value = np.zeros(3072, dtype=np.float32)

        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.gene_resolver.get_dna_sequence = MagicMock(return_value="ACGTACGT")

        result = embedder.embed_gene("TP53", model="enformer_human_rough")
        assert isinstance(result, np.ndarray)
        mock_wrapper.embed.assert_called_once()

    @patch("embpy.embedder.GeneResolver")
    def test_embed_gene_protein_model(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)

        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.gene_resolver.get_protein_sequence = MagicMock(return_value="MTEYKLVVVG")

        result = embedder.embed_gene("TP53", model="esm2_650M")
        assert isinstance(result, np.ndarray)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_gene_text_model(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "text"
        mock_wrapper.embed.return_value = np.zeros(384, dtype=np.float32)

        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.gene_resolver.get_gene_description = MagicMock(return_value="Gene TP53, tumor suppressor")

        result = embedder.embed_gene("TP53", model="minilm_l6_v2")
        assert isinstance(result, np.ndarray)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_gene_dna_not_found_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "dna"

        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.gene_resolver.get_dna_sequence = MagicMock(return_value=None)

        with pytest.raises(IdentifierError, match="DNA not found"):
            embedder.embed_gene("FAKEGENE", model="enformer_human_rough")

    @patch("embpy.embedder.GeneResolver")
    def test_embed_gene_unsupported_model_type(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "unknown"

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        with pytest.raises(ValueError, match="Unsupported model type"):
            embedder.embed_gene("TP53", model="some_model")


class TestBioEmbedderEmbedMolecule:
    """Tests for embed_molecule and embed_molecules_batch."""

    @patch("embpy.embedder.GeneResolver")
    def test_embed_molecule_valid_smiles(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "molecule"
        mock_wrapper.embed.return_value = np.zeros(768, dtype=np.float32)

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        result = embedder.embed_molecule("CCO", model="chemberta2MTR")
        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_molecule_invalid_smiles_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "molecule"

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        with pytest.raises(ValueError, match="Invalid SMILES"):
            embedder.embed_molecule("NOT_A_SMILES_XXX_123", model="chemberta2MTR")

    @patch("embpy.embedder.GeneResolver")
    def test_embed_molecule_wrong_model_type_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "dna"

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        with pytest.raises(ValueError, match="not a molecule embedder"):
            embedder.embed_molecule("CCO", model="enformer_human_rough")

    @patch("embpy.embedder.GeneResolver")
    def test_embed_molecules_batch(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "molecule"
        mock_wrapper.embed_batch.return_value = [
            np.zeros(768, dtype=np.float32),
            np.zeros(768, dtype=np.float32),
        ]

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        results = embedder.embed_molecules_batch(["CCO", "CCC"], model="chemberta2MTR")
        assert len(results) == 2


class TestBioEmbedderEmbedText:
    """Tests for embed_text and embed_texts_batch."""

    @patch("embpy.embedder.GeneResolver")
    def test_embed_text(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "text"
        mock_wrapper.embed.return_value = np.zeros(384, dtype=np.float32)

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        result = embedder.embed_text("Hello world", model="minilm_l6_v2")
        assert isinstance(result, np.ndarray)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_text_wrong_model_type_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "dna"

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        with pytest.raises(ValueError, match="not a text embedder"):
            embedder.embed_text("Hello", model="enformer")

    @patch("embpy.embedder.GeneResolver")
    def test_embed_texts_batch(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "text"
        mock_wrapper.embed_batch.return_value = [
            np.zeros(384, dtype=np.float32),
            np.zeros(384, dtype=np.float32),
        ]

        embedder._get_model = MagicMock(return_value=mock_wrapper)

        results = embedder.embed_texts_batch(["Hello", "World"], model="minilm_l6_v2")
        assert len(results) == 2


class TestBioEmbedderGetModel:
    """Tests for model caching and loading logic."""

    @patch("embpy.embedder.GeneResolver")
    def test_model_cache_hit(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "text"
        embedder.model_cache["test_model"] = mock_wrapper

        result = embedder._get_model("test_model")
        assert result is mock_wrapper

    @patch("embpy.embedder.GeneResolver")
    def test_unknown_model_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        with pytest.raises((ModelNotFoundError, RuntimeError)):
            embedder._get_model("completely_nonexistent_model_xyz_123")


# =====================================================================
# embed_protein / embed_proteins_batch
# =====================================================================


class TestEmbedProtein:
    @patch("embpy.embedder.GeneResolver")
    def test_embed_protein_canonical(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        embedder.protein_resolver.get_canonical_sequence = MagicMock(
            return_value="MEEPQSDPSVEPPLSQETFSDLWKLLP",
        )

        result = embedder.embed_protein(
            "TP53", model="esm2_650M", id_type="symbol", isoform="canonical",
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1280,)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_protein_all_isoforms(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        embedder.protein_resolver.get_isoforms = MagicMock(
            return_value={
                "P04637": "MEEPQSDP",
                "P04637-2": "MEEPQ",
            },
        )

        result = embedder.embed_protein(
            "TP53", model="esm2_650M", id_type="symbol", isoform="all",
        )
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "P04637" in result
        assert result["P04637"].shape == (1280,)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_protein_sequence_direct(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(640, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        result = embedder.embed_protein(
            "MEEPQSDP", model="esm2_35M", id_type="sequence",
        )
        assert isinstance(result, np.ndarray)

    @patch("embpy.embedder.GeneResolver")
    def test_embed_protein_wrong_model_type_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "dna"
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        with pytest.raises(ValueError, match="not a protein model"):
            embedder.embed_protein("TP53", model="enformer")

    @patch("embpy.embedder.GeneResolver")
    def test_embed_protein_not_found_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        embedder.protein_resolver.get_canonical_sequence = MagicMock(
            return_value=None,
        )

        with pytest.raises(IdentifierError):
            embedder.embed_protein("FAKEGENE", model="esm2_650M")


class TestEmbedProteinsBatch:
    @patch("embpy.embedder.GeneResolver")
    def test_batch_canonical(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        embedder.protein_resolver.get_canonical_sequences_batch = MagicMock(
            return_value={"TP53": "MEEPQ", "BRCA1": "MDLSA"},
        )

        result = embedder.embed_proteins_batch(
            ["TP53", "BRCA1"], model="esm2_650M", isoform="canonical",
        )
        assert isinstance(result, dict)
        assert len(result) == 2

    @patch("embpy.embedder.GeneResolver")
    def test_batch_all_isoforms(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)

        embedder.protein_resolver.get_isoforms_batch = MagicMock(
            return_value={
                "TP53": {"P04637": "MEEPQ", "P04637-2": "MEE"},
                "BRCA1": {"P38398": "MDLSA"},
            },
        )

        result = embedder.embed_proteins_batch(
            ["TP53", "BRCA1"], model="esm2_650M", isoform="all",
        )
        assert isinstance(result, dict)
        assert len(result) == 2
        assert isinstance(result["TP53"], dict)
        assert len(result["TP53"]) == 2


# =====================================================================
# embed_cells
# =====================================================================


class TestEmbedCells:
    def _make_adata(self):
        from anndata import AnnData

        rng = np.random.default_rng(42)
        n, g = 50, 200
        X = np.abs(rng.standard_normal((n, g))).astype(np.float32) * 100
        adata = AnnData(X=X)
        adata.var_names = [f"Gene_{i}" for i in range(g)]
        adata.obs_names = [f"cell_{i}" for i in range(n)]
        return adata

    @patch("embpy.embedder.GeneResolver")
    def test_embed_cells_pca(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata()

        result = embedder.embed_cells(
            adata, models=["pca"],
            preprocessing="standard",
            n_pca_components=10,
            n_top_genes=50,
        )
        assert "X_pca" in result.obsm
        assert result.obsm["X_pca"].shape[1] == 10
        assert "counts" in result.layers

    @patch("embpy.embedder.GeneResolver")
    def test_embed_cells_no_preprocessing(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata()

        result = embedder.embed_cells(
            adata, models=["pca"],
            preprocessing="none",
            n_pca_components=5,
        )
        assert "X_pca" in result.obsm

    @patch("embpy.embedder.GeneResolver")
    def test_embed_cells_unknown_model_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata()

        with pytest.raises(ValueError, match="Unknown single-cell model"):
            embedder.embed_cells(adata, models=["nonexistent_model"])

    @patch("embpy.embedder.GeneResolver")
    def test_embed_cells_metadata(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata()

        result = embedder.embed_cells(
            adata, models=["pca"], preprocessing="standard",
            n_pca_components=10, n_top_genes=50,
        )
        assert "embpy_cell_embeddings" in result.uns
        assert "pca" in result.uns["embpy_cell_embeddings"]

    @patch("embpy.embedder.GeneResolver")
    def test_embed_cells_copy(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata()
        original_shape = adata.shape

        _ = embedder.embed_cells(adata, models=["pca"], copy=True,
                                 n_pca_components=5, n_top_genes=50,
                                 preprocessing="standard")
        assert adata.shape == original_shape
        assert "X_pca" not in adata.obsm


# =====================================================================
# embed_adata
# =====================================================================


class TestEmbedAdata:
    def _make_adata_with_perts(self):
        import pandas as pd
        from anndata import AnnData

        rng = np.random.default_rng(42)
        n, g = 50, 200
        X = np.abs(rng.standard_normal((n, g))).astype(np.float32) * 100
        perts = rng.choice(["TP53", "BRCA1", "control"], size=n)
        adata = AnnData(
            X=X,
            obs=pd.DataFrame({"perturbation": perts}),
        )
        adata.var_names = [f"Gene_{i}" for i in range(g)]
        adata.obs_names = [f"cell_{i}" for i in range(n)]
        return adata

    @patch("embpy.embedder.GeneResolver")
    def test_cell_models_only(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata_with_perts()

        result = embedder.embed_adata(
            adata, cell_models=["pca"],
            preprocessing="standard",
            n_pca_components=10, n_top_genes=50,
        )
        assert "X_pca" in result.obsm
        assert "embpy_embeddings" in result.uns

    @patch("embpy.embedder.GeneResolver")
    def test_perturbation_models_only(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata_with_perts()

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.protein_resolver.get_canonical_sequence = MagicMock(
            return_value="MEEPQSDP",
        )

        result = embedder.embed_adata(
            adata,
            perturbation_models=["esm2_650M"],
            perturbation_column="perturbation",
            perturbation_type="symbol",
            preprocessing="none",
        )
        assert "X_esm2_650M" in result.obsm
        assert result.obsm["X_esm2_650M"].shape[0] == adata.n_obs

    @patch("embpy.embedder.GeneResolver")
    def test_missing_perturbation_column_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata_with_perts()

        with pytest.raises(ValueError, match="perturbation_column is required"):
            embedder.embed_adata(
                adata, perturbation_models=["esm2_650M"],
                preprocessing="none",
            )

    @patch("embpy.embedder.GeneResolver")
    def test_wrong_column_name_raises(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata_with_perts()

        with pytest.raises(ValueError, match="not found in adata.obs"):
            embedder.embed_adata(
                adata,
                perturbation_models=["esm2_650M"],
                perturbation_column="nonexistent_column",
                preprocessing="none",
            )

    @patch("embpy.embedder.GeneResolver")
    def test_combined_cell_and_perturbation(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata_with_perts()

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(1280, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.protein_resolver.get_canonical_sequence = MagicMock(
            return_value="MEEPQSDP",
        )

        result = embedder.embed_adata(
            adata,
            cell_models=["pca"],
            perturbation_models=["esm2_650M"],
            perturbation_column="perturbation",
            perturbation_type="symbol",
            preprocessing="standard",
            n_pca_components=10,
            n_top_genes=50,
        )
        assert "X_pca" in result.obsm
        assert "X_esm2_650M" in result.obsm
        assert "embpy_embeddings" in result.uns

    @patch("embpy.embedder.GeneResolver")
    def test_metadata_structure(self, mock_resolver_cls):
        from embpy.embedder import BioEmbedder

        embedder = BioEmbedder(device="cpu")
        adata = self._make_adata_with_perts()

        mock_wrapper = MagicMock()
        mock_wrapper.model_type = "protein"
        mock_wrapper.embed.return_value = np.zeros(64, dtype=np.float32)
        embedder._get_model = MagicMock(return_value=mock_wrapper)
        embedder.protein_resolver.get_canonical_sequence = MagicMock(
            return_value="MEEPQSDP",
        )

        result = embedder.embed_adata(
            adata,
            perturbation_models=["esm2_650M"],
            perturbation_column="perturbation",
            perturbation_type="symbol",
            preprocessing="none",
        )
        meta = result.uns["embpy_embeddings"]["esm2_650M"]
        assert "obsm_key" in meta
        assert "embedding_dim" in meta
        assert "n_perturbations_embedded" in meta
        assert "n_cells_mapped" in meta
        assert meta["type"] == "perturbation"
