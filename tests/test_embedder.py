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
