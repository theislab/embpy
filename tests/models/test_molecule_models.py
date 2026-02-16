"""Tests for molecule model wrappers (ChemBERTa, MolFormer, RDKit) using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.models.molecule_models import RDKitWrapper


class TestChembertaWrapper:
    def test_init_defaults(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        assert w.model_name == "DeepChem/ChemBERTa-77M-MTR"
        assert w.model_type == "molecule"
        assert w.model is None
        assert w.tokenizer is None

    def test_embed_without_load_raises(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        with pytest.raises(RuntimeError, match="Call load"):
            w.embed("CCO")

    def test_preprocess_without_load_raises(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        with pytest.raises(RuntimeError, match="Call load"):
            w._preprocess_smiles("CCO")

    def test_invalid_pooling_raises(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        w.tokenizer = MagicMock()
        w.max_len = 512

        with pytest.raises(ValueError, match="pooling_strategy"):
            w.embed("CCO", pooling_strategy="invalid")

    def test_embed_with_mocked_model(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        w.device = torch.device("cpu")
        w.max_len = 512

        hidden_dim = 768
        seq_len = 8
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.pooler_output = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        mock_tokenizer.side_effect = None
        w.tokenizer = mock_tokenizer

        w._token_length = MagicMock(return_value=5)

        result = w.embed("CCO", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)

    @pytest.mark.parametrize("strategy", ["cls", "mean", "max"])
    def test_embed_all_pooling_strategies(self, strategy):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        w.device = torch.device("cpu")
        w.max_len = 512

        hidden_dim = 768
        seq_len = 8
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.pooler_output = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        w.tokenizer = mock_tokenizer
        w._token_length = MagicMock(return_value=5)

        result = w.embed("CCO", pooling_strategy=strategy)
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)

    def test_skips_too_long_smiles(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        w.device = torch.device("cpu")
        w.model = MagicMock()
        w.tokenizer = MagicMock()
        w.max_len = 10
        w._token_length = MagicMock(return_value=100)

        result = w.embed("C" * 1000, pooling_strategy="mean")
        assert result is None

    def test_embed_batch(self):
        from embpy.models.molecule_models import ChembertaWrapper

        w = ChembertaWrapper()
        w.device = torch.device("cpu")
        w.max_len = 512

        hidden_dim = 768
        mock_emb = np.zeros(hidden_dim, dtype=np.float32)
        w.embed = MagicMock(return_value=mock_emb)

        results = w.embed_batch(["CCO", "CCC"])
        assert len(results) == 2


class TestMolformerWrapper:
    def test_init_defaults(self):
        from embpy.models.molecule_models import MolformerWrapper

        w = MolformerWrapper()
        assert w.model_name == "ibm/MoLFormer-XL-both-10pct"
        assert w.model_type == "molecule"

    def test_embed_without_load_raises(self):
        from embpy.models.molecule_models import MolformerWrapper

        w = MolformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("CCO")

    def test_embed_batch_without_load_raises(self):
        from embpy.models.molecule_models import MolformerWrapper

        w = MolformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["CCO"])

    def test_invalid_pooling_raises(self):
        from embpy.models.molecule_models import MolformerWrapper

        w = MolformerWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("CCO", pooling_strategy="invalid")

    def test_preprocess_without_tokenizer_raises(self):
        from embpy.models.molecule_models import MolformerWrapper

        w = MolformerWrapper()
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            w._preprocess_smiles("CCO")


class TestRDKitWrapper:
    """RDKit tests can run directly (CPU-only, no downloads)."""

    def test_init_defaults(self):
        w = RDKitWrapper()
        assert w.model_type == "molecule"
        assert w.n_bits == 2048
        assert w.fingerprint_type == "rdkit"

    def test_init_morgan(self):
        w = RDKitWrapper(fingerprint_type="morgan", radius=3, n_bits=1024)
        assert w.fingerprint_type == "morgan"
        assert w.radius == 3
        assert w.n_bits == 1024

    def test_embed_without_load_raises(self):
        w = RDKitWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("CCO")

    def test_load_and_embed(self):
        w = RDKitWrapper(fingerprint_type="rdkit")
        w.load(torch.device("cpu"))

        result = w.embed("CCO")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2048,)
        assert result.dtype == np.float32
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_embed_morgan(self):
        w = RDKitWrapper(fingerprint_type="morgan", radius=2, n_bits=1024)
        w.load(torch.device("cpu"))

        result = w.embed("CCO")
        assert result.shape == (1024,)

    def test_embed_invalid_smiles_raises(self):
        w = RDKitWrapper()
        w.load(torch.device("cpu"))

        with pytest.raises(ValueError, match="failed to parse"):
            w.embed("NOT_A_SMILES_XXX")

    def test_embed_batch(self):
        w = RDKitWrapper()
        w.load(torch.device("cpu"))

        results = w.embed_batch(["CCO", "CCC", "c1ccccc1"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (2048,)

    def test_different_molecules_different_fingerprints(self):
        w = RDKitWrapper(fingerprint_type="morgan")
        w.load(torch.device("cpu"))

        ethanol = w.embed("CCO")
        benzene = w.embed("c1ccccc1")
        assert not np.array_equal(ethanol, benzene)

    def test_same_molecule_same_fingerprint(self):
        w = RDKitWrapper(fingerprint_type="morgan")
        w.load(torch.device("cpu"))

        fp1 = w.embed("CCO")
        fp2 = w.embed("CCO")
        np.testing.assert_array_equal(fp1, fp2)

    def test_unknown_fingerprint_type_raises(self):
        w = RDKitWrapper(fingerprint_type="unknown_fp")
        w.load(torch.device("cpu"))

        with pytest.raises(ValueError, match="Unknown fingerprint type"):
            w.embed("CCO")
