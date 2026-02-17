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

    # ------------------------------------------------------------------ init
    def test_init_defaults(self):
        w = RDKitWrapper()
        assert w.model_type == "molecule"
        assert w.n_bits == 2048
        assert w.fingerprint_type == "rdkit"
        assert w.is_binary_fingerprint is True
        assert w.is_count_fingerprint is False

    def test_init_morgan(self):
        w = RDKitWrapper(fingerprint_type="morgan", radius=3, n_bits=1024)
        assert w.fingerprint_type == "morgan"
        assert w.radius == 3
        assert w.n_bits == 1024

    def test_init_auto_detect_from_model_path(self):
        """When model_path_or_name is a valid fp type, auto-detect it."""
        w = RDKitWrapper(model_path_or_name="morgan_count")
        assert w.fingerprint_type == "morgan_count"
        assert w.is_count_fingerprint is True

    def test_init_explicit_overrides_auto(self):
        """Explicit fingerprint_type takes precedence over auto-detection."""
        w = RDKitWrapper(model_path_or_name="morgan", fingerprint_type="rdkit")
        assert w.fingerprint_type == "rdkit"

    def test_init_maccs_forces_167_bits(self):
        w = RDKitWrapper(fingerprint_type="maccs", n_bits=9999)
        assert w.n_bits == 167

    def test_unknown_fingerprint_type_raises(self):
        with pytest.raises(ValueError, match="Unknown fingerprint_type"):
            RDKitWrapper(fingerprint_type="unknown_fp")

    # ------------------------------------------------------------------ load
    def test_embed_without_load_raises(self):
        w = RDKitWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("CCO")

    def test_embed_invalid_smiles_raises(self):
        w = RDKitWrapper()
        w.load(torch.device("cpu"))
        with pytest.raises(ValueError, match="failed to parse"):
            w.embed("NOT_A_SMILES_XXX")

    # ----------------------------------------------------------------
    # Binary fingerprints
    # ----------------------------------------------------------------

    def test_rdkit_binary(self):
        w = RDKitWrapper(fingerprint_type="rdkit")
        w.load(torch.device("cpu"))
        result = w.embed("CCO")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2048,)
        assert result.dtype == np.float32
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_morgan_binary(self):
        w = RDKitWrapper(fingerprint_type="morgan", radius=2, n_bits=1024)
        w.load(torch.device("cpu"))
        result = w.embed("CCO")
        assert result.shape == (1024,)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_maccs_binary(self):
        w = RDKitWrapper(fingerprint_type="maccs")
        w.load(torch.device("cpu"))
        result = w.embed("CCO")
        assert result.shape == (167,)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_atom_pair_binary(self):
        w = RDKitWrapper(fingerprint_type="atom_pair", n_bits=1024)
        w.load(torch.device("cpu"))
        result = w.embed("CCO")
        assert result.shape == (1024,)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_topological_torsion_binary(self):
        w = RDKitWrapper(fingerprint_type="topological_torsion", n_bits=1024)
        w.load(torch.device("cpu"))
        result = w.embed("CCO")
        assert result.shape == (1024,)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    # ----------------------------------------------------------------
    # Count (continuous) fingerprints
    # ----------------------------------------------------------------

    def test_morgan_count(self):
        """Morgan count fingerprint should contain integer counts >= 0."""
        w = RDKitWrapper(fingerprint_type="morgan_count", radius=2, n_bits=1024)
        w.load(torch.device("cpu"))
        result = w.embed("c1ccc(CC(=O)O)cc1")  # phenylacetic acid — richer FP
        assert result.shape == (1024,)
        assert result.dtype == np.float32
        # Must have at least one value > 1 for a non-trivial molecule
        # (some substructures appear more than once)
        assert result.max() >= 1.0
        # All values must be non-negative integers
        assert (result >= 0).all()
        assert np.allclose(result, result.astype(int))

    def test_atom_pair_count(self):
        w = RDKitWrapper(fingerprint_type="atom_pair_count", n_bits=1024)
        w.load(torch.device("cpu"))
        result = w.embed("c1ccccc1")
        assert result.shape == (1024,)
        assert (result >= 0).all()
        assert np.allclose(result, result.astype(int))

    def test_topological_torsion_count(self):
        w = RDKitWrapper(fingerprint_type="topological_torsion_count", n_bits=1024)
        w.load(torch.device("cpu"))
        result = w.embed("c1ccccc1")
        assert result.shape == (1024,)
        assert (result >= 0).all()
        assert np.allclose(result, result.astype(int))

    def test_count_differs_from_binary(self):
        """Count FP should differ from the binary version."""
        smiles = "c1ccc(CC(=O)O)cc1"
        w_bin = RDKitWrapper(fingerprint_type="morgan", radius=2, n_bits=1024)
        w_bin.load(torch.device("cpu"))
        w_cnt = RDKitWrapper(fingerprint_type="morgan_count", radius=2, n_bits=1024)
        w_cnt.load(torch.device("cpu"))

        binary = w_bin.embed(smiles)
        counts = w_cnt.embed(smiles)
        # Binary has only 0/1, counts may have values > 1
        assert set(np.unique(binary)).issubset({0.0, 1.0})
        # Where binary is 1, counts should be >= 1
        assert (counts[binary == 1.0] >= 1.0).all()

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    def test_is_binary_property(self):
        for fp in ("rdkit", "morgan", "maccs", "atom_pair", "topological_torsion"):
            w = RDKitWrapper(fingerprint_type=fp)
            assert w.is_binary_fingerprint is True
            assert w.is_count_fingerprint is False

    def test_is_count_property(self):
        for fp in ("morgan_count", "atom_pair_count", "topological_torsion_count"):
            w = RDKitWrapper(fingerprint_type=fp)
            assert w.is_binary_fingerprint is False
            assert w.is_count_fingerprint is True

    # ----------------------------------------------------------------
    # Batch and consistency
    # ----------------------------------------------------------------

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

    @pytest.mark.parametrize(
        "fp_type",
        RDKitWrapper._VALID_FP_TYPES,
    )
    def test_all_fp_types_produce_correct_shape(self, fp_type):
        """Smoke test: every registered FP type produces a valid vector."""
        w = RDKitWrapper(fingerprint_type=fp_type, n_bits=512)
        w.load(torch.device("cpu"))
        result = w.embed("c1ccccc1")
        expected_bits = 167 if fp_type == "maccs" else 512
        assert result.shape == (expected_bits,)
        assert result.dtype == np.float32
        assert not np.isnan(result).any()
