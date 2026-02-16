"""Tests for DNA model wrappers (Enformer, Borzoi, Evo2) using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.models.dna_models import BorzoiWrapper, EnformerWrapper


class TestEnformerWrapper:
    def test_init_defaults(self):
        w = EnformerWrapper()
        assert w.model_name == "EleutherAI/enformer-official-rough"
        assert w.model_type == "dna"
        assert w.SEQUENCE_LENGTH == 196_608
        assert w.TRUNK_OUTPUT_DIM == 3072

    def test_init_custom_name(self):
        w = EnformerWrapper(model_path_or_name="custom/enformer")
        assert w.model_name == "custom/enformer"

    def test_embed_without_load_raises(self):
        w = EnformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = EnformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = EnformerWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_strategy_raises(self):
        w = EnformerWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_preprocess_pads_short_sequence(self):
        w = EnformerWrapper()
        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w._preprocess_sequence("ACGT")
            assert result.shape[1] == w.SEQUENCE_LENGTH

    def test_preprocess_truncates_long_sequence(self):
        w = EnformerWrapper()
        long_seq = "A" * (w.SEQUENCE_LENGTH + 100)
        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w._preprocess_sequence(long_seq)
            assert result.shape[1] == w.SEQUENCE_LENGTH

    def test_embed_with_mocked_model(self):
        w = EnformerWrapper()
        w.device = torch.device("cpu")

        num_bins = 896
        trunk_tensor = torch.randn(1, num_bins, 3072)
        w.model = MagicMock()
        w.model.return_value = (None, trunk_tensor)

        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w.embed("ACGT", pooling_strategy="mean")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3072,)
        assert not np.isnan(result).any()

    def test_embed_max_pooling(self):
        w = EnformerWrapper()
        w.device = torch.device("cpu")

        num_bins = 896
        trunk_tensor = torch.randn(1, num_bins, 3072)
        w.model = MagicMock()
        w.model.return_value = (None, trunk_tensor)

        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w.embed("ACGT", pooling_strategy="max")

        assert result.shape == (3072,)


class TestBorzoiWrapper:
    def test_init_defaults(self):
        w = BorzoiWrapper()
        assert w.model_name == "johahi/borzoi-replicate-0"
        assert w.model_type == "dna"
        assert w.SEQUENCE_LENGTH == 524_288
        assert w.NUM_CHANNELS == 4

    def test_embed_without_load_raises(self):
        w = BorzoiWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = BorzoiWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = BorzoiWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = BorzoiWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_preprocess_short_sequence(self):
        w = BorzoiWrapper()
        result = w._preprocess_sequence("ACGT")
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_preprocess_long_sequence(self):
        w = BorzoiWrapper()
        long_seq = "A" * (w.SEQUENCE_LENGTH + 100)
        result = w._preprocess_sequence(long_seq)
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_preprocess_exact_length(self):
        w = BorzoiWrapper()
        seq = "A" * w.SEQUENCE_LENGTH
        result = w._preprocess_sequence(seq)
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_embed_with_mocked_model(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")

        hidden_dim = 512
        num_bins = 100
        w.TRUNK_OUTPUT_DIM = hidden_dim
        embs_tensor = torch.randn(1, hidden_dim, num_bins)

        mock_model = MagicMock()
        mock_model.get_embs_after_crop.return_value = embs_tensor
        w.model = mock_model

        result = w.embed("ACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()


class TestEvo2Wrapper:
    """Tests for the Evo2Wrapper (mocked, since evo2 may not be installed)."""

    def test_import_available(self):
        from embpy.models.dna_models import Evo2Wrapper

        assert Evo2Wrapper is not None

    def test_init_defaults(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        assert w.model_name == "evo2_7b"
        assert w.model_type == "dna"
        assert w.layer_name is None

    def test_init_custom_layer(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper(layer_name="blocks.10.mlp.l3")
        assert w.layer_name == "blocks.10.mlp.l3"

    def test_layer_defaults_mapping(self):
        from embpy.models.dna_models import Evo2Wrapper

        assert "evo2_7b" in Evo2Wrapper.LAYER_DEFAULTS
        assert "evo2_40b" in Evo2Wrapper.LAYER_DEFAULTS
        assert "evo2_1b_base" in Evo2Wrapper.LAYER_DEFAULTS

    def test_embed_without_load_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w._evo2_model = MagicMock()
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w._evo2_model = MagicMock()
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_load_without_evo2_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        with patch("embpy.models.dna_models._HAVE_EVO2", False):
            with pytest.raises(ImportError, match="evo2 package is not installed"):
                w.load(torch.device("cpu"))

    def test_embed_with_mocked_evo2(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper(model_path_or_name="evo2_7b")
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10
        embedding_tensor = torch.randn(1, seq_len, hidden_dim)

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))
        mock_evo2.return_value = (None, {"blocks.28.mlp.l3": embedding_tensor})
        w._evo2_model = mock_evo2

        result = w.embed("ACGTACGTAC", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    def test_embed_cls_pooling(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10
        embedding_tensor = torch.randn(1, seq_len, hidden_dim)

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))
        mock_evo2.return_value = (None, {"blocks.28.mlp.l3": embedding_tensor})
        w._evo2_model = mock_evo2

        result = w.embed("ACGT", pooling_strategy="cls")
        assert result.shape == (hidden_dim,)

    def test_embed_max_pooling(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10
        embedding_tensor = torch.randn(1, seq_len, hidden_dim)

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))
        mock_evo2.return_value = (None, {"blocks.28.mlp.l3": embedding_tensor})
        w._evo2_model = mock_evo2

        result = w.embed("ACGT", pooling_strategy="max")
        assert result.shape == (hidden_dim,)

    def test_embed_missing_layer_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = [0, 1, 2, 3]
        mock_evo2.return_value = (None, {"some.other.layer": torch.randn(1, 4, 100)})
        w._evo2_model = mock_evo2

        with pytest.raises(ValueError, match="Layer.*not found"):
            w.embed("ACGT")

    def test_embed_batch_with_mocked_evo2(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))

        def mock_forward(ids, return_embeddings=False, layer_names=None):
            return (None, {"blocks.28.mlp.l3": torch.randn(1, seq_len, hidden_dim)})

        mock_evo2.side_effect = mock_forward
        w._evo2_model = mock_evo2

        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (hidden_dim,)
