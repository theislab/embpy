"""Tests for protein model wrappers (ESM2, ESMC) using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.models.protein_models import ESM2Wrapper


class TestESM2Wrapper:
    def test_init_defaults(self):
        w = ESM2Wrapper()
        assert w.model_name == "facebook/esm2_t6_8M_UR50D"
        assert w.model_type == "protein"
        assert w.model is None
        assert w.tokenizer is None

    def test_init_calls_super(self):
        w = ESM2Wrapper(model_path_or_name="facebook/esm2_t6_8M_UR50D")
        assert w.model_name == "facebook/esm2_t6_8M_UR50D"
        assert w.device is None

    def test_embed_without_load_raises(self):
        w = ESM2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("MTEYKLVVVG")

    def test_embed_batch_without_load_raises(self):
        w = ESM2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["MTEYKLVVVG"])

    def test_embed_batch_empty_returns_empty(self):
        w = ESM2Wrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = ESM2Wrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        w.tokenizer = MagicMock()
        w.tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("MTEYKLVVVG", pooling_strategy="invalid")

    def test_preprocess_without_tokenizer_raises(self):
        w = ESM2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w._preprocess_sequence("MTEYKLVVVG")

    def test_load_empty_name_raises(self):
        w = ESM2Wrapper(model_path_or_name="")
        with pytest.raises(ValueError, match="model_path_or_name"):
            w.load(torch.device("cpu"))

    def test_embed_with_mocked_model(self):
        w = ESM2Wrapper()
        w.device = torch.device("cpu")

        hidden_dim = 320
        seq_len = 12
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.hidden_states = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        w.tokenizer = mock_tokenizer

        result = w.embed("MTEYKLVVVG", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    @pytest.mark.parametrize("strategy", ["mean", "max", "cls"])
    def test_embed_pooling_strategies(self, strategy):
        w = ESM2Wrapper()
        w.device = torch.device("cpu")

        hidden_dim = 320
        seq_len = 12
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.hidden_states = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        w.tokenizer = mock_tokenizer

        result = w.embed("MTEYKLVVVG", pooling_strategy=strategy)
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)

    def test_embed_with_target_layer(self):
        w = ESM2Wrapper()
        w.device = torch.device("cpu")

        hidden_dim = 320
        seq_len = 12
        n_layers = 6

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.hidden_states = tuple(
            torch.randn(1, seq_len, hidden_dim) for _ in range(n_layers)
        )

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        w.tokenizer = mock_tokenizer

        result = w.embed("MTEYKLVVVG", target_layer=3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)

    def test_embed_batch_returns_list(self):
        w = ESM2Wrapper()
        w.device = torch.device("cpu")

        hidden_dim = 320
        seq_len = 12

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.hidden_states = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        w.tokenizer = mock_tokenizer

        results = w.embed_batch(["MTEYKLVVVG", "ACDEFGHIKL"])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (hidden_dim,)


class TestESMCWrapper:
    """Tests for ESMCWrapper (mocked ESMC SDK)."""

    def test_init(self):
        from embpy.models.protein_models import ESMCWrapper

        w = ESMCWrapper()
        assert w.model_name == "esmc_300m"
        assert w.model_type == "protein"
        assert w.client is None

    def test_embed_without_load_raises(self):
        from embpy.models.protein_models import ESMCWrapper

        w = ESMCWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("MTEYKLVVVG")

    def test_invalid_pooling_raises(self):
        from embpy.models.protein_models import ESMCWrapper

        w = ESMCWrapper()
        w.client = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("MTEYKLVVVG", pooling_strategy="invalid")
