"""Tests for TextLLMWrapper using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from embpy.models.text_models import TextLLMWrapper


class TestTextLLMWrapper:
    def test_init_defaults(self):
        w = TextLLMWrapper()
        assert w.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert w.model_type == "text"
        assert w.model is None
        assert w.tokenizer is None
        assert w.cls_token_position == 0

    def test_init_custom(self):
        w = TextLLMWrapper(model_path_or_name="bert-base-uncased", max_length=256)
        assert w.model_name == "bert-base-uncased"
        assert w.max_length == 256

    def test_embed_without_load_raises(self):
        w = TextLLMWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("Hello world")

    def test_embed_batch_without_load_raises(self):
        w = TextLLMWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["Hello", "World"])

    def test_embed_batch_empty_returns_empty(self):
        w = TextLLMWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = TextLLMWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("Hello", pooling_strategy="invalid")

    def test_load_with_none_name_raises(self):
        w = TextLLMWrapper(model_path_or_name=None)
        with pytest.raises(ValueError, match="model_path_or_name"):
            w.load(torch.device("cpu"))

    def test_preprocess_without_tokenizer_raises(self):
        w = TextLLMWrapper()
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            w._preprocess_text_hf("Hello")

    def test_embed_with_mocked_model(self):
        w = TextLLMWrapper()
        w.device = torch.device("cpu")
        w.max_length = 512

        hidden_dim = 384
        seq_len = 8

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.hidden_states = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.config = MagicMock()
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        mock_tokenizer.get = MagicMock(return_value=None)
        w.tokenizer = mock_tokenizer

        result = w.embed("Hello world", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    @pytest.mark.parametrize("strategy", ["mean", "max", "cls"])
    def test_embed_all_pooling_strategies(self, strategy):
        w = TextLLMWrapper()
        w.device = torch.device("cpu")
        w.max_length = 512
        w.cls_token_position = 0

        hidden_dim = 384
        seq_len = 8

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

        result = w.embed("Hello", pooling_strategy=strategy)
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)

    def test_embed_with_target_layer(self):
        w = TextLLMWrapper()
        w.device = torch.device("cpu")
        w.max_length = 512

        hidden_dim = 384
        seq_len = 8
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

        result = w.embed("Hello", target_layer=2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)

    def test_embed_batch_with_mocked_model(self):
        w = TextLLMWrapper()
        w.device = torch.device("cpu")
        w.max_length = 512
        w.cls_token_position = 0

        hidden_dim = 384
        seq_len = 8
        batch_size = 3

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        mock_output.hidden_states = None

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }
        mock_tokenizer.get = MagicMock(return_value=None)
        w.tokenizer = mock_tokenizer

        results = w.embed_batch(["Hello", "World", "Test"])
        assert len(results) == batch_size
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (hidden_dim,)

    def test_embed_batch_with_batch_size(self):
        w = TextLLMWrapper()
        w.device = torch.device("cpu")
        w.max_length = 512
        w.cls_token_position = 0

        hidden_dim = 384
        seq_len = 8

        def make_output(bs):
            out = MagicMock()
            out.last_hidden_state = torch.randn(bs, seq_len, hidden_dim)
            out.hidden_states = None
            return out

        mock_model = MagicMock()
        mock_model.side_effect = [make_output(2), make_output(1)]
        w.model = mock_model

        def tokenize_fn(texts, **kwargs):
            bs = len(texts)
            return {
                "input_ids": torch.zeros(bs, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(bs, seq_len, dtype=torch.long),
            }

        w.tokenizer = MagicMock(side_effect=tokenize_fn)

        results = w.embed_batch(["A", "B", "C"], batch_size=2)
        assert len(results) == 3

    def test_get_cls_embedding_2d(self):
        w = TextLLMWrapper()
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = w._get_cls_embedding(tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0]))

    def test_get_cls_embedding_3d(self):
        w = TextLLMWrapper()
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = w._get_cls_embedding(tensor)
        assert result.shape == (1, 2)

    def test_last_token_pool_right_padding(self):
        w = TextLLMWrapper()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 0]])
        result = w._last_token_pool(hidden, mask)
        assert result.shape == (1, 2)
        assert torch.equal(result, torch.tensor([[3.0, 4.0]]))
