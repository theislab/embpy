"""Tests for embpy.models.api_models.APIEmbeddingWrapper and
embpy.models.text_models.LlamaEmbeddingWrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# =====================================================================
# APIEmbeddingWrapper
# =====================================================================


class TestAPIEmbeddingWrapperInit:
    def test_default_provider(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("text-embedding-3-small")
        assert w.provider == "openai"
        assert w.model_name == "text-embedding-3-small"

    def test_custom_provider(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("embed-english-v3.0", provider="cohere")
        assert w.provider == "cohere"

    def test_model_type_is_text(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("test-model")
        assert w.model_type == "text"


class TestAPIEmbeddingWrapperLoad:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    def test_load_resolves_env_key(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("text-embedding-3-small", provider="openai")
        w.load(torch.device("cpu"))
        assert w._resolved_key == "test-key-123"
        assert "api.openai.com" in w._resolved_url

    def test_load_raises_without_key(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("test", provider="openai")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                w.load(torch.device("cpu"))

    @patch.dict("os.environ", {"LLM_API_KEY": "fallback-key"})
    def test_load_fallback_key(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("test", provider="generic", base_url="http://localhost:8000/v1")
        w.load(torch.device("cpu"))
        assert w._resolved_key == "fallback-key"

    def test_custom_base_url(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("test", provider="openai", api_key="k", base_url="http://local:11434/v1")
        w.load(torch.device("cpu"))
        assert w._resolved_url == "http://local:11434/v1"


class TestAPIEmbeddingWrapperEmbed:
    @pytest.fixture
    def openai_wrapper(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("text-embedding-3-small", provider="openai", api_key="test-key")
        w.load(torch.device("cpu"))
        return w

    @patch("embpy.models.api_models.requests.post")
    def test_embed_openai(self, mock_post, openai_wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
        }
        mock_post.return_value = mock_resp

        emb = openai_wrapper.embed("test text")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (3,)
        np.testing.assert_allclose(emb, [0.1, 0.2, 0.3], atol=1e-6)

    @patch("embpy.models.api_models.requests.post")
    def test_embed_batch_openai(self, mock_post, openai_wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ],
        }
        mock_post.return_value = mock_resp

        results = openai_wrapper.embed_batch(["text 1", "text 2"])
        assert len(results) == 2
        assert results[0].shape == (2,)

    def test_embed_raises_before_load(self):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("test", api_key="k")
        with pytest.raises(RuntimeError, match="not initialized"):
            w.embed("test")


class TestAPIEmbeddingWrapperCohere:
    @patch("embpy.models.api_models.requests.post")
    def test_cohere_embed(self, mock_post):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("embed-english-v3.0", provider="cohere", api_key="test")
        w.load(torch.device("cpu"))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "embeddings": {"float": [[0.5, 0.6, 0.7]]},
        }
        mock_post.return_value = mock_resp

        emb = w.embed("gene description")
        assert emb.shape == (3,)


class TestAPIEmbeddingWrapperGoogle:
    @patch("embpy.models.api_models.requests.post")
    def test_google_embed(self, mock_post):
        from embpy.models.api_models import APIEmbeddingWrapper

        w = APIEmbeddingWrapper("text-embedding-005", provider="google", api_key="test")
        w.load(torch.device("cpu"))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "embedding": {"values": [0.1, 0.2, 0.3, 0.4]},
        }
        mock_post.return_value = mock_resp

        emb = w.embed("gene description")
        assert emb.shape == (4,)


# =====================================================================
# LlamaEmbeddingWrapper
# =====================================================================


class TestLlamaEmbeddingWrapperInit:
    def test_default_params(self):
        from embpy.models.text_models import LlamaEmbeddingWrapper

        w = LlamaEmbeddingWrapper()
        assert w.model_name == "meta-llama/Llama-3.2-3B"
        assert w.max_length == 4096
        assert w.model_type == "text"

    def test_pooling_strategies(self):
        from embpy.models.text_models import LlamaEmbeddingWrapper

        w = LlamaEmbeddingWrapper()
        assert "last_token" in w.available_pooling_strategies
        assert "mean" in w.available_pooling_strategies


class TestLlamaEmbeddingWrapperEmbed:
    @pytest.fixture
    def mock_llama(self):
        from embpy.models.text_models import LlamaEmbeddingWrapper

        w = LlamaEmbeddingWrapper("test-llama")
        w.device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        w.tokenizer = mock_tokenizer

        hidden_dim = 2048
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.hidden_states = (
            torch.randn(1, 10, hidden_dim),
            torch.randn(1, 10, hidden_dim),
        )
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock(return_value=mock_model)
        w.model = mock_model

        return w

    def test_last_token_pooling(self, mock_llama):
        emb = mock_llama.embed("test text", pooling_strategy="last_token")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (2048,)

    def test_mean_pooling(self, mock_llama):
        emb = mock_llama.embed("test text", pooling_strategy="mean")
        assert emb.shape == (2048,)

    def test_max_pooling(self, mock_llama):
        emb = mock_llama.embed("test text", pooling_strategy="max")
        assert emb.shape == (2048,)

    def test_invalid_pooling_raises(self, mock_llama):
        with pytest.raises(ValueError, match="Invalid pooling"):
            mock_llama.embed("test text", pooling_strategy="cls")
