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


class TestLlamaEmbeddingWrapperLoad:
    @pytest.mark.parametrize("device_str", ["cpu", "cuda:0"])
    def test_load_without_device_map_then_moves_to_device(self, device_str):
        """``device_map`` needs ``accelerate`` (in no extra): load plainly and ``.to(device)`` on every device."""
        from embpy.models.text_models import LlamaEmbeddingWrapper

        device = torch.device(device_str)
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2

        with (
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load,
            # load() imports transformers lazily, so patch the classes on transformers itself.
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            w = LlamaEmbeddingWrapper("test-llama")
            w.load(device)

        assert "device_map" not in mock_load.call_args.kwargs
        mock_model.to.assert_called_once_with(device)
        mock_model.eval.assert_called_once()
        assert w.model is mock_model
        assert w.device == device
        assert mock_tokenizer.pad_token == "</s>"


class TestLlamaEmbeddingWrapperEmbedBatch:
    """``embed_batch`` on hidden states only, as ``AutoModelForCausalLM`` outputs have no ``last_hidden_state``."""

    HIDDEN_DIM = 4
    TEXTS = ["a b c d e", "a b c", "a"]

    @pytest.fixture
    def hidden(self):
        torch.manual_seed(0)
        # three layers of (batch=3, seq_len=5, hidden_dim=4)
        return tuple(torch.randn(3, 5, self.HIDDEN_DIM) for _ in range(3))

    def _make(self, hidden, attention_mask):
        from types import SimpleNamespace

        from embpy.models.text_models import LlamaEmbeddingWrapper

        w = LlamaEmbeddingWrapper("test-llama")
        w.device = torch.device("cpu")
        w.tokenizer = MagicMock(
            return_value={"input_ids": torch.ones(3, 5, dtype=torch.long), "attention_mask": attention_mask}
        )
        # SimpleNamespace, not MagicMock: no ``last_hidden_state`` attribute, like CausalLMOutputWithPast.
        w.model = MagicMock(return_value=SimpleNamespace(hidden_states=hidden))
        return w

    @pytest.fixture
    def wrapper(self, hidden):
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        return self._make(hidden, mask)

    def test_returns_one_vector_per_input(self, wrapper):
        out = wrapper.embed_batch(self.TEXTS)
        assert isinstance(out, list)
        assert len(out) == 3
        assert all(isinstance(e, np.ndarray) and e.shape == (self.HIDDEN_DIM,) for e in out)

    def test_last_token_default_picks_last_real_token(self, wrapper, hidden):
        out = wrapper.embed_batch(self.TEXTS)
        for i, last in enumerate([4, 2, 0]):
            np.testing.assert_allclose(out[i], hidden[-1][i, last].numpy())

    def test_last_token_with_left_padding(self, hidden):
        mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]])
        out = self._make(hidden, mask).embed_batch(self.TEXTS)
        for i in range(3):
            np.testing.assert_allclose(out[i], hidden[-1][i, -1].numpy())

    def test_mean_ignores_padding(self, wrapper, hidden):
        out = wrapper.embed_batch(self.TEXTS, pooling_strategy="mean")
        for i, n in enumerate([5, 3, 1]):
            np.testing.assert_allclose(out[i], hidden[-1][i, :n].mean(dim=0).numpy(), rtol=1e-5, atol=1e-6)

    def test_max_ignores_padding(self, wrapper, hidden):
        out = wrapper.embed_batch(self.TEXTS, pooling_strategy="max")
        for i, n in enumerate([5, 3, 1]):
            np.testing.assert_allclose(out[i], hidden[-1][i, :n].max(dim=0).values.numpy())

    def test_none_returns_unpadded_token_matrix(self, wrapper, hidden):
        out = wrapper.embed_batch(self.TEXTS, pooling_strategy="none")
        for i, n in enumerate([5, 3, 1]):
            assert out[i].shape == (n, self.HIDDEN_DIM)
            np.testing.assert_allclose(out[i], hidden[-1][i, :n].numpy())

    def test_target_layer(self, wrapper, hidden):
        out = wrapper.embed_batch(self.TEXTS, pooling_strategy="mean", target_layer=1)
        np.testing.assert_allclose(out[0], hidden[1][0].mean(dim=0).numpy(), rtol=1e-5, atol=1e-6)
        assert wrapper.model.call_args.kwargs["output_hidden_states"] is True

    def test_tokenizes_as_padded_truncated_batch(self, wrapper):
        wrapper.max_length = 7
        wrapper.embed_batch(self.TEXTS)
        assert wrapper.tokenizer.call_args.args[0] == self.TEXTS
        assert wrapper.tokenizer.call_args.kwargs["padding"] is True
        assert wrapper.tokenizer.call_args.kwargs["truncation"] is True
        assert wrapper.tokenizer.call_args.kwargs["max_length"] == 7

    def test_empty_inputs(self, wrapper):
        assert wrapper.embed_batch([]) == []

    def test_invalid_pooling_raises(self, wrapper):
        with pytest.raises(ValueError, match="Invalid pooling"):
            wrapper.embed_batch(["a"], pooling_strategy="cls")

    def test_not_loaded_raises(self):
        from embpy.models.text_models import LlamaEmbeddingWrapper

        with pytest.raises(RuntimeError, match="not loaded"):
            LlamaEmbeddingWrapper("test-llama").embed_batch(["a"])


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    """A tiny random-weight LlamaForCausalLM + tokenizer (no pad token, like real Llama), built offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny_llama")
    corpus = [
        "tumour suppressor",
        "kinase",
        "tumor suppressor protein p53 regulates the cell cycle",
        "serine threonine kinase phosphorylates substrates",
        "a gene encoding a transcription factor",
    ]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


TINY_TEXTS = [
    "kinase",
    "tumor suppressor protein p53 regulates the cell cycle",
    "a gene",
    "serine threonine kinase phosphorylates substrates",
]


class TestLlamaEmbeddingWrapperTinyModel:
    """Real ``LlamaForCausalLM`` (tiny, random weights, local files): embed() and embed_batch() must agree."""

    @pytest.fixture
    def wrapper(self, tiny_llama_dir):
        from embpy.models.text_models import LlamaEmbeddingWrapper

        w = LlamaEmbeddingWrapper(tiny_llama_dir)
        w.load(torch.device("cpu"))
        w.model.float()  # bf16 gives batch-shape-dependent rounding; compare tightly in fp32
        return w

    def test_load_sets_pad_token(self, wrapper):
        assert wrapper.tokenizer.pad_token == wrapper.tokenizer.eos_token

    @pytest.mark.parametrize("padding_side", ["right", "left"])
    @pytest.mark.parametrize("pooling", ["last_token", "mean", "max", "none"])
    @pytest.mark.parametrize("target_layer", [None, 1])
    def test_embed_batch_matches_embed(self, wrapper, pooling, padding_side, target_layer):
        wrapper.tokenizer.padding_side = padding_side
        batch = wrapper.embed_batch(TINY_TEXTS, pooling_strategy=pooling, target_layer=target_layer)
        assert len(batch) == len(TINY_TEXTS)
        for text, emb in zip(TINY_TEXTS, batch, strict=True):
            single = wrapper.embed(text, pooling_strategy=pooling, target_layer=target_layer)
            assert emb.shape == single.shape
            np.testing.assert_allclose(emb, single, rtol=1e-4, atol=1e-5)

    def test_embed_batch_distinguishes_inputs(self, wrapper):
        out = np.stack(wrapper.embed_batch(TINY_TEXTS))
        assert out.shape == (len(TINY_TEXTS), 32)
        assert len({tuple(np.round(row, 5)) for row in out}) == len(TINY_TEXTS)

    def test_embed_batch_chunks_with_batch_size(self, wrapper):
        full = wrapper.embed_batch(TINY_TEXTS, pooling_strategy="mean")
        with patch.object(wrapper, "_process_batch_chunk", wraps=wrapper._process_batch_chunk) as spy:
            chunked = wrapper.embed_batch(TINY_TEXTS, pooling_strategy="mean", batch_size=3)
        assert spy.call_count == 2
        for a, b in zip(full, chunked, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_bioembedder_embed_text_end_to_end(self, tiny_llama_dir):
        """``BioEmbedder.embed`` routes through ``embed_batch``; this used to raise for Llama."""
        from embpy.embedder import BioEmbedder
        from embpy.models.text_models import LlamaEmbeddingWrapper

        texts = ["tumour suppressor", "kinase"]
        with (
            patch("embpy.embedder.GeneResolver"),
            patch("embpy.embedder.ProteinResolver"),
            patch("embpy.embedder.TextResolver"),
            patch.object(
                BioEmbedder, "_discover_models", return_value={"tiny_llama": (LlamaEmbeddingWrapper, tiny_llama_dir)}
            ),
        ):
            embedder = BioEmbedder(device="cpu")
            payload = embedder.embed(
                texts,
                entity_type="text",
                model="tiny_llama",
                pooling_strategy="last_token",
                output="payload",
            )

        matrix = np.asarray(payload["matrix"])
        assert matrix.shape == (2, 32)
        assert np.isfinite(matrix).all()
        # same vectors as embedding each text on its own (model is bf16, so rounding differs with batch shape)
        wrapper = embedder.model_cache["tiny_llama"]
        for row, text in zip(matrix, texts, strict=True):
            np.testing.assert_allclose(row, wrapper.embed(text, pooling_strategy="last_token"), rtol=5e-2, atol=5e-2)
