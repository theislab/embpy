"""Tests for protein model wrappers (ESM2, ESMC) using mocks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from embpy.models.protein_models import ESM2Wrapper


def _tok(seq_len: int = 12, has_mask: bool = True) -> MagicMock:
    """Mock tokenizer that honours ``return_tensors`` and its batch size.

    The protein wrappers use two different tokenizer contracts:

    * ESM2's ``embed_batch`` chunks long sequences itself, so it tokenizes one
      string at a time with ``return_tensors=None`` and expects ``list[int]``
      back -- it then adds special tokens and pads by hand. Handing it a tensor
      makes the ``if ids and ...`` coercion guard raise "Boolean value of Tensor
      with more than one value is ambiguous", which no real tokenizer does.
    * ProtT5's ``embed_batch`` tokenizes the whole batch in one call with
      ``padding="longest", return_tensors="pt"`` and then iterates the batch
      dimension of the model output. A mock returning a fixed ``(1, seq_len)``
      tensor silently yields ONE embedding for N inputs.

    So the mock has to vary both container type and batch size with its input,
    exactly as a real tokenizer does.
    """

    def _encode(text: Any = None, *args: Any, **kwargs: Any) -> dict:
        n = len(text) if isinstance(text, list | tuple) else 1
        if kwargs.get("return_tensors") == "pt":
            enc: dict = {"input_ids": torch.zeros(n, seq_len, dtype=torch.long)}
            if has_mask:
                enc["attention_mask"] = torch.ones(n, seq_len, dtype=torch.long)
            return enc
        ids = [1] * seq_len
        enc = {"input_ids": ids if n == 1 else [ids] * n}
        if has_mask:
            mask = [1] * seq_len
            enc["attention_mask"] = mask if n == 1 else [mask] * n
        return enc

    mock = MagicMock(side_effect=_encode)
    mock.cls_token_id = 2
    mock.sep_token_id = 3
    mock.bos_token_id = None
    mock.eos_token_id = None
    mock.pad_token_id = 0
    mock.model_max_length = 512
    return mock


def _dyn_model(
    hidden_dim: int,
    *,
    last_hidden: bool = True,
    layers: int = 2,
    layer_fills: tuple[float, ...] | None = None,
) -> MagicMock:
    """Model mock whose output is shaped from the ``input_ids`` it receives.

    A static ``return_value`` cannot track the batch size or the token count
    after special tokens are added, which shows up either as a tensor-size
    mismatch during masked pooling or as a silently short result list.
    """

    def _forward(*args: Any, **kwargs: Any) -> MagicMock:
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        n_batch, n_tok = (int(ids.shape[0]), int(ids.shape[1])) if ids is not None else (1, seq_default)
        if layer_fills is not None:
            stack = tuple(
                torch.full((n_batch, n_tok, hidden_dim), float(v)) for v in layer_fills
            )
        else:
            stack = tuple(torch.randn(n_batch, n_tok, hidden_dim) for _ in range(layers))
        out = MagicMock()
        out.hidden_states = stack
        out.last_hidden_state = stack[-1] if last_hidden else None
        return out

    seq_default = 12
    model = MagicMock(side_effect=_forward)
    model.config = MagicMock()
    model.config.num_hidden_layers = layers
    return model


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
        w.tokenizer = _tok()
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
        mock_output.hidden_states = tuple(torch.randn(1, seq_len, hidden_dim) for _ in range(n_layers))

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

        w.model = _dyn_model(hidden_dim)
        w.tokenizer = _tok(seq_len)

        results = w.embed_batch(["MTEYKLVVVG", "ACDEFGHIKL"])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (hidden_dim,)


class TestProtT5Wrapper:
    """Tests for ProtT5Wrapper (mocked T5EncoderModel + T5Tokenizer)."""

    def test_init_defaults(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        assert w.model_name == "Rostlab/prot_t5_xl_half_uniref50-enc"
        assert w.model_type == "protein"
        assert w.model is None
        assert w.tokenizer is None
        assert w.EMBEDDING_DIM == 1024

    def test_init_custom_name(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper(model_path_or_name="Rostlab/prot_t5_xl_uniref50")
        assert w.model_name == "Rostlab/prot_t5_xl_uniref50"

    def test_embed_without_load_raises(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("MTEYKLVVVG")

    def test_embed_batch_without_load_raises(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["MTEYKLVVVG"])

    def test_embed_batch_empty_returns_empty(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        w.model = MagicMock()
        w.tokenizer = _tok()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_load_empty_name_raises(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper(model_path_or_name="")
        with pytest.raises(ValueError, match="model_path_or_name"):
            w.load(torch.device("cpu"))

    def test_invalid_pooling_raises(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        w.tokenizer = MagicMock()
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("MTEYKLVVVG", pooling_strategy="invalid")

    def test_prepare_sequence(self):
        from embpy.models.protein_models import ProtT5Wrapper

        assert ProtT5Wrapper._prepare_sequence("MTEYK") == "M T E Y K"
        # Rare amino acids mapped to X
        assert ProtT5Wrapper._prepare_sequence("MBUZ") == "M X X X"
        # Whitespace stripped
        assert ProtT5Wrapper._prepare_sequence("  MTE  ") == "M T E"

    def test_prepare_sequence_lowercase(self):
        from embpy.models.protein_models import ProtT5Wrapper

        assert ProtT5Wrapper._prepare_sequence("mteyk") == "M T E Y K"

    def _make_loaded_wrapper(self, hidden_dim: int = 1024, seq_len: int = 12):
        """Create a ProtT5Wrapper with mocked model and tokenizer."""
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        w.device = torch.device("cpu")

        mock_model = _dyn_model(hidden_dim)
        w.model = mock_model
        w.tokenizer = _tok(seq_len)

        return w, mock_model

    def test_embed_mean_pooling(self):
        w, _ = self._make_loaded_wrapper()
        result = w.embed("MTEYKLVVVG", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert not np.isnan(result).any()

    def test_embed_max_pooling(self):
        w, _ = self._make_loaded_wrapper()
        result = w.embed("MTEYKLVVVG", pooling_strategy="max")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    def test_embed_cls_pooling(self):
        w, _ = self._make_loaded_wrapper()
        result = w.embed("MTEYKLVVVG", pooling_strategy="cls")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    @pytest.mark.parametrize("strategy", ["mean", "max", "cls"])
    def test_embed_all_pooling_strategies(self, strategy):
        w, _ = self._make_loaded_wrapper()
        result = w.embed("MTEYKLVVVG", pooling_strategy=strategy)
        assert result.shape == (1024,)

    def test_embed_with_target_layer(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        w.device = torch.device("cpu")

        hidden_dim = 1024
        seq_len = 12
        n_layers = 6

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, seq_len, hidden_dim)
        mock_output.hidden_states = tuple(torch.randn(1, seq_len, hidden_dim) for _ in range(n_layers))

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

    def test_embed_target_layer_invalid_raises(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        w.device = torch.device("cpu")

        n_layers = 6
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 1024)
        mock_output.hidden_states = tuple(torch.randn(1, 10, 1024) for _ in range(n_layers))

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        w.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        w.tokenizer = mock_tokenizer

        with pytest.raises(ValueError, match="Invalid target_layer"):
            w.embed("MTEYKLVVVG", target_layer=100)

    def test_embed_batch_returns_list(self):
        w, _ = self._make_loaded_wrapper()
        results = w.embed_batch(["MTEYKLVVVG", "ACDEFGHIKL"])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (1024,)

    def test_embedding_dim_constant(self):
        from embpy.models.protein_models import ProtT5Wrapper

        assert ProtT5Wrapper.EMBEDDING_DIM == 1024

    def test_available_pooling_strategies(self):
        from embpy.models.protein_models import ProtT5Wrapper

        w = ProtT5Wrapper()
        assert "mean" in w.available_pooling_strategies
        assert "max" in w.available_pooling_strategies
        assert "cls" in w.available_pooling_strategies


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


# =====================================================================
# ESM3Wrapper
# =====================================================================


class TestESM3Wrapper:
    def test_init_defaults(self):
        from embpy.models.protein_models import ESM3Wrapper

        w = ESM3Wrapper()
        assert w.model_name == "esm3-small-2024-08"
        assert w.model_type == "protein"
        assert w._client is None

    def test_init_custom_model(self):
        from embpy.models.protein_models import ESM3Wrapper

        w = ESM3Wrapper("esm3-large-2024-03", forge_token="test_token")
        assert w.model_name == "esm3-large-2024-03"
        assert w.forge_token == "test_token"

    def test_embed_without_load_raises(self):
        from embpy.models.protein_models import ESM3Wrapper

        w = ESM3Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("MTEYKLVVVG")

    def test_embed_with_mock(self):
        from embpy.models.protein_models import ESM3Wrapper

        w = ESM3Wrapper()
        mock_client = MagicMock()
        mock_client.encode.return_value = MagicMock()

        mock_logits_output = MagicMock()
        mock_logits_output.embeddings = torch.randn(1, 10, 64)
        mock_client.logits.return_value = mock_logits_output

        w._client = mock_client
        w.device = torch.device("cpu")
        w.model = mock_client

        emb = w.embed("MTEYKLVVVG", pooling_strategy="mean")
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1

    def test_embed_batch(self):
        from embpy.models.protein_models import ESM3Wrapper

        w = ESM3Wrapper()
        mock_client = MagicMock()
        mock_client.encode.return_value = MagicMock()

        mock_logits_output = MagicMock()
        mock_logits_output.embeddings = torch.randn(1, 10, 64)
        mock_client.logits.return_value = mock_logits_output

        w._client = mock_client
        w.device = torch.device("cpu")
        w.model = mock_client

        results = w.embed_batch(["MTEYKLVVVG", "ACDEFGHIK"])
        assert len(results) == 2

    def test_invalid_pooling_raises(self):
        from embpy.models.protein_models import ESM3Wrapper

        w = ESM3Wrapper()
        w._client = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("MTEYKLVVVG", pooling_strategy="invalid")
