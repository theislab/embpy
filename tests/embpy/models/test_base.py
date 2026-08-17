"""Tests for BaseModelWrapper: pooling, layer introspection, and hidden-state extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from embpy.models.base import BaseModelWrapper

# =====================================================================
# Concrete test subclass
# =====================================================================


class ConcreteWrapper(BaseModelWrapper):
    """Minimal concrete subclass for testing the base class."""

    model_type = "unknown"
    available_pooling_strategies = ["mean", "max", "cls", "median"]

    def load(self, device):
        self.device = device

    def embed(self, input, pooling_strategy="mean", **kwargs):
        return np.zeros(4)

    def embed_batch(self, inputs, pooling_strategy="mean", **kwargs):
        return [np.zeros(4) for _ in inputs]


# =====================================================================
# Pooling tests
# =====================================================================


class TestApplyPooling:
    @pytest.fixture
    def wrapper(self):
        return ConcreteWrapper()

    def test_mean_pooling_2d(self, wrapper):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapper._apply_pooling(tensor, "mean")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-6)

    def test_max_pooling_2d(self, wrapper):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapper._apply_pooling(tensor, "max")
        np.testing.assert_allclose(result, [5.0, 6.0], atol=1e-6)

    def test_cls_pooling_2d(self, wrapper):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapper._apply_pooling(tensor, "cls")
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-6)

    def test_mean_pooling_3d(self, wrapper):
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = wrapper._apply_pooling(tensor, "mean")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [[2.0, 3.0]], atol=1e-6)

    def test_max_pooling_3d(self, wrapper):
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = wrapper._apply_pooling(tensor, "max")
        np.testing.assert_allclose(result, [[3.0, 4.0]], atol=1e-6)

    def test_cls_pooling_3d(self, wrapper):
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = wrapper._apply_pooling(tensor, "cls")
        np.testing.assert_allclose(result, [[1.0, 2.0]], atol=1e-6)

    def test_median_pooling_2d(self, wrapper):
        """Median over the token axis, not the first token (see base.py)."""
        tensor = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        result = wrapper._apply_pooling(tensor, "median")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, np.median(tensor.numpy(), axis=0), atol=1e-6)
        # Guard against the old bug, which returned the first token.
        assert not np.allclose(result, tensor.numpy()[0])

    def test_median_pooling_3d(self, wrapper):
        """Batched median pools per item over the token axis."""
        tensor = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        result = wrapper._apply_pooling(tensor, "median")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, np.median(tensor.numpy(), axis=1), atol=1e-6)

    def test_median_pooling_3d_shape_matches_mean_and_max(self, wrapper):
        """Median must reduce (batch, seq_len, hidden) -> (batch, hidden) like mean/max."""
        batch, seq_len, hidden = 2, 5, 3
        tensor = torch.arange(batch * seq_len * hidden, dtype=torch.float32).reshape(
            batch, seq_len, hidden
        )
        median = wrapper._apply_pooling(tensor, "median")
        assert median.shape == (batch, hidden)
        assert median.shape == wrapper._apply_pooling(tensor, "mean").shape
        assert median.shape == wrapper._apply_pooling(tensor, "max").shape

    def test_median_pooling_even_length_uses_lower_middle(self, wrapper):
        """torch.median takes the lower middle value; np.median averages the two.

        The base wrapper deliberately follows torch semantics, matching the
        median pooling already implemented in the Enformer/Borzoi wrappers.
        This test pins that choice so a future switch is a conscious one.
        """
        tensor = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        result = wrapper._apply_pooling(tensor, "median")
        np.testing.assert_allclose(result, [2.0, 20.0], atol=1e-6)
        # np.median would give [2.5, 25.0] here.
        assert not np.allclose(result, np.median(tensor.numpy(), axis=0))

    def test_invalid_strategy_raises(self, wrapper):
        tensor = torch.tensor([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            wrapper._apply_pooling(tensor, "nonexistent")

    def test_1d_tensor_raises(self, wrapper):
        tensor = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="Unsupported embedding tensor dimension"):
            wrapper._apply_pooling(tensor, "mean")


class TestBaseModelWrapperInit:
    def test_init_stores_model_name(self):
        w = ConcreteWrapper(model_path_or_name="test_model")
        assert w.model_name == "test_model"

    def test_init_defaults(self):
        w = ConcreteWrapper()
        assert w.model is None
        assert w.device is None

    def test_init_kwargs(self):
        w = ConcreteWrapper(model_path_or_name="test", foo="bar")
        assert w.config == {"foo": "bar"}


# =====================================================================
# Layer introspection tests
# =====================================================================


class TestGetNumLayers:
    def test_not_loaded_raises(self):
        w = ConcreteWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.get_num_layers()

    def test_hf_config_num_hidden_layers(self):
        w = ConcreteWrapper()
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = 12
        w.model = mock_model
        assert w.get_num_layers() == 12

    def test_hf_config_n_layer(self):
        w = ConcreteWrapper()
        mock_model = MagicMock(spec=torch.nn.Module)
        cfg = MagicMock()
        cfg.num_hidden_layers = None
        cfg.n_layer = 24
        del cfg.num_layers
        del cfg.n_layers
        mock_model.config = cfg
        # Remove ModuleList-like containers
        mock_model.blocks = None
        mock_model.layers = None
        w.model = mock_model
        assert w.get_num_layers() == 24

    def test_blocks_module_list(self):
        w = ConcreteWrapper()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(8)])

        w.model = FakeModel()
        assert w.get_num_layers() == 8

    def test_encoder_layer_module_list(self):
        w = ConcreteWrapper()

        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(6)])

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = FakeEncoder()

        w.model = FakeModel()
        assert w.get_num_layers() == 6

    def test_unknown_architecture_raises(self):
        w = ConcreteWrapper()

        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

        w.model = MinimalModel()
        with pytest.raises(NotImplementedError, match="Cannot auto-detect"):
            w.get_num_layers()


class TestGetLayerModules:
    def test_not_loaded_raises(self):
        w = ConcreteWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w._get_layer_modules()

    def test_blocks_found(self):
        w = ConcreteWrapper()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])

        w.model = FakeModel()
        modules = w._get_layer_modules()
        assert isinstance(modules, torch.nn.ModuleList)
        assert len(modules) == 3

    def test_encoder_layer_found(self):
        w = ConcreteWrapper()

        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = FakeEncoder()

        w.model = FakeModel()
        modules = w._get_layer_modules()
        assert len(modules) == 4

    def test_unknown_structure_raises(self):
        w = ConcreteWrapper()

        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

        w.model = MinimalModel()
        with pytest.raises(NotImplementedError, match="Cannot auto-detect"):
            w._get_layer_modules()


class TestIsHuggingfaceModel:
    def test_no_model_returns_false(self):
        w = ConcreteWrapper()
        assert w._is_huggingface_model() is False

    def test_hf_model_returns_true(self):
        w = ConcreteWrapper()
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = 12
        w.model = mock_model
        assert w._is_huggingface_model() is True

    def test_non_hf_model_returns_false(self):
        w = ConcreteWrapper()

        class PlainModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

        w.model = PlainModel()
        assert w._is_huggingface_model() is False


# =====================================================================
# Hidden-state extraction tests (HF path)
# =====================================================================


class TestExtractHiddenStatesHF:
    """Test HuggingFace-style hidden-state extraction via output_hidden_states."""

    @pytest.fixture
    def hf_wrapper(self):
        """Wrapper with a mock HuggingFace model that returns hidden states."""
        w = ConcreteWrapper()
        batch, seq_len, hidden = 1, 5, 16
        n_layers = 6

        hidden_states = tuple(torch.randn(batch, seq_len, hidden) for _ in range(n_layers + 1))

        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = n_layers
        mock_output = MagicMock()
        mock_output.hidden_states = hidden_states
        mock_model.return_value = mock_output
        w.model = mock_model
        w.device = torch.device("cpu")
        return w, hidden_states

    def test_extract_all_layers(self, hf_wrapper):
        w, hs = hf_wrapper
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        result = w.extract_hidden_states(input_ids, layers=None)
        assert len(result) == 7  # embedding + 6 layers
        for _idx, tensor in result.items():
            assert tensor.shape == (1, 5, 16)

    def test_extract_specific_layers(self, hf_wrapper):
        w, hs = hf_wrapper
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        result = w.extract_hidden_states(input_ids, layers=[0, 3, 6])
        assert set(result.keys()) == {0, 3, 6}

    def test_extract_negative_index(self, hf_wrapper):
        w, hs = hf_wrapper
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        result = w.extract_hidden_states(input_ids, layers=[-1])
        assert 6 in result  # last layer (index 6 for 7 total states)
        torch.testing.assert_close(result[6], hs[-1])

    def test_extract_out_of_range_raises(self, hf_wrapper):
        w, _ = hf_wrapper
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        with pytest.raises(IndexError, match="out of range"):
            w.extract_hidden_states(input_ids, layers=[99])

    def test_extract_negative_out_of_range_raises(self, hf_wrapper):
        w, _ = hf_wrapper
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        with pytest.raises(IndexError, match="out of range"):
            w.extract_hidden_states(input_ids, layers=[-100])

    def test_extract_with_attention_mask(self, hf_wrapper):
        w, _ = hf_wrapper
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        mask = torch.ones(1, 5, dtype=torch.long)
        result = w.extract_hidden_states(input_ids, attention_mask=mask, layers=[0])
        assert 0 in result
        # Verify attention_mask was passed through
        call_kwargs = w.model.call_args.kwargs
        assert call_kwargs["attention_mask"] is mask

    def test_not_loaded_raises(self):
        w = ConcreteWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.extract_hidden_states(torch.zeros(1, 5, dtype=torch.long))


# =====================================================================
# Hidden-state extraction tests (hook path)
# =====================================================================


class TestExtractHiddenStatesHook:
    """Test hook-based hidden-state extraction for non-HF models."""

    @pytest.fixture
    def hook_wrapper(self):
        """Wrapper with a real small PyTorch model (blocks-based)."""
        hidden = 8
        num_blocks = 4

        class SmallBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(hidden, hidden)

            def forward(self, x):
                return self.linear(x)

        class BlockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([SmallBlock() for _ in range(num_blocks)])

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        w = ConcreteWrapper()
        w.model = BlockModel()
        w.model.eval()
        w.device = torch.device("cpu")
        return w

    def test_extract_all_layers_hook(self, hook_wrapper):
        w = hook_wrapper
        x = torch.randn(1, 3, 8)
        result = w.extract_hidden_states(x, layers=None)
        assert len(result) == 4
        for idx in range(4):
            assert idx in result
            assert result[idx].shape == (1, 3, 8)

    def test_extract_specific_layers_hook(self, hook_wrapper):
        w = hook_wrapper
        x = torch.randn(1, 3, 8)
        result = w.extract_hidden_states(x, layers=[0, 3])
        assert set(result.keys()) == {0, 3}

    def test_extract_negative_index_hook(self, hook_wrapper):
        w = hook_wrapper
        x = torch.randn(1, 3, 8)
        result = w.extract_hidden_states(x, layers=[-1])
        assert 3 in result  # last block

    def test_extract_out_of_range_hook(self, hook_wrapper):
        w = hook_wrapper
        x = torch.randn(1, 3, 8)
        with pytest.raises(IndexError, match="out of range"):
            w.extract_hidden_states(x, layers=[10])

    def test_hook_captures_correct_layer(self, hook_wrapper):
        w = hook_wrapper
        x = torch.randn(1, 3, 8)
        # Extract from layers 0 and 3 — they should differ since the model transforms x
        result = w.extract_hidden_states(x, layers=[0, 3])
        assert not torch.equal(result[0], result[3])

    def test_hooks_are_removed(self, hook_wrapper):
        """Verify that forward hooks don't accumulate across calls."""
        w = hook_wrapper
        x = torch.randn(1, 3, 8)
        # Count hooks before
        hooks_before = sum(len(block._forward_hooks) for block in w.model.blocks)
        w.extract_hidden_states(x, layers=[0, 2])
        hooks_after = sum(len(block._forward_hooks) for block in w.model.blocks)
        assert hooks_after == hooks_before

    def test_tuple_output_captured(self):
        """Blocks that return (tensor, cache) tuples should capture the tensor part."""
        hidden = 8

        class TupleBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(hidden, hidden)

            def forward(self, x):
                return self.linear(x), None  # (output, cache)

        class TupleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([TupleBlock() for _ in range(3)])

            def forward(self, x):
                for block in self.blocks:
                    x, _ = block(x)
                return x

        w = ConcreteWrapper()
        w.model = TupleModel()
        w.model.eval()
        w.device = torch.device("cpu")

        x = torch.randn(1, 3, hidden)
        result = w.extract_hidden_states(x, layers=[0, 2])
        for idx in (0, 2):
            assert isinstance(result[idx], torch.Tensor)
            assert result[idx].shape == (1, 3, hidden)


# =====================================================================
# embed_all_layers tests
# =====================================================================


class TestEmbedAllLayers:
    def test_embed_all_layers_hf(self):
        w = ConcreteWrapper()
        batch, seq_len, hidden = 1, 5, 8
        n_layers = 3
        hidden_states = tuple(torch.randn(batch, seq_len, hidden) for _ in range(n_layers + 1))

        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = n_layers
        mock_output = MagicMock()
        mock_output.hidden_states = hidden_states
        mock_model.return_value = mock_output
        w.model = mock_model
        w.device = torch.device("cpu")

        input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        result = w.embed_all_layers(input_ids, pooling_strategy="mean")
        assert len(result) == n_layers + 1
        for _idx, emb in result.items():
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (batch, hidden)

    def test_embed_all_layers_invalid_pooling_raises(self):
        w = ConcreteWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            w.embed_all_layers(torch.zeros(1, 5, dtype=torch.long), pooling_strategy="bad")


# =====================================================================
# embed_from_layer tests
# =====================================================================


class TestEmbedFromLayer:
    def test_embed_from_layer_delegates_to_embed(self):
        w = ConcreteWrapper()
        result = w.embed_from_layer("hello", layer=3, pooling_strategy="mean")
        assert isinstance(result, np.ndarray)


# =====================================================================
# extract_attention tests
# =====================================================================


class AttentionFreeWrapper(ConcreteWrapper):
    """Stand-in for state-space / convolution / GNN architectures."""

    has_attention = False


class TestExtractAttention:
    N_LAYERS = 6
    N_HEADS = 4
    SEQ = 5

    @pytest.fixture
    def hf_wrapper(self):
        """Wrapper whose mock HF model returns one attention tensor per layer."""
        w = ConcreteWrapper()
        attns = tuple(torch.rand(1, self.N_HEADS, self.SEQ, self.SEQ) for _ in range(self.N_LAYERS))

        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = self.N_LAYERS
        mock_output = MagicMock()
        mock_output.attentions = attns
        mock_model.return_value = mock_output
        w.model = mock_model
        w.device = torch.device("cpu")
        return w, attns

    @property
    def ids(self):
        return torch.zeros(1, self.SEQ, dtype=torch.long)

    # -- guard rails ---------------------------------------------------

    def test_not_loaded_raises(self):
        w = ConcreteWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.extract_attention(self.ids)

    def test_attention_free_architecture_raises(self):
        w = AttentionFreeWrapper()
        w.model = MagicMock()
        w.model.config.num_hidden_layers = 4
        w.device = torch.device("cpu")
        with pytest.raises(NotImplementedError, match="attention-free"):
            w.extract_attention(self.ids)

    def test_non_huggingface_model_falls_back_to_hooks_then_raises(self):
        """Non-HF models now try the hook path first (see _extract_attention_hook).

        This model exposes no layer modules, so the hook path finds nothing and the
        error explains why rather than claiming HuggingFace is required.
        """

        class _Opaque:  # no .config -> not a HF model, and no layer container
            pass

        w = ConcreteWrapper()
        w.model = _Opaque()
        w.device = torch.device("cpu")
        with pytest.raises(NotImplementedError):
            w.extract_attention(self.ids)

    def test_fused_kernel_none_attentions_raises(self, hf_wrapper):
        """SDPA / FlashAttention return no weights -> loud error, not silence."""
        w, _ = hf_wrapper
        w.model.return_value.attentions = None
        with pytest.raises(RuntimeError, match="no attention weights"):
            w.extract_attention(self.ids)

    def test_partially_none_attentions_raises(self, hf_wrapper):
        w, attns = hf_wrapper
        w.model.return_value.attentions = (attns[0], None, *attns[2:])
        with pytest.raises(RuntimeError, match="no attention weights"):
            w.extract_attention(self.ids)

    # -- behaviour -----------------------------------------------------

    def test_extract_all_layers(self, hf_wrapper):
        w, _ = hf_wrapper
        result = w.extract_attention(self.ids, layers=None)
        assert set(result.keys()) == set(range(self.N_LAYERS))

    def test_no_embedding_layer_offset(self, hf_wrapper):
        """Unlike hidden states, attentions have NO embedding-layer entry."""
        w, _ = hf_wrapper
        result = w.extract_attention(self.ids, layers=None)
        assert len(result) == self.N_LAYERS  # not N_LAYERS + 1

    def test_attention_shape_is_batch_heads_seq_seq(self, hf_wrapper):
        w, _ = hf_wrapper
        result = w.extract_attention(self.ids, layers=[0])
        assert result[0].shape == (1, self.N_HEADS, self.SEQ, self.SEQ)

    def test_extract_specific_layers(self, hf_wrapper):
        w, _ = hf_wrapper
        result = w.extract_attention(self.ids, layers=[0, 2, 5])
        assert set(result.keys()) == {0, 2, 5}

    def test_negative_index_maps_to_last_layer(self, hf_wrapper):
        w, attns = hf_wrapper
        result = w.extract_attention(self.ids, layers=[-1])
        assert set(result.keys()) == {self.N_LAYERS - 1}
        torch.testing.assert_close(result[self.N_LAYERS - 1], attns[-1])

    def test_layer_index_maps_to_that_transformer_block(self, hf_wrapper):
        w, attns = hf_wrapper
        result = w.extract_attention(self.ids, layers=[3])
        torch.testing.assert_close(result[3], attns[3])

    def test_out_of_range_raises(self, hf_wrapper):
        w, _ = hf_wrapper
        with pytest.raises(IndexError, match="out of range"):
            w.extract_attention(self.ids, layers=[99])

    def test_negative_out_of_range_raises(self, hf_wrapper):
        w, _ = hf_wrapper
        with pytest.raises(IndexError, match="out of range"):
            w.extract_attention(self.ids, layers=[-99])

    def test_attention_mask_is_forwarded(self, hf_wrapper):
        w, _ = hf_wrapper
        mask = torch.ones(1, self.SEQ, dtype=torch.long)
        w.extract_attention(self.ids, attention_mask=mask)
        kwargs = w.model.call_args.kwargs
        assert kwargs["output_attentions"] is True
        torch.testing.assert_close(kwargs["attention_mask"], mask)


class TestHasAttentionFlags:
    """Attention-free architectures must be declared, not discovered at runtime."""

    def test_default_is_true(self):
        assert BaseModelWrapper.has_attention is True

    @pytest.mark.parametrize(
        ("module", "cls_name"),
        [
            ("embpy.models.dna_models", "HyenaDNAWrapper"),  # implicit long convolution
            ("embpy.models.dna_models", "CaduceusWrapper"),  # Mamba / SSM
            ("embpy.models.molecule_models", "MiniMolWrapper"),  # message-passing GNN
            ("embpy.models.molecule_models", "MHGGNNWrapper"),  # GIN graph autoencoder
        ],
    )
    def test_attention_free_wrappers_declare_false(self, module, cls_name):
        import importlib

        cls = getattr(importlib.import_module(module), cls_name)
        assert cls.has_attention is False, f"{cls_name} must declare has_attention = False"

    @pytest.mark.parametrize(
        ("module", "cls_name"),
        [
            ("embpy.models.protein_models", "ESM2Wrapper"),
            ("embpy.models.molecule_models", "ChembertaWrapper"),
            ("embpy.models.dna_models", "NucleotideTransformerWrapper"),
        ],
    )
    def test_transformer_wrappers_keep_attention(self, module, cls_name):
        import importlib

        cls = getattr(importlib.import_module(module), cls_name)
        assert cls.has_attention is True


# =====================================================================
# hook-based attention extraction (non-HuggingFace models)
# =====================================================================


class TestExtractAttentionHook:
    """Non-HF fallback: hooks can see materialised attention, never fused kernels."""

    SEQ = 6
    HEADS = 2

    def _wrapper_with(self, block_cls, n_layers=3):
        """A non-HF model (no .config) whose layers are `block_cls`."""

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([block_cls() for _ in range(n_layers)])

            def forward(self, x):
                for layer in self.layers:
                    out = layer(x)
                    # blocks may return (hidden, attn); chain on the hidden state
                    x = out[0] if isinstance(out, tuple) else out
                return x

        w = ConcreteWrapper()
        w.model = TinyModel().eval()
        w.device = torch.device("cpu")
        return w

    def test_captures_attention_from_explicit_softmax_block(self):
        """TranscriptFormer-style: attention materialised by an explicit softmax."""
        seq, heads = self.SEQ, self.HEADS

        class ExplicitAttnBlock(torch.nn.Module):
            def forward(self, x):
                scores = torch.rand(x.shape[0], heads, seq, seq)
                weights = torch.softmax(scores, dim=-1)
                self._w = weights
                return x, weights

        w = self._wrapper_with(ExplicitAttnBlock)
        # model returns a tuple; drive the hook directly
        out = w._extract_attention_hook(torch.zeros(1, seq, 4), layers=None)
        assert set(out) == {0, 1, 2}
        for tensor in out.values():
            assert tensor.shape == (1, heads, seq, seq)

    def test_fused_kernel_yields_nothing(self):
        """STATE-style: F.scaled_dot_product_attention never materialises weights."""
        seq = self.SEQ

        class FusedBlock(torch.nn.Module):
            def forward(self, x):
                q = k = v = x.unsqueeze(1)
                return torch.nn.functional.scaled_dot_product_attention(q, k, v).squeeze(1)

        w = self._wrapper_with(FusedBlock)
        out = w._extract_attention_hook(torch.rand(1, seq, 4), layers=None)
        assert out == {}, "a fused kernel cannot expose weights; nothing should be captured"

    def test_extract_attention_raises_informatively_for_fused_models(self):
        seq = self.SEQ

        class FusedBlock(torch.nn.Module):
            def forward(self, x):
                q = k = v = x.unsqueeze(1)
                return torch.nn.functional.scaled_dot_product_attention(q, k, v).squeeze(1)

        w = self._wrapper_with(FusedBlock)
        with pytest.raises(NotImplementedError, match="fused attention kernel"):
            w.extract_attention(torch.rand(1, seq, 4))

    def test_square_non_attention_output_is_not_mistaken_for_attention(self):
        """A square activation whose rows do not sum to 1 must be rejected."""

        class SquareButNotAttention(torch.nn.Module):
            def forward(self, x):
                return torch.full((1, 4, 4), 3.0)

        w = self._wrapper_with(SquareButNotAttention)
        assert w._extract_attention_hook(torch.zeros(1, 4), layers=None) == {}

    def test_layer_subset_and_negative_index(self):
        seq, heads = self.SEQ, self.HEADS

        class ExplicitAttnBlock(torch.nn.Module):
            def forward(self, x):
                return x, torch.softmax(torch.rand(1, heads, seq, seq), dim=-1)

        w = self._wrapper_with(ExplicitAttnBlock, n_layers=4)
        assert set(w._extract_attention_hook(torch.zeros(1, seq, 4), layers=[0, 2])) == {0, 2}
        assert set(w._extract_attention_hook(torch.zeros(1, seq, 4), layers=[-1])) == {3}

    def test_out_of_range_layer_raises(self):
        class ExplicitAttnBlock(torch.nn.Module):
            def forward(self, x):
                return x, torch.softmax(torch.rand(1, 2, 6, 6), dim=-1)

        w = self._wrapper_with(ExplicitAttnBlock)
        with pytest.raises(IndexError, match="out of range"):
            w._extract_attention_hook(torch.zeros(1, 6, 4), layers=[99])

    def test_multihead_attention_need_weights_is_forced_on(self):
        """nn.TransformerEncoderLayer hardcodes need_weights=False internally."""
        d_model, heads, seq = 8, 2, self.SEQ

        def block():
            return torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=16, batch_first=True
            )

        w = self._wrapper_with(block, n_layers=2)
        out = w._extract_attention_hook(torch.rand(1, seq, d_model), layers=None)
        # the pre-hook flips need_weights back on, so weights become observable
        assert out, "pre-hook should have forced need_weights=True on MultiheadAttention"
        for tensor in out.values():
            assert tensor.shape[-1] == tensor.shape[-2] == seq
