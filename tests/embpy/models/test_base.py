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
