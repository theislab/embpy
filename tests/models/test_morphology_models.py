"""Tests for embpy.models.morphology_models.SubCellWrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestSubCellWrapperInit:
    def test_default_params(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper()
        assert w.model_type == "morphology"
        assert w.variant == "contrast"
        assert w.image_size == 448

    def test_custom_variant(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper(variant="mae")
        assert w.variant == "mae"

    def test_pooling_strategies(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper()
        assert "cls" in w.available_pooling_strategies
        assert "mean" in w.available_pooling_strategies
        assert "attention_pool" in w.available_pooling_strategies


class TestSubCellPreprocessing:
    @pytest.fixture
    def wrapper(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper()
        w.device = torch.device("cpu")
        return w

    def test_numpy_hwc(self, wrapper):
        img = np.random.rand(100, 100, 4).astype(np.float32)
        tensor = wrapper._preprocess_image(img)
        assert tensor.shape == (1, 4, 448, 448)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_numpy_chw(self, wrapper):
        img = np.random.rand(4, 200, 200).astype(np.float32)
        tensor = wrapper._preprocess_image(img)
        assert tensor.shape == (1, 4, 448, 448)

    def test_torch_tensor(self, wrapper):
        img = torch.rand(4, 300, 300)
        tensor = wrapper._preprocess_image(img)
        assert tensor.shape == (1, 4, 448, 448)

    def test_wrong_channels_raises(self, wrapper):
        img = np.random.rand(3, 100, 100).astype(np.float32)
        with pytest.raises(ValueError, match="4 channels"):
            wrapper._preprocess_image(img)

    def test_already_correct_size(self, wrapper):
        img = np.random.rand(4, 448, 448).astype(np.float32)
        tensor = wrapper._preprocess_image(img)
        assert tensor.shape == (1, 4, 448, 448)


class TestSubCellEmbed:
    @pytest.fixture
    def mock_wrapper(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper()
        w.device = torch.device("cpu")

        n_patches = 784
        hidden_dim = 768
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, n_patches + 1, hidden_dim)

        mock_encoder = MagicMock()
        mock_encoder.return_value = mock_output
        w._encoder = mock_encoder
        w.model = mock_encoder
        w._pool_model = None

        return w

    def test_cls_pooling(self, mock_wrapper):
        img = np.random.rand(4, 448, 448).astype(np.float32)
        emb = mock_wrapper.embed(img, pooling_strategy="cls")
        assert emb.shape == (768,)
        assert emb.dtype == np.float32

    def test_mean_pooling(self, mock_wrapper):
        img = np.random.rand(4, 448, 448).astype(np.float32)
        emb = mock_wrapper.embed(img, pooling_strategy="mean")
        assert emb.shape == (768,)

    def test_none_pooling(self, mock_wrapper):
        img = np.random.rand(4, 448, 448).astype(np.float32)
        emb = mock_wrapper.embed(img, pooling_strategy="none")
        assert emb.ndim == 2
        assert emb.shape == (785, 768)
        assert emb.dtype == np.float32

    def test_attention_pool_fallback(self, mock_wrapper):
        img = np.random.rand(4, 448, 448).astype(np.float32)
        emb = mock_wrapper.embed(img, pooling_strategy="attention_pool")
        assert emb.shape == (768,)

    def test_invalid_pooling_raises(self, mock_wrapper):
        img = np.random.rand(4, 448, 448).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown pooling"):
            mock_wrapper.embed(img, pooling_strategy="invalid")

    def test_not_loaded_raises(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper()
        img = np.random.rand(4, 448, 448).astype(np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed(img)


class TestSubCellBatch:
    @pytest.fixture
    def mock_wrapper(self):
        from embpy.models.morphology_models import SubCellWrapper

        w = SubCellWrapper()
        w.device = torch.device("cpu")

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 785, 768)
        mock_encoder = MagicMock()
        mock_encoder.return_value = mock_output
        w._encoder = mock_encoder
        w.model = mock_encoder
        w._pool_model = None

        return w

    def test_batch_returns_list(self, mock_wrapper):
        imgs = [np.random.rand(4, 100, 100).astype(np.float32) for _ in range(3)]
        results = mock_wrapper.embed_batch(imgs)
        assert len(results) == 3
        assert all(r.shape == (768,) for r in results)

    def test_batch_none_pooling(self, mock_wrapper):
        imgs = [np.random.rand(4, 100, 100).astype(np.float32) for _ in range(3)]
        results = mock_wrapper.embed_batch(imgs, pooling_strategy="none")
        assert len(results) == 3
        assert all(r.ndim == 2 for r in results)
        assert all(r.shape == (785, 768) for r in results)


class TestAttentionPooler:
    def test_gated_attention_pooler(self):
        from embpy.models.morphology_models import _build_attention_pooler

        pooler = _build_attention_pooler(dim=768, int_dim=512, num_heads=2)
        x = torch.randn(1, 100, 768)
        out, attn = pooler(x)
        assert out.shape == (1, 1536)
        assert attn.shape == (1, 2, 100)
