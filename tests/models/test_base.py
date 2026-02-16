"""Tests for BaseModelWrapper._apply_pooling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embpy.models.base import BaseModelWrapper


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
