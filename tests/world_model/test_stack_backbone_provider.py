"""Tests for :class:`StackBackbone` with a mocked StackWrapper.

No real STACK weights are downloaded; we replace the wrapper at the
import path the provider's ``_load`` consults
(``embpy.models.singlecell_models.StackWrapper``).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------
# Fake StackWrapper
# ---------------------------------------------------------------------


class _FakeStackModel(torch.nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)
        self.embedding_dim = dim


class _FakeStackWrapper:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._model: Any = None
        self.embed_calls = 0
        self.generate_calls = 0
        self.dim = 512

    def load(self, device: str = "cpu") -> None:
        self._model = _FakeStackModel(dim=self.dim)
        self.device = device

    def embed_cells(self, adata: Any) -> np.ndarray:
        self.embed_calls += 1
        n = int(getattr(adata, "n_obs", 0))
        rng = np.random.default_rng(7)
        return rng.standard_normal((n, self.dim)).astype(np.float32)

    def generate_cells(self, base_adata: Any, test_adata: Any, **kwargs: Any) -> np.ndarray:
        del base_adata, kwargs
        self.generate_calls += 1
        n = int(getattr(test_adata, "n_obs", 0))
        return np.zeros((n, 64), dtype=np.float32)


def _install_fake_singlecell_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    fake_mod = types.ModuleType("embpy.models.singlecell_models")
    captured: dict[str, Any] = {"instance": None}

    def _ctor(*args: Any, **kwargs: Any) -> _FakeStackWrapper:
        inst = _FakeStackWrapper(*args, **kwargs)
        captured["instance"] = inst
        return inst

    fake_mod.StackWrapper = _ctor  # type: ignore[attr-defined]
    fake_mod.StateEmbeddingWrapper = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "embpy.models.singlecell_models", fake_mod)
    return captured


def _fake_adata(n: int = 8) -> Any:
    class _A:
        def __init__(self, n: int) -> None:
            self.n_obs = n
            self.n_vars = 16
            self.var_names = [f"g{i}" for i in range(self.n_vars)]
            self.obs_names = [f"c{i}" for i in range(self.n_obs)]

    return _A(n)


def _import_stack_backbone():
    return importlib.import_module(
        "world_model.models.encoders.backbones.stack"
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_stack_backbone_encode_returns_correct_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_singlecell_module(monkeypatch)
    mod = _import_stack_backbone()

    provider = mod.StackBackbone(
        checkpoint="dummy.ckpt", genelist="genes.pkl", freeze=False, cache_dir=None,
    )
    emb = provider.encode(_fake_adata(n=12))
    assert emb.shape == (12, 512)
    assert provider.embedding_dim == 512
    assert captured["instance"].embed_calls == 1


def test_stack_backbone_decode_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_singlecell_module(monkeypatch)
    mod = _import_stack_backbone()
    provider = mod.StackBackbone(
        checkpoint="dummy.ckpt", genelist="genes.pkl", freeze=False, cache_dir=None,
    )
    with pytest.raises(NotImplementedError, match="generate_cells"):
        provider.decode(np.zeros((1, 512), dtype=np.float32), gene_names=["a"])


def test_stack_backbone_generate_cells_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_singlecell_module(monkeypatch)
    mod = _import_stack_backbone()
    provider = mod.StackBackbone(
        checkpoint="dummy.ckpt", genelist="genes.pkl", freeze=False, cache_dir=None,
    )
    out = provider.generate_cells(_fake_adata(n=5), _fake_adata(n=3))
    assert out.shape == (3, 64)
    assert captured["instance"].generate_calls == 1


def test_stack_backbone_freeze_disables_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_singlecell_module(monkeypatch)
    mod = _import_stack_backbone()
    provider = mod.StackBackbone(
        checkpoint="dummy.ckpt", genelist="genes.pkl", freeze=True, cache_dir=None,
    )
    provider.encode(_fake_adata(n=2))
    params = list(provider.parameters())
    assert params
    assert all(not p.requires_grad for p in params)


def test_stack_backbone_supports_decode_is_false() -> None:
    from world_model.models.encoders.backbones.stack import StackBackbone

    assert StackBackbone.supports_decode is False
