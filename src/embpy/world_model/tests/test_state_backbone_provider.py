"""Tests for :class:`StateBackbone` with a mocked StateEmbeddingWrapper.

No real STATE weights are downloaded; we replace the wrapper at the
import path the provider's ``_load`` consults
(``embpy.models.singlecell_models.StateEmbeddingWrapper``).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------
# Fake StateEmbeddingWrapper
# ---------------------------------------------------------------------


class _FakeInferer:
    def __init__(self, dim: int = 768) -> None:
        self.model = torch.nn.Linear(dim, dim)
        self.model.z_dim = dim  # type: ignore[attr-defined]
        self.model.z_dim_ds = 0  # type: ignore[attr-defined]


class _FakeStateWrapper:
    """Deterministic stand-in for StateEmbeddingWrapper."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inferer: Any = None
        self.embed_calls = 0
        self.decode_calls = 0
        self.dim = 768

    def load(self, device: str = "cpu") -> None:
        self._inferer = _FakeInferer(dim=self.dim)
        self.device = device

    def embed_cells(self, adata: Any) -> np.ndarray:
        self.embed_calls += 1
        n = int(getattr(adata, "n_obs", 0))
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, self.dim)).astype(np.float32)

    def decode_cells(
        self,
        latent: np.ndarray,
        *,
        gene_names: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        self.decode_calls += 1
        return np.zeros((latent.shape[0], len(gene_names)), dtype=np.float32)


def _install_fake_singlecell_module(monkeypatch: pytest.MonkeyPatch) -> _FakeStateWrapper:
    """Replace embpy.models.singlecell_models.StateEmbeddingWrapper for the test."""
    fake_mod = types.ModuleType("embpy.models.singlecell_models")
    captured = {"instance": None}

    def _ctor(*args: Any, **kwargs: Any) -> _FakeStateWrapper:
        inst = _FakeStateWrapper(*args, **kwargs)
        captured["instance"] = inst
        return inst

    fake_mod.StateEmbeddingWrapper = _ctor  # type: ignore[attr-defined]
    fake_mod.StackWrapper = object  # type: ignore[attr-defined]  -- harmless placeholder
    monkeypatch.setitem(sys.modules, "embpy.models.singlecell_models", fake_mod)
    return captured  # type: ignore[return-value]


def _fake_adata(n: int = 16) -> Any:
    class _A:
        def __init__(self, n: int) -> None:
            self.n_obs = n
            self.n_vars = 32
            self.var_names = [f"g{i}" for i in range(self.n_vars)]
            self.obs_names = [f"c{i}" for i in range(self.n_obs)]

    return _A(n)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def _import_state_backbone():
    return importlib.import_module(
        "embpy.world_model.models.encoders.backbones.state"
    )


def test_state_backbone_encode_returns_correct_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_singlecell_module(monkeypatch)
    mod = _import_state_backbone()

    provider = mod.StateBackbone(checkpoint="dummy.ckpt", freeze=False, cache_dir=None)
    adata = _fake_adata(n=20)
    emb = provider.encode(adata)

    assert emb.shape == (20, 768)
    assert emb.dtype == np.float32
    assert provider.embedding_dim == 768
    assert captured["instance"].embed_calls == 1


def test_state_backbone_cache_hit_skips_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _install_fake_singlecell_module(monkeypatch)
    mod = _import_state_backbone()

    provider = mod.StateBackbone(
        checkpoint="dummy.ckpt", freeze=False, cache_dir=tmp_path,
    )
    adata = _fake_adata(n=10)
    emb1 = provider.encode(adata)
    assert captured["instance"].embed_calls == 1

    provider2 = mod.StateBackbone(
        checkpoint="dummy.ckpt", freeze=False, cache_dir=tmp_path,
    )
    emb2 = provider2.encode(adata)
    np.testing.assert_array_equal(emb1, emb2)
    # second provider's wrapper was never embedded because cache hit
    assert captured["instance"].embed_calls == 1


def test_state_backbone_freeze_disables_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_singlecell_module(monkeypatch)
    mod = _import_state_backbone()

    provider = mod.StateBackbone(checkpoint="dummy.ckpt", freeze=True, cache_dir=None)
    provider.encode(_fake_adata(n=4))  # triggers load + freeze
    params = list(provider.parameters())
    assert params, "fake wrapper should expose parameters"
    assert all(not p.requires_grad for p in params)


def test_state_backbone_decode_forwards_to_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_singlecell_module(monkeypatch)
    mod = _import_state_backbone()

    provider = mod.StateBackbone(checkpoint="dummy.ckpt", freeze=False, cache_dir=None)
    provider.encode(_fake_adata(n=4))
    out = provider.decode(np.zeros((3, 768), dtype=np.float32), gene_names=["a", "b"])
    assert out.shape == (3, 2)
    assert captured["instance"].decode_calls == 1


def test_state_backbone_require_cache_hit_raises_on_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_fake_singlecell_module(monkeypatch)
    mod = _import_state_backbone()
    provider = mod.StateBackbone(
        checkpoint="dummy.ckpt",
        freeze=False,
        cache_dir=tmp_path,
        require_cache_hit=True,
    )
    with pytest.raises(RuntimeError, match="require_cache_hit"):
        provider.encode(_fake_adata(n=4))
