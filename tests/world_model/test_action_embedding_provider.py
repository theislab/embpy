"""Tests for the action-embedding providers and disk cache.

The :class:`BioEmbedderProvider` is exercised against a *mock*
:class:`embpy.embedder.BioEmbedder` so the suite does not download
real model weights. Cache hits, missing-symbol handling, and atomic
writes are all checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from world_model.data.embeddings import (
    BioEmbedderProvider,
    EmbeddingCacheKey,
    PrecomputedProvider,
    load_cached,
    save_cached,
)
from world_model.data.embeddings import bio_embedder as bio_embedder_module


# ---------------------------------------------------------------------
# Fake BioEmbedder for tests
# ---------------------------------------------------------------------


class _FakeBioEmbedder:
    """Deterministic BioEmbedder replacement with a call counter."""

    def __init__(self, *args: Any, dim: int = 8, unresolvable: tuple[str, ...] = (), **kwargs: Any) -> None:
        self.dim = int(dim)
        self.unresolvable = set(unresolvable)
        self.call_count = 0
        self.symbol_calls: list[list[str]] = []

    def embed_genes_batch(  # type: ignore[no-untyped-def]
        self, *, model: str, identifiers, **kwargs,
    ):
        identifiers = list(identifiers)
        self.call_count += 1
        self.symbol_calls.append(identifiers)
        out: list[np.ndarray | None] = []
        for i, sym in enumerate(identifiers):
            if sym in self.unresolvable:
                out.append(None)
            else:
                vec = np.linspace(i * 0.1, i * 0.1 + self.dim, self.dim, dtype=np.float32)
                out.append(vec)
        return out


def _install_fake_embedder_module(monkeypatch, factory) -> None:
    """Inject a stub ``embpy.embedder`` so the BioEmbedderProvider's lazy
    import resolves to ``factory`` without pulling the heavy real module.

    This sidesteps environments where importing the real
    :mod:`embpy.embedder` is broken (e.g. torchvision / transformers
    binary mismatches in the dev env) -- the provider only needs the
    ``BioEmbedder`` symbol on the module.
    """
    import sys
    import types

    stub = types.ModuleType("embpy.embedder")
    stub.BioEmbedder = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "embpy.embedder", stub)


@pytest.fixture
def patch_bio_embedder(monkeypatch):
    """Replace ``embpy.embedder.BioEmbedder`` with the fake."""
    fake_holder: dict[str, _FakeBioEmbedder] = {}

    def factory(*args: Any, **kwargs: Any) -> _FakeBioEmbedder:
        inst = _FakeBioEmbedder(*args, **kwargs)
        fake_holder.setdefault("instance", inst)
        return inst

    _install_fake_embedder_module(monkeypatch, factory)
    return fake_holder


# ---------------------------------------------------------------------
# Precomputed provider
# ---------------------------------------------------------------------


def test_precomputed_provider_returns_table_with_zero_row(tmp_path: Path):
    syms = ["GENE_A", "GENE_B", "GENE_C"]
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    npz_path = tmp_path / "tiny.npz"
    np.savez(npz_path, symbols=np.asarray(syms, dtype=object), embeddings=emb)

    provider = PrecomputedProvider(npz_path)
    table, indexer = provider.build_table(syms)
    assert table.shape == (4, 4)
    np.testing.assert_array_equal(table[0], np.zeros(4, dtype=np.float32))
    for i, sym in enumerate(syms):
        idx = indexer.symbol_to_index[sym]
        np.testing.assert_array_equal(table[idx], emb[i])


# ---------------------------------------------------------------------
# BioEmbedder provider
# ---------------------------------------------------------------------


def test_bio_embedder_provider_embeds_and_caches(tmp_path: Path, patch_bio_embedder):
    provider = BioEmbedderProvider(
        model_name="fake_model",
        cache_dir=tmp_path / "cache",
    )
    syms = ["AAA", "BBB", "CCC"]
    out = provider.embed(syms)
    assert out.shape == (3, 8)
    assert provider.embedding_dim == 8
    assert patch_bio_embedder["instance"].call_count == 1

    # Second call with the same symbols must NOT trigger a new embedder call.
    out2 = provider.embed(syms)
    np.testing.assert_array_equal(out, out2)
    assert patch_bio_embedder["instance"].call_count == 1


def test_bio_embedder_provider_partial_cache(tmp_path: Path, patch_bio_embedder):
    provider = BioEmbedderProvider(
        model_name="fake_model",
        cache_dir=tmp_path / "cache",
    )
    provider.embed(["A", "B"])
    n_initial = patch_bio_embedder["instance"].call_count
    out = provider.embed(["A", "B", "C", "D"])
    assert out.shape == (4, 8)
    # Only C, D should have hit the embedder (one extra batch call).
    assert patch_bio_embedder["instance"].call_count == n_initial + 1
    last = patch_bio_embedder["instance"].symbol_calls[-1]
    assert sorted(last) == ["C", "D"]


def test_bio_embedder_provider_unresolved_rows_are_zero(tmp_path: Path, monkeypatch):
    def factory(*args, **kwargs):
        return _FakeBioEmbedder(unresolvable=("BAD",))

    _install_fake_embedder_module(monkeypatch, factory)

    provider = BioEmbedderProvider(
        model_name="fake_model",
        cache_dir=tmp_path / "cache",
    )
    out = provider.embed(["GOOD", "BAD"])
    assert out.shape == (2, 8)
    np.testing.assert_array_equal(out[1], np.zeros(8, dtype=np.float32))
    assert "BAD" in provider._last_unresolved


def test_bio_embedder_provider_build_table_shape(tmp_path: Path, patch_bio_embedder):
    provider = BioEmbedderProvider(
        model_name="fake_model",
        cache_dir=tmp_path / "cache",
    )
    table, indexer = provider.build_table(["ALPHA", "BETA"])
    assert table.shape == (3, 8)
    np.testing.assert_array_equal(table[0], np.zeros(8, dtype=np.float32))
    assert indexer.symbol_to_index["ALPHA"] == 1
    assert indexer.symbol_to_index["BETA"] == 2


# ---------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------


def test_cache_save_and_load_roundtrip(tmp_path: Path):
    key = EmbeddingCacheKey(model_name="m", region="full", pooling_strategy="mean", organism="human")
    syms = ["G1", "G2", "G3"]
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    save_cached(tmp_path, key, syms, emb)

    cached = load_cached(tmp_path, key, ["G1", "G2", "MISSING"])
    assert set(cached.keys()) == {"G1", "G2"}
    np.testing.assert_array_equal(cached["G1"], emb[0])


def test_cache_overlapping_writes_keep_latest_values(tmp_path: Path):
    key = EmbeddingCacheKey(model_name="m")
    save_cached(tmp_path, key, ["A", "B"], np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))
    save_cached(tmp_path, key, ["B", "C"], np.array([[20.0, 20.0], [3.0, 3.0]], dtype=np.float32))
    cached = load_cached(tmp_path, key, ["A", "B", "C"])
    np.testing.assert_array_equal(cached["A"], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(cached["B"], np.array([20.0, 20.0], dtype=np.float32))
    np.testing.assert_array_equal(cached["C"], np.array([3.0, 3.0], dtype=np.float32))


def test_cache_handles_dim_change(tmp_path: Path):
    key = EmbeddingCacheKey(model_name="m")
    save_cached(tmp_path, key, ["X"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    save_cached(tmp_path, key, ["X"], np.array([[10.0, 20.0]], dtype=np.float32))
    cached = load_cached(tmp_path, key, ["X"])
    np.testing.assert_array_equal(cached["X"], np.array([10.0, 20.0], dtype=np.float32))
