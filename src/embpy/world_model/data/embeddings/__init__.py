"""Action embedding providers.

The world model treats a perturbation as the embedding of the
perturbed gene(s). This subpackage exposes a small ABC,
:class:`ActionEmbeddingProvider`, plus two backends:

* :class:`PrecomputedProvider` -- wraps a CSV / NPZ on disk (the
  legacy code path).
* :class:`BioEmbedderProvider` -- delegates to
  :class:`embpy.embedder.BioEmbedder`, with a disk-backed cache.

Use :func:`build_provider` to construct one from an
:class:`world_model.configs.ActionEmbeddingConfig`. The two backends
return identical ``(table, indexer)`` tuples so downstream code is
agnostic to the source.
"""

from __future__ import annotations

from .bio_embedder import BioEmbedderProvider
from .cache import EmbeddingCacheKey, load_cached, save_cached
from .precomputed import PrecomputedProvider
from .provider import ActionEmbeddingProvider
from .registry import build_provider

__all__ = [
    "ActionEmbeddingProvider",
    "BioEmbedderProvider",
    "EmbeddingCacheKey",
    "PrecomputedProvider",
    "build_provider",
    "load_cached",
    "save_cached",
]
