"""Action embedding providers.

The world model treats a perturbation as the embedding of the
perturbed gene(s). This subpackage exposes a small ABC,
:class:`ActionEmbeddingProvider`, plus two backends:

* :class:`PrecomputedProvider` -- wraps a CSV / NPZ on disk (the
  legacy code path).
* :class:`BioEmbedderProvider` -- delegates to
  :class:`embpy.embedder.BioEmbedder`, with a disk-backed cache.

Both backends speak the status-aware contract introduced in Part A.2:
:meth:`ActionEmbeddingProvider.embed_with_status` returns
``(rows, statuses)`` where each row in ``statuses`` is an
:class:`EmbeddingStatus` value (``RESOLVED`` / ``CONTROL`` /
``UNRESOLVED``).

Use :func:`build_provider` to construct one from an
:class:`world_model.configs.ActionEmbeddingConfig`. The two backends
return identical ``(table, indexer)`` tuples so downstream code is
agnostic to the source.
"""

from __future__ import annotations

from .bio_embedder import BioEmbedderProvider
from .cache import EmbeddingCacheKey, load_cached, save_cached
from .precomputed import PrecomputedProvider
from .provider import ActionEmbeddingProvider, ProviderMetadata
from .registry import build_provider
from .sentinel import (
    CONTROL_SENTINEL_SEED,
    EmbeddingStatus,
    describe_status_counts,
    make_control_vector,
    make_unresolved_vector,
    status_array,
)

__all__ = [
    "CONTROL_SENTINEL_SEED",
    "ActionEmbeddingProvider",
    "BioEmbedderProvider",
    "EmbeddingCacheKey",
    "EmbeddingStatus",
    "PrecomputedProvider",
    "ProviderMetadata",
    "build_provider",
    "describe_status_counts",
    "load_cached",
    "make_control_vector",
    "make_unresolved_vector",
    "save_cached",
    "status_array",
]
