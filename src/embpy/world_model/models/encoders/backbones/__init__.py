"""Single-cell foundation backbones for the world-model state encoder.

Phase 5 introduces a provider abstraction over the observation encoder
so the world model can plug into a real frozen foundation backbone
(STATE, STACK) instead of the small from-scratch
:class:`StateStackEncoder`. Three concrete providers are shipped:

* :class:`LocalStackBackbone`  -- wraps the existing
  :class:`StateStackEncoder`. Default; bit-equivalent to Phase 1-4.
* :class:`StateBackbone`       -- adapter over
  :class:`embpy.models.singlecell_models.StateEmbeddingWrapper`.
  Lazy-imports ``arc-state``.
* :class:`StackBackbone`       -- adapter over
  :class:`embpy.models.singlecell_models.StackWrapper`.
  Lazy-imports ``arc-stack``.

All three implement :class:`StateBackboneProvider`. Build them via
:func:`build_backbone`. The dataloader pre-encodes the full AnnData
through ``provider.encode(...)`` once (cached to disk via
:mod:`cache`) and downstream training sees only ``(N, embedding_dim)``
arrays.
"""

from __future__ import annotations

from .cache import inspect_cache, load_cached, save_cached
from .foreign_head import ForeignBackboneHead
from .local import LocalBackbone, LocalStackBackbone
from .provider import ProviderMetadata, StateBackboneProvider
from .registry import build_backbone
from .stack import StackBackbone
from .state import StateBackbone

__all__ = [
    "ForeignBackboneHead",
    "LocalBackbone",
    "LocalStackBackbone",
    "ProviderMetadata",
    "StackBackbone",
    "StateBackbone",
    "StateBackboneProvider",
    "build_backbone",
    "inspect_cache",
    "load_cached",
    "save_cached",
]
