"""Embedding-native storage and AnnData semantic accessor for embpy."""

from __future__ import annotations

from .accessor import EmbpyAccessor, register_anndata_accessor
from .core import EmbeddingBlock, EmbeddingStore, RelationTable

register_anndata_accessor()

__all__ = [
    "EmbeddingBlock",
    "EmbeddingStore",
    "EmbpyAccessor",
    "RelationTable",
    "register_anndata_accessor",
]
