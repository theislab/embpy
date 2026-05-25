"""Embedding-native storage and AnnData semantic accessor for embpy."""

from __future__ import annotations

from .accessor import EmbpyAccessor, register_anndata_accessor
from .actions import ActionStatus, ActionTable, compile_action_table, control_sentinel_vector
from .core import EmbeddingBlock, EmbeddingStore, RelationTable
from .migrate import gene_store_from_table, migrate_table_to_emstore

register_anndata_accessor()

__all__ = [
    "ActionStatus",
    "ActionTable",
    "EmbeddingBlock",
    "EmbeddingStore",
    "EmbpyAccessor",
    "RelationTable",
    "compile_action_table",
    "control_sentinel_vector",
    "gene_store_from_table",
    "migrate_table_to_emstore",
    "register_anndata_accessor",
]
