"""Abstract base class for action-embedding providers.

A provider answers two questions consistent with the legacy
``load_gene_embedding_table`` contract:

* :meth:`embed`        -> ``(len(symbols), embedding_dim)`` matrix,
                          rows for unresolvable symbols are zero.
* :meth:`build_table`  -> ``(table, indexer)`` aligned to ``symbols``,
                          with row 0 reserved for the control / padding
                          token.

Adding a new backend (e.g. fetching from a vector database) is a
matter of subclassing :class:`ActionEmbeddingProvider`, implementing
:meth:`embed`, and registering it in :mod:`registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..datasets.base import GeneIndexer


@dataclass
class ProviderMetadata:
    """Small JSON-serialisable description of what the provider did.

    Persisted as ``outputs/<run_id>/action_embedding_meta.json`` by
    :func:`world_model.data.build_dataloaders` so a stale or wrong
    embedding table is one ``cat`` away from being noticed.
    """

    source: str
    embedding_dim: int
    n_symbols: int
    n_unresolved: int
    model_name: str | None = None
    region: str | None = None
    pooling_strategy: str | None = None
    organism: str | None = None
    cache_path: str | None = None
    extras: dict[str, Any] | None = None


class ActionEmbeddingProvider(ABC):
    """Common interface for every action-embedding backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in metadata / logs (e.g. ``"esm2_650M"``)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding rows."""

    @abstractmethod
    def embed(self, symbols: Sequence[str]) -> np.ndarray:
        """Return ``(len(symbols), embedding_dim)`` rows.

        Rows for unresolvable symbols MUST be zero (not ``nan``); the
        downstream :class:`GeneEmbeddingAction` already treats index 0
        as padding, but symbols that *resolved to a known gene that
        the model could not embed* still go into the table at non-zero
        rows -- they only get zeroed out here.
        """

    def build_table(self, symbols: Sequence[str]) -> tuple[np.ndarray, GeneIndexer]:
        """Build a ``(n + 1, embedding_dim)`` table aligned to ``symbols``.

        Default implementation calls :meth:`embed` once over the
        deduplicated symbol list and assembles the table. Subclasses
        whose source already hands back ``(table, indexer)`` (notably
        :class:`PrecomputedProvider`) can override this to skip the
        intermediate copy.
        """
        indexer = GeneIndexer.from_symbols(symbols)
        # Materialise rows in indexer order so position i+1 in the
        # table corresponds to indexer.index_to_symbol[i+1].
        ordered = [
            indexer.index_to_symbol[i] for i in range(1, len(indexer))
        ]
        rows = self.embed(ordered) if ordered else np.zeros((0, self.embedding_dim), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != len(ordered):
            raise ValueError(
                f"Provider returned shape {rows.shape}, expected ({len(ordered)}, *)."
            )
        table = np.zeros((len(indexer), rows.shape[1]), dtype=np.float32)
        if rows.shape[0]:
            table[1:] = rows
        return table, indexer

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        """Default provider metadata; subclasses override for richer fields."""
        return ProviderMetadata(
            source=type(self).__name__,
            embedding_dim=self.embedding_dim,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved),
        )


__all__ = ["ActionEmbeddingProvider", "ProviderMetadata"]
