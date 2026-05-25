"""Abstract base class for action-embedding providers.

A provider answers three questions:

* :meth:`embed_with_status`  -- ``((N, D) embeddings, (N,) statuses)``.
                                The status array uses
                                :class:`world_model.data.embeddings.sentinel.EmbeddingStatus`
                                and is the canonical way to tell apart
                                RESOLVED genes, CONTROL sentinels, and
                                UNRESOLVED zero rows.
* :meth:`embed`              -- legacy ``(N, D)`` ndarray. Kept for one
                                release with a ``DeprecationWarning``;
                                wraps :meth:`embed_with_status`.
* :meth:`build_table`        -- ``(table, indexer)`` aligned to ``symbols``,
                                with row 0 reserved for the control /
                                padding token. Implemented in this base
                                class on top of :meth:`embed_with_status`
                                so subclasses only override the latter.

Adding a new backend (e.g. fetching from a vector database) is a
matter of subclassing :class:`ActionEmbeddingProvider`, implementing
:meth:`embed_with_status`, and registering it in :mod:`registry`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from world_model.data.datasets.base import GeneIndexer
from world_model.data.embeddings.sentinel import EmbeddingStatus, make_control_vector


@dataclass
class ProviderMetadata:
    """Small JSON-serialisable description of what the provider did.

    Persisted as ``runs/<run_id>/action_embedding_meta.json`` by
    :func:`world_model.data.build_dataloaders` so a stale or wrong
    embedding table is one ``cat`` away from being noticed.

    The new per-bucket counts (``n_resolved`` / ``n_control`` /
    ``n_unresolved``) plus the explicit symbol lists are populated by
    the status-aware code path in Part A.2.
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

    n_resolved: int = 0
    n_control: int = 0
    n_mixed: int = 0
    unresolved_symbols: list[str] = field(default_factory=list)
    control_symbols: list[str] = field(default_factory=list)
    mixed_symbols: list[str] = field(default_factory=list)
    control_sentinel_seed: int | None = None


class ActionEmbeddingProvider(ABC):
    """Common interface for every action-embedding backend.

    Subclasses must implement :meth:`embed_with_status`. The legacy
    :meth:`embed` and the table-builder default to using it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in metadata / logs (e.g. ``"esm2_650M"``)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding rows."""

    @abstractmethod
    def embed_with_status(
        self,
        symbols: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``((N, D) rows, (N,) statuses)`` aligned to ``symbols``.

        Implementations MUST:

        * run :class:`ControlPolicy.classify` (or equivalent) FIRST and
          short-circuit control rows with a deterministic sentinel
          vector via :func:`make_control_vector`. Never send control
          labels to the underlying embedder.
        * emit zero rows for UNRESOLVED symbols, but always pair them
          with a structured WARNING log line.
        * set ``self._embedding_dim`` so :attr:`embedding_dim` returns
          the correct value on subsequent property reads.
        """

    def embed(self, symbols: Sequence[str]) -> np.ndarray:
        """Legacy single-return API.

        Deprecated; use :meth:`embed_with_status` instead. This wrapper
        will be removed in the next minor release.
        """
        warnings.warn(
            "ActionEmbeddingProvider.embed(symbols) is deprecated and will "
            "be removed in the next release. Use embed_with_status(symbols) "
            "to receive a per-row EmbeddingStatus array alongside the "
            "embeddings; controls now emit a deterministic non-zero "
            "sentinel vector instead of silently mapping to a zero row.",
            DeprecationWarning,
            stacklevel=2,
        )
        rows, _ = self.embed_with_status(symbols)
        return rows

    def build_table(self, symbols: Sequence[str]) -> tuple[np.ndarray, GeneIndexer]:
        """Build a ``(n + 1, embedding_dim)`` table aligned to ``symbols``.

        Default implementation calls :meth:`embed_with_status` once over
        the deduplicated symbol list and assembles the table. Subclasses
        whose source already hands back ``(table, indexer)`` can override
        this to skip the intermediate copy.

        Row 0 is the "<control>" padding row. Its concrete contents are
        controlled by the dataset / provider: in the status-aware code
        path, row 0 is left at its existing zero default because the
        per-row CONTROL sentinel vector is materialised *at use time*
        when the gene-action table is populated by the dataloader.
        """
        indexer = GeneIndexer.from_symbols(symbols)
        ordered = [indexer.index_to_symbol[i] for i in range(1, len(indexer))]
        if ordered:
            rows, _statuses = self.embed_with_status(ordered)
        else:
            rows = np.zeros((0, self.embedding_dim), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != len(ordered):
            raise ValueError(f"Provider returned shape {rows.shape}, expected ({len(ordered)}, *).")
        dim = int(rows.shape[1]) if rows.size else int(self.embedding_dim)
        table = np.zeros((len(indexer), dim), dtype=np.float32)
        if rows.shape[0]:
            table[1:] = rows
        return table, indexer

    # ------------------------------------------------------------------
    # Helpers shared by every subclass
    # ------------------------------------------------------------------

    @staticmethod
    def _control_row(dim: int, *, seed: int = 0) -> np.ndarray:
        """Convenience accessor for :func:`make_control_vector`."""
        return make_control_vector(dim, seed=seed)

    @staticmethod
    def _status_array(statuses: list[EmbeddingStatus]) -> np.ndarray:
        return np.asarray([s.value for s in statuses], dtype=object)

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        """Default provider metadata; subclasses override for richer fields."""
        return ProviderMetadata(
            source=type(self).__name__,
            embedding_dim=self.embedding_dim,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved),
        )


__all__ = [
    "ActionEmbeddingProvider",
    "ProviderMetadata",
]
