"""Precomputed CSV / NPZ provider (legacy code path).

Wraps :func:`world_model.data.datasets.base.load_gene_embedding_table`
behind the :class:`ActionEmbeddingProvider` ABC so the rest of the
package can stay backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..datasets.base import GeneIndexer, load_gene_embedding_table
from .provider import ActionEmbeddingProvider, ProviderMetadata


class PrecomputedProvider(ActionEmbeddingProvider):
    """Loads embeddings from a CSV (rows: gene -> embedding) or NPZ archive.

    Parameters
    ----------
    path
        Either a CSV with the gene symbol in the first column, or a
        NPZ archive carrying ``"symbols"`` and ``"embeddings"`` arrays.
    """

    def __init__(self, path: str | Path) -> None:
        if not path:
            raise ValueError("PrecomputedProvider needs a non-empty path.")
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self._path}")
        self._embedding_dim: int | None = None
        self._last_n_unresolved: int = 0

    @property
    def name(self) -> str:
        return f"precomputed:{self._path.name}"

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            # Cheap probe: load the file, infer the embedding dim, and discard.
            tbl, _ = load_gene_embedding_table(self._path, symbols=[])
            self._embedding_dim = int(tbl.shape[1]) if tbl.size else 0
        return self._embedding_dim

    def embed(self, symbols: Sequence[str]) -> np.ndarray:
        # The default :meth:`build_table` path uses :meth:`embed` to fill
        # rows 1..n. Reuse the legacy loader to honour its CSV / NPZ
        # parsing and missing-symbol handling.
        if not symbols:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        table, indexer = load_gene_embedding_table(self._path, symbols=symbols)
        self._embedding_dim = int(table.shape[1])
        # Detect rows that the loader filled with zeros because the symbol
        # was missing; we report the count via :meth:`metadata`.
        unresolved = 0
        for sym in symbols:
            row = indexer.symbol_to_index.get(sym, 0)
            if row == 0:
                continue
            if not np.any(table[row]):
                unresolved += 1
        self._last_n_unresolved = unresolved
        return np.stack(
            [table[indexer.symbol_to_index[s]] for s in symbols], axis=0,
        ).astype(np.float32)

    def build_table(self, symbols: Sequence[str]) -> tuple[np.ndarray, GeneIndexer]:
        # Skip the default round-trip: the legacy loader already builds
        # a (table, indexer) of the right shape directly.
        table, indexer = load_gene_embedding_table(self._path, symbols=symbols)
        self._embedding_dim = int(table.shape[1])
        unresolved = 0
        for sym, idx in indexer.symbol_to_index.items():
            if idx == 0:
                continue
            if not np.any(table[idx]):
                unresolved += 1
        self._last_n_unresolved = unresolved
        return table, indexer

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        return ProviderMetadata(
            source="precomputed",
            embedding_dim=self.embedding_dim,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved or self._last_n_unresolved),
            model_name=str(self._path),
        )


__all__ = ["PrecomputedProvider"]
