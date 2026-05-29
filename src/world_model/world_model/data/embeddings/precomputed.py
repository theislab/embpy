"""Precomputed CSV / NPZ provider (legacy code path).

Wraps :func:`world_model.data.datasets.base.load_gene_embedding_table`
behind the :class:`ActionEmbeddingProvider` ABC so the rest of the
package can stay backend-agnostic.

Status-aware contract (Part A.2): controls and unresolved symbols are
reported through :class:`EmbeddingStatus`. For the precomputed path,
"unresolved" means "the gene is missing from the CSV / NPZ", and "control"
is decided by the policy passed at construction time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from embpy.resources.gene.control import ControlPolicy

from ..datasets.base import GeneIndexer, load_gene_embedding_table
from .provider import ActionEmbeddingProvider, ProviderMetadata
from .sentinel import (
    CONTROL_SENTINEL_SEED,
    EmbeddingStatus,
    make_control_vector,
    make_unresolved_vector,
)


class PrecomputedProvider(ActionEmbeddingProvider):
    """Loads embeddings from a CSV (rows: gene -> embedding) or NPZ archive.

    Parameters
    ----------
    path
        Either a CSV with the gene symbol in the first column, or a
        NPZ archive carrying ``"symbols"`` and ``"embeddings"`` arrays.
    control_policy
        Optional ControlPolicy. Defaults to ``ControlPolicy.default()``.
    control_sentinel_seed
        Seed for the deterministic CONTROL sentinel vector.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        control_policy: ControlPolicy | None = None,
        control_sentinel_seed: int = CONTROL_SENTINEL_SEED,
    ) -> None:
        if not path:
            raise ValueError("PrecomputedProvider needs a non-empty path.")
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self._path}")
        self._embedding_dim: int | None = None
        self._last_n_unresolved: int = 0
        self._last_unresolved: list[str] = []
        self._last_controls: list[str] = []
        self.control_policy: ControlPolicy = control_policy or ControlPolicy.default()
        self.control_sentinel_seed: int = int(control_sentinel_seed)

    @property
    def name(self) -> str:
        return f"precomputed:{self._path.name}"

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            tbl, _ = load_gene_embedding_table(self._path, symbols=[])
            self._embedding_dim = int(tbl.shape[1]) if tbl.size else 0
        return self._embedding_dim

    def embed_with_status(
        self,
        symbols: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        symbols = list(symbols)
        if not symbols:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), np.asarray(
                [], dtype=object,
            )
        classifications = [self.control_policy.classify(s) for s in symbols]
        # The precomputed loader keys by exact gene symbol. For mixed
        # labels we look up each gene component; the row is the mean of
        # the resolved components. For pure-control labels we never
        # touch the CSV / NPZ.
        gene_targets: dict[int, list[str]] = {}
        flat_genes: list[str] = []
        for idx, c in enumerate(classifications):
            if c.kind == "control":
                continue
            gs = list(c.gene_components) if c.kind == "mixed" else list(c.components)
            if not gs:
                continue
            gene_targets[idx] = gs
            flat_genes.extend(gs)
        unique_genes = sorted({g for g in flat_genes if g})

        if not unique_genes:
            dim = self.embedding_dim or 1
            out = np.zeros((len(symbols), dim), dtype=np.float32)
            ctrl_vec = make_control_vector(dim, seed=self.control_sentinel_seed)
            statuses: list[EmbeddingStatus] = []
            controls: list[str] = []
            for i, c in enumerate(classifications):
                out[i] = ctrl_vec
                statuses.append(EmbeddingStatus.CONTROL)
                controls.append(c.label)
            self._last_controls = controls
            self._last_unresolved = []
            self._last_n_unresolved = 0
            return out, self._status_array(statuses)

        table, indexer = load_gene_embedding_table(
            self._path, symbols=unique_genes,
        )
        self._embedding_dim = int(table.shape[1])
        dim = self._embedding_dim

        out = np.zeros((len(symbols), dim), dtype=np.float32)
        statuses_arr: list[EmbeddingStatus] = []
        unresolved_symbols: list[str] = []
        control_symbols: list[str] = []
        ctrl_vec = make_control_vector(dim, seed=self.control_sentinel_seed)

        for i, c in enumerate(classifications):
            if c.kind == "control":
                out[i] = ctrl_vec
                statuses_arr.append(EmbeddingStatus.CONTROL)
                control_symbols.append(c.label)
                continue
            genes = gene_targets.get(i, [])
            vecs: list[np.ndarray] = []
            missing: list[str] = []
            for g in genes:
                idx_row = indexer.symbol_to_index.get(g, 0)
                if idx_row == 0:
                    missing.append(g)
                    continue
                row = table[idx_row]
                if not np.any(row):
                    # Row is in the table but all-zero -- treat as
                    # missing for status purposes.
                    missing.append(g)
                    continue
                vecs.append(np.asarray(row, dtype=np.float32))
            if vecs:
                out[i] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
                if missing:
                    unresolved_symbols.extend(missing)
                statuses_arr.append(EmbeddingStatus.RESOLVED)
            else:
                out[i] = make_unresolved_vector(dim)
                unresolved_symbols.append(c.label)
                statuses_arr.append(EmbeddingStatus.UNRESOLVED)
        self._last_unresolved = unresolved_symbols
        self._last_controls = control_symbols
        self._last_n_unresolved = sum(
            1 for s in statuses_arr if s == EmbeddingStatus.UNRESOLVED
        )
        return out, self._status_array(statuses_arr)

    def build_table(self, symbols: Sequence[str]) -> tuple[np.ndarray, GeneIndexer]:
        # Skip the default round-trip: the legacy loader builds a
        # (table, indexer) of the right shape directly. We still
        # compute control / unresolved counts so metadata is honest.
        table, indexer = load_gene_embedding_table(self._path, symbols=symbols)
        self._embedding_dim = int(table.shape[1])
        unresolved = 0
        unresolved_symbols: list[str] = []
        for sym, idx in indexer.symbol_to_index.items():
            if idx == 0:
                continue
            if not np.any(table[idx]):
                unresolved += 1
                unresolved_symbols.append(sym)
        self._last_n_unresolved = unresolved
        self._last_unresolved = unresolved_symbols
        return table, indexer

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        n_missing = len(self._last_unresolved) or int(n_unresolved or self._last_n_unresolved)
        n_resolved = max(int(n_symbols) - n_missing - len(self._last_controls), 0)
        return ProviderMetadata(
            source="precomputed",
            embedding_dim=self.embedding_dim,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved or self._last_n_unresolved),
            model_name=str(self._path),
            n_resolved=n_resolved,
            n_control=len(self._last_controls),
            unresolved_symbols=list(self._last_unresolved),
            control_symbols=list(self._last_controls),
            control_sentinel_seed=int(self.control_sentinel_seed),
        )


__all__ = ["PrecomputedProvider"]
