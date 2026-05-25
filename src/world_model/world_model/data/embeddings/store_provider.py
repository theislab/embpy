"""Action-embedding provider backed by an embpy ``EmbeddingStore`` (``.emstore``).

This is the world model's canonical action-embedding source after the embpy
migration: it replaces the legacy CSV/NPZ :class:`PrecomputedProvider`. It reads
a ``.emstore`` directory (or wraps a live :class:`~embpy.store.EmbeddingStore`),
looks up gene-symbol embeddings in one of the store's
:class:`~embpy.store.EmbeddingBlock` s, and speaks the same status-aware
contract (``RESOLVED`` / ``CONTROL`` / ``UNRESOLVED``) as the other backends.

Memory-mapped reads (``backed=True``, the default) keep large gene universes
-- e.g. GenePT's ~45k genes x 3072 dims -- cheap to open: only the rows for the
requested perturbation symbols are touched.

Migrate a legacy CSV/NPZ gene-embedding table into a ``.emstore`` with
:func:`embpy.store.migrate_table_to_emstore` (or ``python -m embpy.store.migrate``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from embpy.resources.gene.control import ControlPolicy

from .provider import ActionEmbeddingProvider, ProviderMetadata
from .sentinel import (
    CONTROL_SENTINEL_SEED,
    EmbeddingStatus,
    make_control_vector,
    make_unresolved_vector,
)

if TYPE_CHECKING:
    from embpy.store import EmbeddingBlock, EmbeddingStore

logger = logging.getLogger(__name__)


class StoreProvider(ActionEmbeddingProvider):
    """Look up gene embeddings in an embpy :class:`EmbeddingStore`.

    Parameters
    ----------
    store_path
        Path to a ``.emstore`` directory. Mutually exclusive with ``store``.
    store
        A live :class:`~embpy.store.EmbeddingStore` (used in tests / headless
        callers that already hold one). Mutually exclusive with ``store_path``.
    store_key
        Embedding key within the store to resolve against. When omitted, the
        sole embedding is used, or the single ``entity_type == "gene"`` block.
    backed
        Open on-disk matrices with NumPy memory mapping (default ``True``).
    control_policy
        Classifier deciding which labels are controls. Defaults to the curated
        :meth:`ControlPolicy.default`.
    control_sentinel_seed
        Seed for the deterministic CONTROL sentinel vector.
    """

    def __init__(
        self,
        store_path: str | Path | None = None,
        *,
        store: EmbeddingStore | None = None,
        store_key: str | None = None,
        backed: bool = True,
        control_policy: ControlPolicy | None = None,
        control_sentinel_seed: int = CONTROL_SENTINEL_SEED,
    ) -> None:
        if store is None and store_path is None:
            raise ValueError("StoreProvider needs store_path= or store=.")
        if store is not None and store_path is not None:
            raise ValueError("StoreProvider: pass store_path= or store=, not both.")
        self._store_path = Path(store_path) if store_path is not None else None
        self._store: EmbeddingStore | None = store
        self._store_key: str | None = store_key
        self._backed = bool(backed)
        self._block: EmbeddingBlock | None = None
        self._lookup: dict[str, int] | None = None
        self._embedding_dim: int | None = None
        self._last_unresolved: list[str] = []
        self._last_controls: list[str] = []
        self._last_mixed: list[str] = []
        self.control_policy: ControlPolicy = control_policy or ControlPolicy.default()
        self.control_sentinel_seed: int = int(control_sentinel_seed)

    # ------------------------------------------------------------------
    # ActionEmbeddingProvider API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for metadata / logs (e.g. ``store:gene:genept``)."""
        return f"store:{self._store_key}" if self._store_key else "store"

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality (loads the store block on first access)."""
        if self._embedding_dim is None:
            self._load_block()
        return int(self._embedding_dim)  # type: ignore[arg-type]

    def embed_with_status(
        self,
        symbols: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve ``symbols`` against the store block; see the module docstring."""
        symbols = list(symbols)
        block = self._load_block()
        dim = int(self._embedding_dim)  # type: ignore[arg-type]
        matrix = block.matrix
        lookup = self._lookup or {}
        if not symbols:
            return np.zeros((0, dim), dtype=np.float32), np.asarray([], dtype=object)

        classifications = [self.control_policy.classify(s) for s in symbols]
        out = np.zeros((len(symbols), dim), dtype=np.float32)
        statuses: list[EmbeddingStatus] = []
        unresolved_symbols: list[str] = []
        control_symbols: list[str] = []
        mixed_symbols: list[str] = []
        control_vec = make_control_vector(dim, seed=self.control_sentinel_seed)

        for i, c in enumerate(classifications):
            if c.kind == "control":
                out[i] = control_vec
                statuses.append(EmbeddingStatus.CONTROL)
                control_symbols.append(c.label)
                continue
            if c.kind == "mixed":
                mixed_symbols.append(c.label)
            genes = list(c.gene_components) if c.kind == "mixed" else list(c.components)
            vecs: list[np.ndarray] = []
            missing_components: list[str] = []
            for g in genes:
                idx = lookup.get(g)
                if idx is None:
                    missing_components.append(g)
                    continue
                vecs.append(np.asarray(matrix[idx], dtype=np.float32))
            if vecs:
                out[i] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
                statuses.append(EmbeddingStatus.RESOLVED)
                if missing_components:
                    unresolved_symbols.extend(missing_components)
                    logger.warning(
                        "StoreProvider: label %r partially unresolved: %d/%d components "
                        "missing from store (%s). Row is the mean of resolved components.",
                        c.label,
                        len(missing_components),
                        len(genes),
                        missing_components,
                    )
            else:
                out[i] = make_unresolved_vector(dim)
                statuses.append(EmbeddingStatus.UNRESOLVED)
                unresolved_symbols.append(c.label)

        self._last_unresolved = unresolved_symbols
        self._last_controls = control_symbols
        self._last_mixed = mixed_symbols
        if unresolved_symbols:
            logger.warning(
                "StoreProvider: %d / %d input rows are UNRESOLVED against store key %r "
                "(first 10: %s). UNRESOLVED rows are zero vectors -- treat this as a "
                "data-quality bug, not a default.",
                len(unresolved_symbols),
                len(symbols),
                self._store_key,
                unresolved_symbols[:10],
            )
            logger.debug("Full UNRESOLVED list: %s", unresolved_symbols)
        if control_symbols:
            logger.info(
                "StoreProvider: %d / %d input rows mapped to CONTROL sentinel (seed=%d, dim=%d).",
                len(control_symbols),
                len(symbols),
                self.control_sentinel_seed,
                dim,
            )
        return out, self._status_array(statuses)

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        """Provider metadata for ``action_embedding_meta.json`` (source ``store``)."""
        prov = dict(self._block.provenance) if self._block is not None else {}
        extra = prov.get("extra") if isinstance(prov.get("extra"), dict) else {}
        n_resolved = max(
            int(n_symbols) - len(self._last_unresolved) - len(self._last_controls),
            0,
        )
        return ProviderMetadata(
            source="store",
            embedding_dim=self.embedding_dim,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved or len(self._last_unresolved)),
            model_name=str(prov.get("model") or self._store_key or self._store_path or "store"),
            pooling_strategy=prov.get("pooling"),
            organism=(extra or {}).get("organism"),
            cache_path=str(self._store_path) if self._store_path is not None else None,
            extras={
                "store_key": self._store_key,
                "store_path": str(self._store_path) if self._store_path is not None else None,
                "control_policy_patterns": list(self.control_policy.patterns),
                "control_policy_extra_labels": list(self.control_policy.extra_labels),
            },
            n_resolved=n_resolved,
            n_control=len(self._last_controls),
            n_mixed=len(self._last_mixed),
            unresolved_symbols=list(self._last_unresolved),
            control_symbols=list(self._last_controls),
            mixed_symbols=list(self._last_mixed),
            control_sentinel_seed=int(self.control_sentinel_seed),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_block(self) -> EmbeddingBlock:
        if self._block is not None:
            return self._block
        store = self._store
        if store is None:
            from embpy.store import EmbeddingStore  # lazy import: pulls anndata

            assert self._store_path is not None
            if not self._store_path.exists():
                raise FileNotFoundError(f"StoreProvider: .emstore not found: {self._store_path}")
            store = EmbeddingStore.read(self._store_path, backed=self._backed)
            self._store = store
        block = self._select_block(store)
        self._block = block
        self._embedding_dim = int(block.n_dims)
        self._lookup = {eid: i for i, eid in enumerate(block.entity_ids)}
        logger.info(
            "StoreProvider: loaded block %r (%d entities, dim=%d) from %s",
            block.key,
            block.n_entities,
            block.n_dims,
            self._store_path if self._store_path is not None else "<in-memory store>",
        )
        return block

    def _select_block(self, store: EmbeddingStore) -> EmbeddingBlock:
        if self._store_key is not None:
            return store.embedding(self._store_key)
        keys = store.keys()
        if len(keys) == 1:
            self._store_key = keys[0]
            return store.embedding(keys[0])
        gene_keys = [k for k in keys if store.embedding(k).entity_type == "gene"]
        if len(gene_keys) == 1:
            self._store_key = gene_keys[0]
            return store.embedding(gene_keys[0])
        raise ValueError(f"StoreProvider: store has multiple embeddings {keys}; pass store_key= to pick one.")


__all__ = ["StoreProvider"]
