"""Provider that delegates to :class:`embpy.embedder.BioEmbedder`.

The :class:`BioEmbedder` import is *lazy* -- deferred until
:meth:`_get_embedder` is called -- so machines without the heavy
embedding stack (Boltz, ESM, RDKit) can still use the precomputed
provider without dragging the dependencies in.

A disk-backed cache keyed by ``(model, region, pooling, organism)``
guarantees that re-running a config with the same provider settings
re-uses cached vectors and only invokes the foundation model on
genuinely new symbols.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .cache import EmbeddingCacheKey, load_cached, save_cached
from .provider import ActionEmbeddingProvider, ProviderMetadata

logger = logging.getLogger(__name__)


class BioEmbedderProvider(ActionEmbeddingProvider):
    """Look up gene embeddings on demand via :class:`embpy.embedder.BioEmbedder`."""

    def __init__(
        self,
        model_name: str,
        *,
        organism: str = "human",
        resolver_backend: Literal["api", "local"] = "api",
        mart_file: str | None = None,
        chromosome_folder: str | None = None,
        id_type: Literal["symbol", "ensembl_id"] = "symbol",
        region: Literal["full", "exons", "introns"] = "full",
        pooling_strategy: str = "mean",
        device: str = "auto",
        cache_dir: str | Path | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.organism = str(organism)
        self.resolver_backend = resolver_backend
        self.mart_file = mart_file
        self.chromosome_folder = chromosome_folder
        self.id_type = id_type
        self.region = region
        self.pooling_strategy = pooling_strategy
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.extra_kwargs: dict[str, Any] = dict(extra_kwargs or {})

        self._embedder: Any | None = None
        self._embedding_dim: int | None = None
        self._last_unresolved: list[str] = []

    # ------------------------------------------------------------------
    # ActionEmbeddingProvider API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError(
                "BioEmbedderProvider.embedding_dim is only known after the first "
                "call to embed() / build_table(). Probe with a single dummy symbol "
                "first if you need the dim ahead of time."
            )
        return int(self._embedding_dim)

    @property
    def cache_key(self) -> EmbeddingCacheKey:
        return EmbeddingCacheKey(
            model_name=self.model_name,
            region=self.region,
            pooling_strategy=self.pooling_strategy,
            organism=self.organism,
        )

    def embed(self, symbols: Sequence[str]) -> np.ndarray:
        symbols = list(symbols)
        if not symbols:
            return np.zeros((0, 0), dtype=np.float32)

        cache_dir = self.cache_dir
        cached: dict[str, np.ndarray] = {}
        if cache_dir is not None:
            cached = load_cached(cache_dir, self.cache_key, symbols)
            logger.info(
                "BioEmbedderProvider cache hit %d/%d symbols (model=%s)",
                len(cached), len(symbols), self.model_name,
            )

        missing = [s for s in symbols if s not in cached]
        new_vecs: dict[str, np.ndarray] = {}
        if missing:
            new_vecs = self._compute(missing)
            if cache_dir is not None and new_vecs:
                ordered_syms = list(new_vecs.keys())
                ordered_emb = np.stack(
                    [new_vecs[s] for s in ordered_syms], axis=0,
                ).astype(np.float32)
                save_cached(cache_dir, self.cache_key, ordered_syms, ordered_emb)

        # Combine the two sources, preserving the requested order.
        first: np.ndarray | None = None
        for source in (cached, new_vecs):
            for v in source.values():
                first = v
                break
            if first is not None:
                break
        if first is None:
            raise ValueError(
                f"BioEmbedder returned no embeddings for any of {len(symbols)} symbols. "
                f"Check model_name={self.model_name!r}, organism={self.organism!r}, "
                f"resolver_backend={self.resolver_backend!r}."
            )
        dim = int(first.shape[0])
        self._embedding_dim = dim
        out = np.zeros((len(symbols), dim), dtype=np.float32)
        unresolved: list[str] = []
        for i, sym in enumerate(symbols):
            # Explicit `is not None` checks: numpy arrays do not have a
            # well-defined boolean truth value.
            vec: np.ndarray | None
            if sym in cached:
                vec = cached[sym]
            elif sym in new_vecs:
                vec = new_vecs[sym]
            else:
                vec = None
            if vec is None:
                unresolved.append(sym)
                continue
            out[i] = np.asarray(vec, dtype=np.float32)
        self._last_unresolved = unresolved
        if unresolved:
            logger.warning(
                "%d / %d symbols unresolvable for model=%s; rows zeroed.",
                len(unresolved), len(symbols), self.model_name,
            )
            logger.debug("Unresolved symbols: %s", unresolved)
        return out

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        cache_path: str | None = None
        if self.cache_dir is not None:
            cache_path = str(self.cache_dir / self.cache_key.relative_path())
        return ProviderMetadata(
            source="bio_embedder",
            embedding_dim=self.embedding_dim if self._embedding_dim is not None else 0,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved or len(self._last_unresolved)),
            model_name=self.model_name,
            region=self.region,
            pooling_strategy=self.pooling_strategy,
            organism=self.organism,
            cache_path=cache_path,
            extras={
                "resolver_backend": self.resolver_backend,
                "id_type": self.id_type,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_embedder(self):  # type: ignore[no-untyped-def]
        if self._embedder is not None:
            return self._embedder
        # Lazy import: keeps the precomputed code path free of the
        # BioEmbedder heavy dependency tree.
        from embpy.embedder import BioEmbedder  # noqa: PLC0415

        self._embedder = BioEmbedder(
            device=self.device,
            organism=self.organism,
            resolver_backend=self.resolver_backend,
            mart_file=self.mart_file,
            chromosome_folder=self.chromosome_folder,
        )
        return self._embedder

    def _compute(self, symbols: Sequence[str]) -> dict[str, np.ndarray]:
        embedder = self._get_embedder()
        results = embedder.embed_genes_batch(
            model=self.model_name,
            identifiers=list(symbols),
            id_type=self.id_type,
            organism=self.organism,
            pooling_strategy=self.pooling_strategy,
            region=self.region,
            **self.extra_kwargs,
        )
        out: dict[str, np.ndarray] = {}
        for sym, vec in zip(symbols, results, strict=False):
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            out[sym] = arr
        return out


__all__ = ["BioEmbedderProvider"]
