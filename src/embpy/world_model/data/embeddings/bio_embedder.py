"""Provider that delegates to :class:`embpy.embedder.BioEmbedder`.

The :class:`BioEmbedder` import is *lazy* -- deferred until
:meth:`_get_embedder` is called -- so machines without the heavy
embedding stack (Boltz, ESM, RDKit) can still use the precomputed
provider without dragging the dependencies in.

A disk-backed cache keyed by ``(model, region, pooling, organism)``
guarantees that re-running a config with the same provider settings
re-uses cached vectors and only invokes the foundation model on
genuinely new symbols.

Status-aware contract (Part A.2):

* Control / non-targeting labels are detected upstream of the embedder
  via :class:`ControlPolicy` and never sent to BioEmbedder. They map to
  a deterministic non-zero sentinel row.
* Genuinely unresolved gene symbols map to zero rows; every batch with
  unresolved rows logs a structured WARNING + DEBUG with the symbol
  list, and the status array is exposed for downstream metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from embpy.resources.gene.control import ControlPolicy

from .cache import EmbeddingCacheKey, load_cached, save_cached
from .provider import ActionEmbeddingProvider, ProviderMetadata
from .sentinel import (
    CONTROL_SENTINEL_SEED,
    EmbeddingStatus,
    make_control_vector,
    make_unresolved_vector,
)

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
        control_policy: ControlPolicy | None = None,
        control_sentinel_seed: int = CONTROL_SENTINEL_SEED,
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
        self.control_policy: ControlPolicy = control_policy or ControlPolicy.default()
        self.control_sentinel_seed: int = int(control_sentinel_seed)

        self._embedder: Any | None = None
        self._embedding_dim: int | None = None
        self._last_unresolved: list[str] = []
        self._last_controls: list[str] = []
        self._last_mixed: list[str] = []

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
                "call to embed_with_status() / build_table(). Probe with a single "
                "dummy gene symbol first if you need the dim ahead of time."
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

    def embed_with_status(
        self,
        symbols: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Status-aware embedding pass.

        See the module docstring for the contract. The status array is
        ``dtype=object`` carrying ``EmbeddingStatus.value`` strings so it
        survives JSON round-trips and can be stacked alongside the
        embedding matrix.
        """
        symbols = list(symbols)
        if not symbols:
            empty = np.zeros((0, 0), dtype=np.float32)
            return empty, np.asarray([], dtype=object)

        # Step 1: classify every label up-front. Controls and mixed
        # labels are routed away from the embedder.
        classifications = [self.control_policy.classify(s) for s in symbols]
        control_mask = np.asarray(
            [c.kind == "control" for c in classifications], dtype=bool
        )
        mixed_mask = np.asarray(
            [c.kind == "mixed" for c in classifications], dtype=bool
        )
        # For "mixed" labels, only the gene components reach the embedder;
        # we record but otherwise ignore the control parts. The aggregated
        # row for a mixed label is the per-gene mean computed below.
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

        # Step 2: cache hit / miss for the unique flat gene list.
        cached: dict[str, np.ndarray] = {}
        cache_dir = self.cache_dir
        if cache_dir is not None and unique_genes:
            cached = load_cached(cache_dir, self.cache_key, unique_genes)
            logger.info(
                "BioEmbedderProvider cache hit %d/%d unique gene symbols (model=%s)",
                len(cached), len(unique_genes), self.model_name,
            )

        missing = [g for g in unique_genes if g not in cached]
        new_vecs: dict[str, np.ndarray] = {}
        if missing:
            new_vecs = self._compute(missing)
            if cache_dir is not None and new_vecs:
                ordered_syms = list(new_vecs.keys())
                ordered_emb = np.stack(
                    [new_vecs[s] for s in ordered_syms], axis=0,
                ).astype(np.float32)
                save_cached(cache_dir, self.cache_key, ordered_syms, ordered_emb)

        # Step 3: figure out the embedding dimensionality before
        # constructing the output buffer. Probing one resolved vector is
        # enough; if no genes resolved at all we still need a sane fallback.
        first_resolved: np.ndarray | None = None
        for source in (cached, new_vecs):
            for v in source.values():
                first_resolved = v
                break
            if first_resolved is not None:
                break

        # If no genes were embedded but at least one control exists, we
        # cannot infer the dim from the embedder; that's an operational
        # bug that we surface explicitly. We refuse to silently fall back
        # to a zero-dim array because downstream collation would crash
        # at training time with a far less actionable error.
        if first_resolved is None and not control_mask.any():
            raise ValueError(
                f"BioEmbedder returned no embeddings for any of {len(symbols)} "
                f"symbols. Check model_name={self.model_name!r}, "
                f"organism={self.organism!r}, "
                f"resolver_backend={self.resolver_backend!r}. "
                f"If your dataset is mostly controls, set "
                f"action_embedding.fail_on_unresolved=False and verify the "
                f"output of embed_perturbations.py."
            )
        if first_resolved is None:
            # Probe the embedder with one *known-good* gene to learn the
            # dim. We pick "TP53" because every gene embedder in the
            # registry has been smoke-tested on it. If even this fails
            # we surface a clear error.
            probe = self._compute(["TP53"])
            if not probe:
                raise ValueError(
                    "Cannot determine embedding dimensionality: TP53 probe "
                    "failed too. The embedder is misconfigured."
                )
            first_resolved = next(iter(probe.values()))

        dim = int(first_resolved.shape[0])
        self._embedding_dim = dim

        # Step 4: assemble rows + statuses in the order of the input.
        out = np.zeros((len(symbols), dim), dtype=np.float32)
        statuses: list[EmbeddingStatus] = [EmbeddingStatus.RESOLVED] * len(symbols)
        unresolved_symbols: list[str] = []
        control_symbols: list[str] = []
        mixed_symbols: list[str] = []

        control_vec = make_control_vector(dim, seed=self.control_sentinel_seed)
        for i, c in enumerate(classifications):
            if c.kind == "control":
                out[i] = control_vec
                statuses[i] = EmbeddingStatus.CONTROL
                control_symbols.append(c.label)
                continue
            if c.kind == "mixed":
                mixed_symbols.append(c.label)
            genes = gene_targets.get(i, [])
            if not genes:
                # Empty after split / classification -- treat as control
                # so we never emit a silent zero. The dataset author
                # should fix their metadata; we log loudly.
                out[i] = control_vec
                statuses[i] = EmbeddingStatus.CONTROL
                control_symbols.append(c.label)
                logger.warning(
                    "Label %r has no embeddable components after split; "
                    "treating as CONTROL.", c.label,
                )
                continue
            # Collect per-gene vectors. Missing components count as
            # unresolved at the component level. If ALL components fail
            # we mark the whole row UNRESOLVED; if SOME succeed we mean-
            # aggregate the resolved ones (matching the byte-equivalent
            # mean-pool semantics of the GeneEmbeddingAction encoder).
            vecs: list[np.ndarray] = []
            missing_components: list[str] = []
            for g in genes:
                v: np.ndarray | None
                if g in cached:
                    v = cached[g]
                elif g in new_vecs:
                    v = new_vecs[g]
                else:
                    v = None
                if v is None:
                    missing_components.append(g)
                else:
                    vecs.append(np.asarray(v, dtype=np.float32))
            if vecs:
                out[i] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
                if missing_components:
                    # Partial failure: keep the resolved aggregate but
                    # surface the missing pieces so the audit log is
                    # honest about what made it into the row.
                    unresolved_symbols.extend(missing_components)
                    logger.warning(
                        "Label %r is partially unresolved: %d/%d "
                        "components missing (%s). Row is the mean of "
                        "the resolved components.",
                        c.label,
                        len(missing_components),
                        len(genes),
                        missing_components,
                    )
            else:
                out[i] = make_unresolved_vector(dim)
                statuses[i] = EmbeddingStatus.UNRESOLVED
                unresolved_symbols.append(c.label)

        # Step 5: loud failures, never silent.
        self._last_unresolved = unresolved_symbols
        self._last_controls = control_symbols
        self._last_mixed = mixed_symbols
        if unresolved_symbols:
            preview = unresolved_symbols[:10]
            logger.warning(
                "BioEmbedderProvider: %d / %d input rows are UNRESOLVED "
                "for model=%s (first 10: %s). UNRESOLVED rows are zero "
                "vectors -- treat this as a data-quality bug, not a "
                "default. Re-run embed_perturbations.py after fixing "
                "alias drift / Ensembl 4xx errors.",
                len(unresolved_symbols), len(symbols), self.model_name, preview,
            )
            logger.debug("Full UNRESOLVED list: %s", unresolved_symbols)
        if control_symbols:
            logger.info(
                "BioEmbedderProvider: %d / %d input rows mapped to CONTROL "
                "sentinel (seed=%d, dim=%d).",
                len(control_symbols), len(symbols),
                self.control_sentinel_seed, dim,
            )

        return out, self._status_array(statuses)

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        cache_path: str | None = None
        if self.cache_dir is not None:
            cache_path = str(self.cache_dir / self.cache_key.relative_path())
        n_resolved = max(
            int(n_symbols)
            - len(self._last_unresolved)
            - len(self._last_controls),
            0,
        )
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
                "control_policy_patterns": list(self.control_policy.patterns),
                "control_policy_extra_labels": list(
                    self.control_policy.extra_labels
                ),
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
        if not symbols:
            return {}
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
