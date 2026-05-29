"""Action-embedding provider backed by an AnnData ``.obsm`` matrix.

The expected layout is deliberately plain AnnData:

* ``adata.obs[perturbation_key]`` contains the per-cell perturbation label.
* ``adata.obsm[obsm_key]`` contains a per-cell action embedding aligned to
  ``adata.obs_names``.

The provider deduplicates that cell-level matrix into one row per
perturbation label, so the world model can keep using its compact action
table internally while the persistent artifact remains the dataset's
``.h5ad`` file.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from embpy.resources.gene.control import ControlPolicy

from .provider import ActionEmbeddingProvider, ProviderMetadata
from .sentinel import (
    CONTROL_SENTINEL_SEED,
    EmbeddingStatus,
    make_control_vector,
    make_unresolved_vector,
)

logger = logging.getLogger(__name__)


class AnnDataObsmProvider(ActionEmbeddingProvider):
    """Resolve perturbation labels from an AnnData ``.obsm`` matrix."""

    def __init__(
        self,
        h5ad_path: str | Path,
        *,
        obsm_key: str,
        perturbation_key: str = "perturbation",
        control_policy: ControlPolicy | None = None,
        control_sentinel_seed: int = CONTROL_SENTINEL_SEED,
    ) -> None:
        if not h5ad_path:
            raise ValueError("AnnDataObsmProvider requires h5ad_path.")
        if not obsm_key:
            raise ValueError("AnnDataObsmProvider requires obsm_key.")
        self.h5ad_path = Path(h5ad_path)
        self.obsm_key = str(obsm_key)
        self.perturbation_key = str(perturbation_key)
        self.control_policy: ControlPolicy = control_policy or ControlPolicy.default()
        self.control_sentinel_seed = int(control_sentinel_seed)

        self._embedding_dim: int | None = None
        self._vectors: dict[str, np.ndarray] | None = None
        self._last_unresolved: list[str] = []
        self._last_controls: list[str] = []
        self._last_mixed: list[str] = []
        self._meta_from_adata: dict | None = None

    @property
    def name(self) -> str:
        return f"anndata_obsm:{self.obsm_key}"

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load_vectors()
        return int(self._embedding_dim)  # type: ignore[arg-type]

    def embed_with_status(
        self,
        symbols: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        symbols = [str(s) for s in symbols]
        vectors = self._load_vectors()
        dim = self.embedding_dim
        if not symbols:
            return np.zeros((0, dim), dtype=np.float32), np.asarray([], dtype=object)

        out = np.zeros((len(symbols), dim), dtype=np.float32)
        statuses: list[EmbeddingStatus] = []
        unresolved: list[str] = []
        controls: list[str] = []
        mixed: list[str] = []
        control_vec = make_control_vector(dim, seed=self.control_sentinel_seed)

        for i, symbol in enumerate(symbols):
            classification = self.control_policy.classify(symbol)
            if classification.kind == "control":
                out[i] = control_vec
                statuses.append(EmbeddingStatus.CONTROL)
                controls.append(symbol)
                continue
            if classification.kind == "mixed":
                mixed.append(symbol)

            vec = vectors.get(symbol)
            if vec is None and classification.kind == "mixed":
                # If a user attached component-level embeddings but not a
                # literal combo label, mean-pool the component vectors. The
                # common Replogle/Nadig path stores one vector per label, so
                # this is just a tolerant fallback.
                parts = [p for p in classification.gene_components if p in vectors]
                if parts:
                    vec = np.mean(np.stack([vectors[p] for p in parts], axis=0), axis=0)

            if vec is None or not np.isfinite(vec).all() or not np.any(vec):
                out[i] = make_unresolved_vector(dim)
                statuses.append(EmbeddingStatus.UNRESOLVED)
                unresolved.append(symbol)
            else:
                out[i] = np.asarray(vec, dtype=np.float32)
                statuses.append(EmbeddingStatus.RESOLVED)

        self._last_unresolved = unresolved
        self._last_controls = controls
        self._last_mixed = mixed
        if unresolved:
            logger.warning(
                "AnnDataObsmProvider: %d / %d perturbation label(s) missing or zero in %s[%r] "
                "(first 10: %s). UNRESOLVED rows are zero vectors.",
                len(unresolved),
                len(symbols),
                self.h5ad_path,
                self.obsm_key,
                unresolved[:10],
            )
        if controls:
            logger.info(
                "AnnDataObsmProvider: %d / %d label(s) mapped to CONTROL sentinel (seed=%d, dim=%d).",
                len(controls),
                len(symbols),
                self.control_sentinel_seed,
                dim,
            )
        return out, self._status_array(statuses)

    def metadata(self, n_symbols: int, n_unresolved: int) -> ProviderMetadata:
        model_name = self.obsm_key
        extras: dict = {
            "h5ad_path": str(self.h5ad_path),
            "obsm_key": self.obsm_key,
            "perturbation_key": self.perturbation_key,
            "control_policy_patterns": list(self.control_policy.patterns),
            "control_policy_extra_labels": list(self.control_policy.extra_labels),
        }
        if self._meta_from_adata:
            model_name = str(self._meta_from_adata.get("model_name") or model_name)
            extras["anndata_uns"] = self._meta_from_adata
        n_resolved = max(
            int(n_symbols) - len(self._last_unresolved) - len(self._last_controls),
            0,
        )
        return ProviderMetadata(
            source="anndata_obsm",
            embedding_dim=self.embedding_dim,
            n_symbols=int(n_symbols),
            n_unresolved=int(n_unresolved or len(self._last_unresolved)),
            model_name=model_name,
            cache_path=str(self.h5ad_path),
            extras=extras,
            n_resolved=n_resolved,
            n_control=len(self._last_controls),
            n_mixed=len(self._last_mixed),
            unresolved_symbols=list(self._last_unresolved),
            control_symbols=list(self._last_controls),
            mixed_symbols=list(self._last_mixed),
            control_sentinel_seed=int(self.control_sentinel_seed),
        )

    def _load_vectors(self) -> dict[str, np.ndarray]:
        if self._vectors is not None:
            return self._vectors
        if not self.h5ad_path.exists():
            raise FileNotFoundError(f"AnnData action embedding file not found: {self.h5ad_path}")

        import anndata as ad  # noqa: PLC0415

        adata = ad.read_h5ad(self.h5ad_path)
        if self.perturbation_key not in adata.obs.columns:
            raise KeyError(
                f"{self.perturbation_key!r} not in adata.obs for {self.h5ad_path} "
                f"(available: {list(adata.obs.columns)})"
            )
        if self.obsm_key not in adata.obsm:
            raise KeyError(
                f"{self.obsm_key!r} not in adata.obsm for {self.h5ad_path} "
                f"(available: {list(adata.obsm.keys())})"
            )

        matrix = adata.obsm[self.obsm_key]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"adata.obsm[{self.obsm_key!r}] must be 2D, got {matrix.shape!r}.")
        if matrix.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{self.obsm_key!r}] has {matrix.shape[0]} rows but adata has {adata.n_obs} obs."
            )

        labels = adata.obs[self.perturbation_key].astype(str).to_numpy()
        vectors: dict[str, np.ndarray] = {}
        for label in dict.fromkeys(labels):
            idx = np.flatnonzero(labels == label)
            rows = np.asarray(matrix[idx], dtype=np.float32)
            finite_rows = rows[np.isfinite(rows).all(axis=1)]
            if finite_rows.size == 0:
                vectors[str(label)] = np.zeros(matrix.shape[1], dtype=np.float32)
                continue
            if finite_rows.shape[0] > 1:
                spread = np.nanmax(finite_rows, axis=0) - np.nanmin(finite_rows, axis=0)
                if bool(np.any(spread > 1e-5)):
                    logger.warning(
                        "AnnDataObsmProvider: perturbation %r has non-identical rows in obsm[%r]; "
                        "using the per-label mean.",
                        label,
                        self.obsm_key,
                    )
            vectors[str(label)] = finite_rows.mean(axis=0).astype(np.float32)

        self._embedding_dim = int(matrix.shape[1])
        self._vectors = vectors
        meta_root = adata.uns.get("world_model_action_embeddings", {})
        if isinstance(meta_root, dict):
            meta = meta_root.get(self.obsm_key)
            self._meta_from_adata = dict(meta) if isinstance(meta, dict) else None
        logger.info(
            "AnnDataObsmProvider: loaded %d perturbation labels from %s obsm[%r] (dim=%d).",
            len(vectors),
            self.h5ad_path,
            self.obsm_key,
            self._embedding_dim,
        )
        return vectors


__all__ = ["AnnDataObsmProvider"]
