"""Factory mapping :class:`ActionEmbeddingConfig` to a concrete provider.

This is the single point that decides which backend is used. Adding a
new backend is a four-line patch here plus a new file under
``data/embeddings/``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from embpy.resources.gene.control import ControlPolicy

from .bio_embedder import BioEmbedderProvider
from .precomputed import PrecomputedProvider
from .provider import ActionEmbeddingProvider

if TYPE_CHECKING:
    from ...configs import ActionEmbeddingConfig, DataConfig

logger = logging.getLogger(__name__)


def _policy_from_cfg(cfg: "ActionEmbeddingConfig") -> ControlPolicy:
    """Build a :class:`ControlPolicy` from the YAML knobs.

    All three knobs are optional; defaults are the curated regex set.
    Patterns and extras live under the ``action_embedding`` block so a
    dataset can override them per-config without touching code.
    """
    extras = tuple(getattr(cfg, "control_extra_labels", ()) or ())
    patterns_override = getattr(cfg, "control_patterns", None)
    strict = bool(getattr(cfg, "control_strict", False))
    if patterns_override is None or len(patterns_override) == 0:
        return ControlPolicy.from_iterable(extras, strict=strict)
    return ControlPolicy.from_iterable(
        extras, patterns=tuple(patterns_override), strict=strict,
    )


def build_provider(
    cfg: "ActionEmbeddingConfig",
    *,
    data_cfg: "DataConfig | None" = None,
) -> ActionEmbeddingProvider:
    """Build an :class:`ActionEmbeddingProvider` from config.

    The precomputed-fallback rule lives here: when
    ``cfg.source == "precomputed"`` and ``cfg.path`` is empty, we fall
    back to ``data_cfg.gene_embedding_path`` (the legacy single field).
    A one-time ``warning`` log line documents the deprecation.
    """
    src = cfg.source
    if src == "precomputed":
        path = cfg.path
        if not path and data_cfg is not None and data_cfg.gene_embedding_path:
            logger.warning(
                "ActionEmbeddingConfig.path is empty; falling back to "
                "DataConfig.gene_embedding_path=%s. This legacy path is "
                "supported but deprecated -- migrate to the explicit "
                "action_embedding: block in your YAML.",
                data_cfg.gene_embedding_path,
            )
            path = data_cfg.gene_embedding_path
        if not path:
            raise ValueError(
                "Precomputed action embeddings require either "
                "action_embedding.path or data.gene_embedding_path to be set."
            )
        return PrecomputedProvider(
            path=path,
            control_policy=_policy_from_cfg(cfg),
            control_sentinel_seed=int(
                getattr(cfg, "control_sentinel_seed", 0) or 0
            ),
        )

    if src == "bio_embedder":
        return BioEmbedderProvider(
            model_name=cfg.model_name,
            organism=cfg.organism,
            resolver_backend=cfg.resolver_backend,
            mart_file=cfg.mart_file,
            chromosome_folder=cfg.chromosome_folder,
            id_type=cfg.id_type,
            region=cfg.region,
            pooling_strategy=cfg.pooling_strategy,
            device=cfg.device,
            cache_dir=cfg.cache_dir or None,
            extra_kwargs=dict(cfg.extra_kwargs or {}),
            control_policy=_policy_from_cfg(cfg),
            control_sentinel_seed=int(
                getattr(cfg, "control_sentinel_seed", 0) or 0
            ),
        )

    raise ValueError(
        f"Unknown action_embedding.source={src!r}. "
        f"Supported: 'precomputed', 'bio_embedder'."
    )


__all__ = ["build_provider"]
