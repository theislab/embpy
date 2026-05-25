"""Factory mapping :class:`ActionEmbeddingConfig` to a concrete provider.

This is the single point that decides which backend is used. Adding a
new backend is a four-line patch here plus a new file under
``data/embeddings/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from embpy.resources.gene.control import ControlPolicy

from .bio_embedder import BioEmbedderProvider
from .provider import ActionEmbeddingProvider
from .store_provider import StoreProvider

if TYPE_CHECKING:
    from world_model.configs import ActionEmbeddingConfig, DataConfig


def _policy_from_cfg(cfg: ActionEmbeddingConfig) -> ControlPolicy:
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
        extras,
        patterns=tuple(patterns_override),
        strict=strict,
    )


def build_provider(
    cfg: ActionEmbeddingConfig,
    *,
    data_cfg: DataConfig | None = None,
) -> ActionEmbeddingProvider:
    """Build an :class:`ActionEmbeddingProvider` from config.

    New runs should use ``source='store'`` and point at a ``.emstore``.
    The old CSV/NPZ provider remains importable for one-shot migrations and
    legacy tests, but the registry no longer routes production configs through
    ``data.gene_embedding_path``. That prevents accidental training on stale
    CSV tables when the store path was forgotten.
    """
    src = cfg.source
    if src == "precomputed":
        legacy_path = cfg.path or (data_cfg.gene_embedding_path if data_cfg is not None else "")
        hint = f" Existing path: {legacy_path!r}." if legacy_path else ""
        raise ValueError(
            "action_embedding.source='precomputed' is retired for world-model runs."
            f"{hint} Convert CSV/NPZ tables once with "
            "`python -m embpy.store.migrate <table> <out.emstore> --model <name>` "
            "and set action_embedding.source='store' plus action_embedding.store_path."
        )

    if src == "store":
        if not cfg.store_path:
            raise ValueError(
                "action_embedding.source='store' requires action_embedding.store_path "
                "(path to a .emstore). Migrate a legacy CSV/NPZ with "
                "`python -m embpy.store.migrate <table> <out.emstore> --model <name>`."
            )
        return StoreProvider(
            store_path=cfg.store_path,
            store_key=cfg.store_key or None,
            control_policy=_policy_from_cfg(cfg),
            control_sentinel_seed=int(getattr(cfg, "control_sentinel_seed", 0) or 0),
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
            control_sentinel_seed=int(getattr(cfg, "control_sentinel_seed", 0) or 0),
        )

    raise ValueError(f"Unknown action_embedding.source={src!r}. Supported: 'store', 'bio_embedder'.")


__all__ = ["build_provider"]
