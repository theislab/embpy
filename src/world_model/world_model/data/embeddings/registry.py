"""Factory mapping :class:`ActionEmbeddingConfig` to a concrete provider.

This is the single point that decides which backend is used. Adding a
new backend is a four-line patch here plus a new file under
``data/embeddings/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from embpy.resources.gene.control import ControlPolicy

from .anndata_obsm import AnnDataObsmProvider
from .provider import ActionEmbeddingProvider

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

    Training has a single input contract: perturbation embeddings must
    already live in ``adata.obsm[cfg.obsm_key]`` in the same AnnData
    that provides the state embeddings. CSV / NPZ tables and BioEmbedder
    models remain available only as offline attachment sources in
    ``world_model.scripts.embed_perturbations``.
    """
    src = cfg.source
    if src == "anndata_obsm":
        if cfg.h5ad_path:
            data_path = data_cfg.h5ad_path if data_cfg is not None else ""
            if not data_path or str(cfg.h5ad_path) != str(data_path):
                raise ValueError(
                    "action_embedding.h5ad_path overrides are no longer supported. "
                    "Training reads action embeddings from data.h5ad_path so state "
                    "and action matrices are indexed against the same AnnData."
                )
        h5ad_path = data_cfg.h5ad_path if data_cfg is not None else ""
        perturbation_key = cfg.perturbation_key or (
            data_cfg.perturbation_key if data_cfg is not None else "perturbation"
        )
        if not h5ad_path:
            raise ValueError(
                "action_embedding.source='anndata_obsm' requires action_embedding.h5ad_path "
                "or data.h5ad_path."
            )
        if not cfg.obsm_key:
            raise ValueError(
                "action_embedding.source='anndata_obsm' requires action_embedding.obsm_key."
            )
        return AnnDataObsmProvider(
            h5ad_path=h5ad_path,
            obsm_key=cfg.obsm_key,
            perturbation_key=perturbation_key,
            control_policy=_policy_from_cfg(cfg),
            control_sentinel_seed=int(getattr(cfg, "control_sentinel_seed", 0) or 0),
        )

    if src == "store":
        raise ValueError(
            "action_embedding.source='store' / .emstore has been removed from the "
            "world-model training path. Attach perturbation embeddings to AnnData "
            "with `python -m world_model.scripts.embed_perturbations --output-h5ad ...` "
            "and use action_embedding.source='anndata_obsm'."
        )

    if src in {"precomputed", "bio_embedder"}:
        raise ValueError(
            f"action_embedding.source={src!r} is not a training input anymore. "
            "Attach embeddings to AnnData first with "
            "`python -m world_model.scripts.embed_perturbations --output-h5ad ...` "
            "and train with action_embedding.source='anndata_obsm'."
        )

    raise ValueError(
        f"Unknown action_embedding.source={src!r}. Supported: "
        "'anndata_obsm'."
    )


__all__ = ["build_provider"]
