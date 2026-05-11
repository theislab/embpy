"""Factory that turns a :class:`StateBackboneConfig` into a provider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .local import LocalBackbone
from .provider import StateBackboneProvider
from .stack import StackBackbone
from .state import StateBackbone

if TYPE_CHECKING:
    from ....configs.base import StateBackboneConfig

logger = logging.getLogger(__name__)


def build_backbone(
    cfg: "StateBackboneConfig",
    *,
    n_genes: int | None = None,
    d_model: int | None = None,
    stack_size: int | None = None,
    encoder_kind: str = "transformer",
    encoder_layers: int = 2,
    encoder_heads: int = 4,
    dropout: float = 0.1,
    **_unused: Any,
) -> StateBackboneProvider:
    """Build a provider from ``cfg``.

    Local backbones additionally need the (``n_genes``, ``d_model``,
    ``stack_size``) tuple that the world-model factory already knows
    about. Foreign backbones derive ``embedding_dim`` from their loaded
    weights, so they only need the paths configured in ``cfg``.
    """
    kind = cfg.kind
    if kind == "local":
        if n_genes is None or d_model is None or stack_size is None:
            raise ValueError(
                "build_backbone(kind='local') requires n_genes, d_model, stack_size."
            )
        return LocalBackbone(
            encoder_kind=encoder_kind,
            n_genes=n_genes,
            d_model=d_model,
            stack_size=stack_size,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            dropout=dropout,
        )
    if kind == "state":
        return StateBackbone(
            checkpoint=cfg.state_checkpoint,
            model_folder=cfg.state_model_folder,
            protein_embeddings=cfg.state_protein_embeddings,
            config=cfg.state_config,
            device=cfg.device,
            freeze=cfg.freeze,
            batch_size=cfg.batch_size,
            cache_dir=cfg.cache_dir,
            require_cache_hit=cfg.require_cache_hit,
        )
    if kind == "stack":
        return StackBackbone(
            checkpoint=cfg.stack_checkpoint,
            genelist=cfg.stack_genelist,
            gene_name_col=cfg.stack_gene_name_col,
            device=cfg.device,
            freeze=cfg.freeze,
            batch_size=cfg.batch_size,
            cache_dir=cfg.cache_dir,
            require_cache_hit=cfg.require_cache_hit,
        )
    raise ValueError(f"Unknown state_backbone.kind={kind!r}. Use 'local', 'state' or 'stack'.")


__all__ = ["build_backbone"]
