"""Adapter factory: maps :class:`ActionAdapterConfig` -> concrete :class:`ActionAdapter`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ActionAdapter
from .linear import LinearAdapter
from .lora import LoRAAdapter
from .mlp import MLPAdapter

if TYPE_CHECKING:
    from ....configs import ActionAdapterConfig


def build_action_adapter(
    d_in: int,
    d_model: int,
    cfg: "ActionAdapterConfig",
) -> ActionAdapter:
    """Construct an :class:`ActionAdapter` from a config.

    The factory is the *only* place that knows the kind enum; the rest
    of the world-model code talks to the abstract :class:`ActionAdapter`.
    """
    kind = cfg.kind
    if kind == "linear":
        return LinearAdapter(d_in=d_in, d_model=d_model)
    if kind == "mlp":
        return MLPAdapter(
            d_in=d_in,
            d_model=d_model,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
    if kind == "lora":
        return LoRAAdapter(
            d_in=d_in,
            d_model=d_model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.dropout,
        )
    raise ValueError(
        f"Unknown action_adapter.kind={kind!r}. Supported: 'linear', 'mlp', 'lora'."
    )


__all__ = ["build_action_adapter"]
