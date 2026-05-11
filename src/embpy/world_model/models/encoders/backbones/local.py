"""Local provider that wraps the existing :class:`StateStackEncoder`.

This is the default backbone. It introduces zero new dependencies and
is *bit-equivalent* to the pre-Phase-5 path:

* ``encode`` is intentionally *not* the foundation-style cell-embedding
  encode -- the local backbone's job in the world-model pipeline is to
  expose the underlying :class:`StateStackEncoder` as a torch module
  the dataloader does *not* pre-encode through. The dataloader stays
  on its raw-expression code path. ``encode()`` exists only so the
  provider abstraction is total; calling it produces the encoder's
  ``(B, T, K, G) -> (B, T, d_model)`` transformation on a torch input.

The CLI smoke target only uses ``encode`` for the *foreign* backbones
(STATE / STACK). Tests pin both the parity-with-old-behavior contract
and the local encoder's torch forward.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from ..state_stack_encoder import StateStackEncoder, build_state_stack_encoder
from .provider import StateBackboneProvider

logger = logging.getLogger(__name__)


class LocalBackbone(StateBackboneProvider):
    """Provider that *is* the existing state-stack encoder.

    Backwards-compatible alias :class:`LocalStackBackbone` is re-exported
    from this module so old import paths keep working.
    """

    name: str = "local"
    supports_decode: bool = False

    def __init__(
        self,
        *,
        encoder: StateStackEncoder | None = None,
        encoder_kind: str = "transformer",
        n_genes: int | None = None,
        d_model: int | None = None,
        stack_size: int | None = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if encoder is None:
            if n_genes is None or d_model is None or stack_size is None:
                raise ValueError(
                    "LocalStackBackbone needs either a prebuilt encoder or "
                    "(n_genes, d_model, stack_size) to build one."
                )
            encoder = build_state_stack_encoder(
                kind=encoder_kind,
                n_genes=n_genes,
                d_model=d_model,
                stack_size=stack_size,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
            )
        self._encoder = encoder

    @property
    def encoder(self) -> StateStackEncoder:
        return self._encoder

    @property
    def embedding_dim(self) -> int:
        return int(self._encoder.d_model)

    @property
    def n_genes(self) -> int:
        return int(self._encoder.n_genes)

    @property
    def stack_size(self) -> int:
        return int(self._encoder.stack_size)

    def encode(self, adata: Any, *, batch_size: int | None = None) -> np.ndarray:
        del batch_size
        if isinstance(adata, torch.Tensor):
            with torch.no_grad():
                out = self._encoder(adata)
            return out.detach().cpu().numpy()
        raise TypeError(
            "LocalStackBackbone.encode expects a torch.Tensor of shape "
            "(B, T, K, G). For AnnData encoding use a foreign backbone "
            "(kind='state' or 'stack')."
        )

    def freeze(self) -> None:
        n = 0
        for p in self._encoder.parameters():
            p.requires_grad_(False)
            n += p.numel()
        logger.info("LocalBackbone froze %d parameters.", n)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return iter(())

    def train_mode(self, flag: bool) -> None:
        del flag


LocalStackBackbone = LocalBackbone


__all__ = ["LocalBackbone", "LocalStackBackbone"]
