"""Deterministic latent-dynamics world model.

The :class:`LatentDynamicsWorldModel` predicts the perturbation-induced
shift ``delta`` in latent space and adds it to the basal latent::

    z_pred = z_basal + delta_theta(z_basal, cond)

The residual parameterisation matches the inductive bias of typical
perturbation-response problems where the perturbed state is a small
displacement of the unperturbed one. ``delta_theta`` is a stack of
FiLM-conditioned residual MLP blocks (see :mod:`embpy.world_model.modules`).

This is the recommended starting point. For stochastic or flow-matching
variants subclass :class:`embpy.world_model.base.BaseWorldModel` directly.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from .base import BaseWorldModel
from .modules import MLP, ResidualBlock

logger = logging.getLogger(__name__)


class LatentDynamicsWorldModel(BaseWorldModel):
    """Residual latent-dynamics model with FiLM conditioning.

    Parameters
    ----------
    latent_dim
        Dimensionality of the cell-state latent.
    cond_dim
        Dimensionality of the perturbation conditioning vector. Set to
        ``0`` for an unconditional model (identity by default).
    n_blocks
        Number of residual blocks in the dynamics network.
    hidden_mult
        Hidden width of each residual block, expressed as a multiple of
        ``latent_dim``.
    cond_hidden_dims
        Sequence of hidden widths for the conditioning encoder MLP, which
        maps ``cond -> R^{cond_embed_dim}`` before FiLM. If ``None``, the
        conditioning vector is used directly (a single linear projection
        to ``latent_dim`` is still applied).
    cond_embed_dim
        Output dimension of the conditioning encoder. Defaults to
        ``latent_dim``.
    dropout
        Dropout probability inside each residual block.
    delta_scale
        Multiplicative factor on the predicted shift before it is added
        back to ``z``. A small value (e.g. ``0.1``) initialises the model
        close to the identity, which can help early-training stability.
    """

    name = "latent_dynamics"

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        n_blocks: int = 4,
        hidden_mult: int = 4,
        cond_hidden_dims: tuple[int, ...] | None = (256, 256),
        cond_embed_dim: int | None = None,
        dropout: float = 0.0,
        delta_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            n_blocks=n_blocks,
            hidden_mult=hidden_mult,
            cond_hidden_dims=tuple(cond_hidden_dims) if cond_hidden_dims is not None else None,
            cond_embed_dim=cond_embed_dim,
            dropout=dropout,
            delta_scale=delta_scale,
            **kwargs,
        )

        self.delta_scale = float(delta_scale)
        self._effective_cond_dim = int(cond_embed_dim or latent_dim) if cond_dim > 0 else 0

        if cond_dim > 0:
            self.cond_encoder: nn.Module = MLP(
                in_dim=cond_dim,
                hidden_dims=tuple(cond_hidden_dims) if cond_hidden_dims is not None else (),
                out_dim=self._effective_cond_dim,
                dropout=dropout,
                layer_norm=True,
            )
        else:
            self.cond_encoder = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=latent_dim,
                    cond_dim=self._effective_cond_dim,
                    hidden_mult=hidden_mult,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_condition(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        """Map a raw conditioning vector to the FiLM input space."""
        if self.cond_dim == 0 or cond is None:
            return None
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"Expected cond with last-dim {self.cond_dim}, got {tuple(cond.shape)}."
            )
        return self.cond_encoder(cond)

    def predict_delta(
        self,
        z: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the predicted latent shift ``delta`` (without adding ``z``).

        Useful for inspection and for parameterisations that want to
        regularise the magnitude of the perturbation effect directly.
        """
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected z with last-dim {self.latent_dim}, got {tuple(z.shape)}."
            )
        c = self.encode_condition(cond)
        h = z
        for block in self.blocks:
            h = block(h, c)
        return self.head(h) * self.delta_scale

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        delta = self.predict_delta(z, cond)
        return z + delta


__all__ = ["LatentDynamicsWorldModel"]
