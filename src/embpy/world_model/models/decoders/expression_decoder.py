"""Decoder mapping state tokens back to gene-expression vectors.

A separate, swappable module so we can experiment with alternative
parameterisations (Gaussian, ZINB, ...) without touching the rest of the
world-model stack.
"""

from __future__ import annotations

import torch
from torch import nn

from ..blocks import MLP


class ExpressionDecoder(nn.Module):
    """Project a state token ``s in R^d`` back to gene space.

    By default returns a single tensor of shape ``(..., n_genes)`` that
    is interpreted as the mean of a Gaussian observation model with
    fixed unit variance (so :func:`F.mse_loss` matches the negative log
    likelihood up to additive constants).

    Set ``predict_log_var=True`` to get an additional output channel
    representing per-gene log-variance, suitable for plugging into
    :func:`world_model.training.losses.gaussian_nll`.
    """

    def __init__(
        self,
        d_model: int,
        n_genes: int,
        hidden_dims: tuple[int, ...] = (512, 1024),
        dropout: float = 0.1,
        predict_log_var: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.predict_log_var = predict_log_var

        out_dim = 2 * n_genes if predict_log_var else n_genes
        self.mlp = MLP(
            in_dim=d_model,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout=dropout,
            layer_norm=True,
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode state tokens.

        Parameters
        ----------
        s
            ``(..., d_model)`` tensor.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            If ``predict_log_var`` is False: ``(..., n_genes)`` mean.
            Otherwise a tuple ``(mu, log_var)`` of identical shape.
        """
        if s.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {s.shape[-1]}")
        out = self.mlp(s)
        if self.predict_log_var:
            mu, log_var = out.chunk(2, dim=-1)
            return mu, log_var
        return out


__all__ = ["ExpressionDecoder"]
