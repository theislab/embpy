"""Reusable neural building blocks for ``embpy.world_model``.

Kept intentionally small and dependency-light: these are the primitives
that latent-dynamics models in this subpackage compose. Pulling them out
of :mod:`latent_dynamics` makes them easy to share with future flow-
matching or stochastic variants.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    """Simple multi-layer perceptron with optional layer norm and dropout.

    Parameters
    ----------
    in_dim
        Input feature dimension.
    hidden_dims
        Sequence of hidden layer widths. Empty sequence yields a linear map.
    out_dim
        Output feature dimension.
    activation
        Activation factory (callable returning an ``nn.Module``).
        Defaults to :class:`torch.nn.SiLU`.
    dropout
        Dropout probability applied after each hidden activation.
        Set to ``0`` to disable.
    layer_norm
        If ``True``, apply :class:`torch.nn.LayerNorm` before each
        activation in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioning layer.

    Maps a conditioning vector ``c`` to per-feature scale ``gamma`` and
    shift ``beta`` and applies ``gamma * x + beta``.

    References
    ----------
    Perez et al., *FiLM: Visual Reasoning with a General Conditioning
    Layer*, AAAI 2018.
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class ResidualBlock(nn.Module):
    """Pre-norm residual block ``x -> x + MLP(LN(x), cond)``.

    Conditioning is injected with FiLM after the layer-norm. If
    ``cond_dim == 0`` the block falls back to an unconditional residual
    MLP.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int = 0,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.cond_dim = cond_dim
        self.film: FiLM | None = FiLM(cond_dim, dim) if cond_dim > 0 else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        if self.film is not None:
            if cond is None:
                raise ValueError(
                    "ResidualBlock was built with cond_dim > 0 but received cond=None."
                )
            h = self.film(h, cond)
        return x + self.mlp(h)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar time/dose input.

    Useful for flow-matching or diffusion-style world models where the
    velocity field is conditioned on continuous ``t in [0, 1]``.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


__all__ = ["FiLM", "MLP", "ResidualBlock", "SinusoidalTimeEmbedding"]
