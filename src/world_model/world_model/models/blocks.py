"""Reusable neural building blocks shared by encoder and dynamics.

Kept dependency-light (pure PyTorch) so that swapping in a third-party
transformer (xformers, flash-attn, ...) only requires touching this
file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Standard feed-forward MLP with optional LayerNorm and dropout.

    Notes
    -----
    The final layer is a plain linear projection (no activation, no norm)
    so this module composes cleanly into residual stacks.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: type[nn.Module] = nn.GELU,
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


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an optional causal mask.

    The mask is built lazily on the device of the first forward call and
    cached up to ``max_sequence_length``. Set ``causal=False`` for
    bidirectional attention (used inside the state-stack encoder).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.max_sequence_length = max_sequence_length

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if causal:
            mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool))
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, L, D)
        b, l, d = x.shape
        qkv = self.qkv_proj(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)

        if self.causal:
            if l > self.max_sequence_length:
                raise ValueError(
                    f"Sequence length {l} exceeds max_sequence_length {self.max_sequence_length}."
                )
            mask = self.causal_mask[:l, :l]  # type: ignore[index]
            scores = scores.masked_fill(~mask, float("-inf"))

        if attn_mask is not None:
            # attn_mask: (B, L) with True for valid positions
            key_mask = attn_mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: ``x = x + Attn(LN(x)); x = x + FF(LN(x))``.

    Parameters
    ----------
    d_model
        Model width.
    n_heads
        Number of attention heads (must divide ``d_model``).
    ff_mult
        Feed-forward expansion factor; hidden width = ``ff_mult * d_model``.
    dropout
        Dropout probability applied inside attention and FF.
    causal
        Whether to mask future positions.
    max_sequence_length
        Upper bound on the sequence length seen at training/inference time.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = True,
        max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            causal=causal,
            max_sequence_length=max_sequence_length,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


__all__ = ["CausalSelfAttention", "MLP", "TransformerBlock"]
