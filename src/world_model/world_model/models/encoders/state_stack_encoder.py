"""State-stack encoder for transcriptomics observations.

Replaces the ResNet image-stack encoder used in the reference paper
(`Learning World Models for Unconstrained Goal Navigation
<https://arxiv.org/pdf/2405.18193>`_) with a transcriptomics-aware
analogue: each timestep stacks ``K`` gene-expression vectors (real
replicates or temporally-adjacent measurements) and the encoder
aggregates them into a single state token.

Why frame stacking carries over to transcriptomics
--------------------------------------------------
* scRNA-seq counts are extremely noisy and zero-inflated. Stacking
  multiple cells / timepoints averages out per-cell sampling noise the
  same way frame stacking averages out per-frame motion blur in Atari.
* The downstream dynamics model is autoregressive, so we expose a
  ``stack_size`` window of past observations to give the dynamics
  short-horizon temporal context without bloating the sequence length.
* The encoder must remain permutation-aware along the stack dimension
  (replicates have no canonical order). The transformer variant achieves
  this with a learned per-stack position embedding (which an ablation
  can disable via ``use_stack_position_embedding=False``).

Two implementations are provided:

* :class:`TransformerStateStackEncoder` -- per-frame linear projection
  followed by a small transformer encoder + CLS pooling. This is the
  default and the closest analogue of the paper's ResNet trunk.
* :class:`MLPStateStackEncoder` -- flatten stack and run a plain MLP.
  Cheap baseline / ablation.

Both subclass :class:`StateStackEncoder` and expose the same input/output
contract:

* Input  ``x``: ``(B, T, K, G)``
* Output ``s``: ``(B, T, d_model)``
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..blocks import TransformerBlock


class StateStackEncoder(nn.Module, ABC):
    """Common interface for state-stack encoders.

    Subclasses must accept inputs of shape ``(B, T, K, G)`` and return
    state tokens of shape ``(B, T, d_model)``.
    """

    def __init__(self, n_genes: int, d_model: int, stack_size: int) -> None:
        super().__init__()
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if stack_size <= 0:
            raise ValueError(f"stack_size must be positive, got {stack_size}")
        self.n_genes = n_genes
        self.d_model = d_model
        self.stack_size = stack_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a stack of observations.

        Parameters
        ----------
        x
            Tensor of shape ``(B, T, K, G)``.

        Returns
        -------
        torch.Tensor
            State tokens of shape ``(B, T, d_model)``.
        """


class TransformerStateStackEncoder(StateStackEncoder):
    """Transformer encoder over the K-frame stack with CLS-token readout.

    Parameters
    ----------
    n_genes
        Dimensionality G of each gene-expression vector.
    d_model
        Token width.
    stack_size
        K -- number of frames stacked at each timestep.
    n_layers
        Number of transformer layers in the intra-stack encoder.
    n_heads
        Number of attention heads.
    dropout
        Dropout probability applied inside the transformer.
    use_stack_position_embedding
        If False, omit the per-position embedding -- treats the stack as
        an unordered set (matches the replicate use-case).
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int,
        stack_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_stack_position_embedding: bool = True,
    ) -> None:
        super().__init__(n_genes=n_genes, d_model=d_model, stack_size=stack_size)
        self.frame_proj = nn.Linear(n_genes, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.use_stack_position_embedding = use_stack_position_embedding
        if use_stack_position_embedding:
            # +1 for the CLS token slot.
            self.pos_embed = nn.Parameter(torch.zeros(1, stack_size + 1, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None  # type: ignore[assignment]

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=4,
                    dropout=dropout,
                    causal=False,
                    max_sequence_length=stack_size + 1,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, T, K, G), got {tuple(x.shape)}")
        b, t, k, g = x.shape
        if k != self.stack_size:
            raise ValueError(f"stack_size mismatch: got K={k}, expected {self.stack_size}")
        if g != self.n_genes:
            raise ValueError(f"n_genes mismatch: got G={g}, expected {self.n_genes}")

        # Fold the (B, T) batch into one big batch so we can run the
        # intra-stack transformer in a single matmul.
        h = x.reshape(b * t, k, g)
        h = self.frame_proj(h)  # (B*T, K, d)
        cls = self.cls_token.expand(b * t, -1, -1)  # (B*T, 1, d)
        h = torch.cat([cls, h], dim=1)  # (B*T, K+1, d)
        if self.pos_embed is not None:
            h = h + self.pos_embed  # broadcast over batch

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        s = h[:, 0]  # (B*T, d) -- CLS token
        return s.view(b, t, self.d_model)


class MLPStateStackEncoder(StateStackEncoder):
    """Flatten-and-MLP baseline encoder.

    Useful as an ablation against :class:`TransformerStateStackEncoder`.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int,
        stack_size: int,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__(n_genes=n_genes, d_model=d_model, stack_size=stack_size)
        layers: list[nn.Module] = []
        prev = n_genes * stack_size
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, T, K, G), got {tuple(x.shape)}")
        b, t, k, g = x.shape
        if k != self.stack_size or g != self.n_genes:
            raise ValueError(
                f"shape mismatch: got K={k}, G={g}, expected K={self.stack_size}, G={self.n_genes}"
            )
        h = x.reshape(b * t, k * g)
        s = self.net(h)
        return s.view(b, t, self.d_model)


def build_state_stack_encoder(
    kind: str,
    *,
    n_genes: int,
    d_model: int,
    stack_size: int,
    n_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
) -> StateStackEncoder:
    """Factory used by :class:`world_model.models.world_model.WorldModel`."""
    if kind == "transformer":
        return TransformerStateStackEncoder(
            n_genes=n_genes,
            d_model=d_model,
            stack_size=stack_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
    if kind == "mlp":
        return MLPStateStackEncoder(
            n_genes=n_genes,
            d_model=d_model,
            stack_size=stack_size,
            dropout=dropout,
        )
    raise ValueError(f"Unknown state-stack encoder kind: {kind!r}. Use 'transformer' or 'mlp'.")


__all__ = [
    "MLPStateStackEncoder",
    "StateStackEncoder",
    "TransformerStateStackEncoder",
    "build_state_stack_encoder",
]
