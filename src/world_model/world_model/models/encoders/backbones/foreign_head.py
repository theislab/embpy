"""Small trainable head that consumes pre-encoded foundation-model embeddings.

Shape contract
--------------
::

    input:  (B, T, K, embedding_dim)   -- K cell embeddings per timestep
    output: (B, T, d_model)            -- state tokens for the dynamics

Why stack-aware?
The dataset slicing logic is shared with the local backbone, which
emits ``(B, T, K, G)``. For foreign backbones we keep the K dimension
identical (replicate window of K cell embeddings) and pool over it
inside the head. Mean pool is the simplest sensible choice -- the
foundation model already produced a single-cell summary, so within a
replicate window of biologically-similar cells the mean is a
well-behaved aggregator. A learned attention pool would be a Phase-6
experiment.
"""

from __future__ import annotations

import torch
from torch import nn


class ForeignBackboneHead(nn.Module):
    """``(B, T, K, E) -> (B, T, d_model)`` head for STATE / STACK embeddings.

    When ``embedding_dim == d_model`` the projection collapses to
    :class:`nn.Identity`, matching what the user explicitly asked for
    in the spec ("allow the head to be Identity for byte-equivalent
    behavior when dims line up and the backbone is frozen").
    """

    def __init__(
        self,
        embedding_dim: int,
        d_model: int,
        stack_size: int,
        *,
        identity_when_dims_match: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or d_model <= 0:
            raise ValueError(
                f"embedding_dim and d_model must be positive, got "
                f"{embedding_dim} and {d_model}"
            )
        self.embedding_dim = int(embedding_dim)
        self.d_model = int(d_model)
        self.stack_size = int(stack_size)
        self.n_genes = int(embedding_dim)
        if identity_when_dims_match and embedding_dim == d_model:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Linear(embedding_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            b, t, k, e = x.shape
            if e != self.embedding_dim:
                raise ValueError(
                    f"embedding dim mismatch: got E={e}, expected {self.embedding_dim}"
                )
            if k != self.stack_size:
                raise ValueError(
                    f"stack_size mismatch: got K={k}, expected {self.stack_size}"
                )
            x = x.mean(dim=2)  # (B, T, E)
        elif x.ndim == 3:
            b, t, e = x.shape
            if e != self.embedding_dim:
                raise ValueError(
                    f"embedding dim mismatch: got E={e}, expected {self.embedding_dim}"
                )
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {tuple(x.shape)}")
        return self.proj(x)


__all__ = ["ForeignBackboneHead"]
