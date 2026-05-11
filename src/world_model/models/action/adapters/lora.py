"""LoRA adapter: frozen base linear + low-rank trainable residual.

``y = W0 x + b0 + (alpha / rank) * B (A x)`` where ``W0, b0`` are the
parameters of a regular ``nn.Linear`` (initialized normally and frozen),
``A`` is initialised with Kaiming uniform, and ``B`` is initialised with
zeros. The adapter therefore equals ``W0 x + b0`` at the start of
training (because ``B = 0``), so a freshly built LoRA adapter is
function-equivalent to its frozen base.

``rank == 0`` skips the LoRA path entirely; the adapter degenerates to
its frozen base linear -- useful as a control in the adapter sweep
(no trainable params anywhere in the action path).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import ActionAdapter


class LoRAAdapter(ActionAdapter):
    """Frozen base linear + low-rank residual."""

    name = "lora"

    def __init__(
        self,
        d_in: int,
        d_model: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._d_model = int(d_model)
        self.rank = int(rank)
        self.alpha = float(alpha)
        # The "frozen W0" path. Initialised exactly like a vanilla
        # nn.Linear, then frozen so the optimizer ignores it.
        self.W0 = nn.Linear(int(d_in), int(d_model))
        for p in self.W0.parameters():
            p.requires_grad_(False)

        if self.rank > 0:
            # Low-rank residual: B initialised to zero so the adapter
            # equals W0 at init.
            self.A = nn.Linear(int(d_in), self.rank, bias=False)
            self.B = nn.Linear(self.rank, int(d_model), bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.scale = self.alpha / float(self.rank)
            self.dropout: nn.Module = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        else:
            # rank == 0 -> degenerate to W0 only. No trainable params.
            self.A = None  # type: ignore[assignment]
            self.B = None  # type: ignore[assignment]
            self.scale = 0.0
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.W0(x)
        if self.rank > 0:
            assert self.A is not None and self.B is not None
            out = out + self.scale * self.B(self.A(self.dropout(x)))
        return out


__all__ = ["LoRAAdapter"]
