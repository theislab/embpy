"""Linear adapter: drop-in replacement for the inline nn.Linear baseline.

Construction order is identical to the pre-Phase-3 path: a single
``nn.Linear(d_in, d_model)`` is the only RNG-consuming call, so given
the same seed this adapter produces byte-equivalent parameters and
forward outputs to the legacy code (verified by the parity test in
``tests/test_action_adapter_factory.py``).
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ActionAdapter


class LinearAdapter(ActionAdapter):
    """``y = W x + b`` -- the smallest possible adapter."""

    name = "linear"

    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()
        self._d_model = int(d_model)
        self.linear = nn.Linear(int(d_in), int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


__all__ = ["LinearAdapter"]
