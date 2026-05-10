"""Base class for action adapters."""

from __future__ import annotations

import torch
from torch import nn


class ActionAdapter(nn.Module):
    """Common contract for every action-embedding -> dynamics-token adapter.

    Subclasses must:

    * Set the class attribute :attr:`name` to a short identifier used
      in metadata and logs (``"linear"``, ``"mlp"``, ``"lora"``, ...).
    * Implement :meth:`forward` with the shape contract
      ``(B, ..., d_in) -> (B, ..., d_model)``.
    * Set ``self._d_model`` to the output width so :attr:`out_dim`
      stays correct.
    """

    name: str = "adapter"

    def __init__(self) -> None:
        super().__init__()
        self._d_model: int = 0

    @property
    def out_dim(self) -> int:
        """Output token width."""
        return int(self._d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


__all__ = ["ActionAdapter"]
