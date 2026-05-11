"""Two-layer MLP adapter."""

from __future__ import annotations

import torch
from torch import nn

from .base import ActionAdapter


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class MLPAdapter(ActionAdapter):
    """``y = W2 (act(W1 x))`` with optional dropout between the two linears."""

    name = "mlp"

    def __init__(
        self,
        d_in: int,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self._d_model = int(d_model)
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )
        act_cls = _ACTIVATIONS[activation]
        modules: list[nn.Module] = [
            nn.Linear(int(d_in), int(hidden_dim)),
            act_cls(),
        ]
        if dropout > 0.0:
            modules.append(nn.Dropout(float(dropout)))
        modules.append(nn.Linear(int(hidden_dim), int(d_model)))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["MLPAdapter"]
