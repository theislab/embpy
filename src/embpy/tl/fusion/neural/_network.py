"""Optional PyTorch modules, imported only when neural fusion is fitted/loaded."""

import torch
from torch import nn


class FusionNetwork(nn.Module):
    def __init__(self, widths, hidden_dim, n_components, target_dim, aggregation):
        super().__init__()
        self.aggregation = aggregation
        if aggregation == "concat":
            self.concat = nn.Sequential(nn.Linear(sum(widths), hidden_dim), nn.GELU())
        else:
            self.adapters = nn.ModuleList([nn.Linear(width, hidden_dim) for width in widths])
            self.phi = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            if aggregation == "gated":
                self.gate = nn.Linear(hidden_dim, 1)
        self.rho = nn.Sequential(nn.Linear(hidden_dim, n_components), nn.GELU())
        self.head = nn.Linear(n_components, target_dim)

    def encode(self, blocks):
        if self.aggregation == "concat":
            return self.rho(self.concat(torch.cat(blocks, dim=1))), None
        tokens = torch.stack([self.phi(layer(x)) for layer, x in zip(self.adapters, blocks, strict=True)], dim=1)
        if self.aggregation == "gated":
            weights = torch.softmax(self.gate(tokens).squeeze(-1), dim=1)
        else:
            weights = torch.ones(tokens.shape[:2], device=tokens.device, dtype=tokens.dtype) / len(blocks)
        return self.rho((tokens * weights.unsqueeze(-1)).sum(dim=1)), weights

    def forward(self, blocks):
        embedding, _ = self.encode(blocks)
        return self.head(embedding)
