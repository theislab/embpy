"""Factory parity test: ``kind="linear"`` is byte-equivalent to the legacy nn.Linear path.

This is the regression test the user asked for. We reconstruct the
exact pre-Phase-3 instantiation order:

    embed = nn.Embedding(...)
    proj = nn.Linear(d_in, d_model)

and assert it is bit-exact equal to the new path:

    embed = nn.Embedding(...)
    proj = build_action_adapter(d_in, d_model, ActionAdapterConfig())
"""

from __future__ import annotations

import torch
from torch import nn

from embpy.world_model.configs import ActionAdapterConfig
from embpy.world_model.models.action.adapters import (
    LinearAdapter,
    build_action_adapter,
)


D_IN = 32
D_MODEL = 16
N_ROWS = 7


def _build_legacy(seed: int) -> tuple[nn.Embedding, nn.Linear]:
    torch.manual_seed(seed)
    embed = nn.Embedding(num_embeddings=N_ROWS, embedding_dim=D_IN, padding_idx=0)
    proj = nn.Linear(D_IN, D_MODEL)
    return embed, proj


def _build_new(seed: int) -> tuple[nn.Embedding, LinearAdapter]:
    torch.manual_seed(seed)
    embed = nn.Embedding(num_embeddings=N_ROWS, embedding_dim=D_IN, padding_idx=0)
    proj = build_action_adapter(D_IN, D_MODEL, ActionAdapterConfig(kind="linear"))
    assert isinstance(proj, LinearAdapter)
    return embed, proj


def test_adapter_factory_parity() -> None:
    """kind="linear" -> bit-exact same parameters as nn.Linear at the same point."""
    legacy_embed, legacy_proj = _build_legacy(seed=12345)
    new_embed, new_proj = _build_new(seed=12345)
    assert isinstance(new_proj, LinearAdapter)

    assert torch.equal(legacy_embed.weight, new_embed.weight)
    assert torch.equal(legacy_proj.weight, new_proj.linear.weight)
    assert torch.equal(legacy_proj.bias, new_proj.linear.bias)


def test_adapter_factory_parity_forward_output() -> None:
    """Forward pass agrees bit-for-bit on a random input."""
    legacy_embed, legacy_proj = _build_legacy(seed=999)
    new_embed, new_proj = _build_new(seed=999)
    x = torch.randn(4, D_IN)
    assert torch.equal(legacy_proj(x), new_proj(x))
    idx = torch.randint(0, N_ROWS, (3, 5), dtype=torch.long)
    assert torch.equal(legacy_embed(idx), new_embed(idx))


def test_factory_dispatch_kinds() -> None:
    """Factory returns the right concrete class for each kind."""
    from embpy.world_model.models.action.adapters import LoRAAdapter, MLPAdapter

    cfg_lin = ActionAdapterConfig(kind="linear")
    cfg_mlp = ActionAdapterConfig(kind="mlp", hidden_dim=12, dropout=0.1)
    cfg_lora = ActionAdapterConfig(kind="lora", lora_rank=4, lora_alpha=2.0)
    assert isinstance(build_action_adapter(D_IN, D_MODEL, cfg_lin), LinearAdapter)
    assert isinstance(build_action_adapter(D_IN, D_MODEL, cfg_mlp), MLPAdapter)
    assert isinstance(build_action_adapter(D_IN, D_MODEL, cfg_lora), LoRAAdapter)


def test_factory_unknown_kind_raises() -> None:
    cfg = ActionAdapterConfig(kind="unknown")
    try:
        build_action_adapter(D_IN, D_MODEL, cfg)
    except ValueError as exc:
        assert "Unknown action_adapter.kind" in str(exc)
    else:
        raise AssertionError("Factory should reject unknown adapter kinds.")
