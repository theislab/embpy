"""Unit tests for the action-adapter modules.

These tests run on CPU only and never touch BioEmbedder.
"""

from __future__ import annotations

import torch

from embpy.world_model.models.action.adapters import (
    LinearAdapter,
    LoRAAdapter,
    MLPAdapter,
)


D_IN = 16
D_MODEL = 8


def test_linear_adapter_shape_contract() -> None:
    torch.manual_seed(0)
    adapter = LinearAdapter(d_in=D_IN, d_model=D_MODEL)
    x = torch.randn(3, 5, D_IN)
    y = adapter(x)
    assert y.shape == (3, 5, D_MODEL)
    assert adapter.out_dim == D_MODEL
    assert adapter.name == "linear"


def test_mlp_adapter_shape_and_param_growth() -> None:
    torch.manual_seed(0)
    small = MLPAdapter(d_in=D_IN, d_model=D_MODEL, hidden_dim=32, dropout=0.0)
    big = MLPAdapter(d_in=D_IN, d_model=D_MODEL, hidden_dim=256, dropout=0.0)
    x = torch.randn(2, D_IN)
    assert small(x).shape == (2, D_MODEL)
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_big > n_small


def test_mlp_adapter_invalid_activation() -> None:
    try:
        MLPAdapter(d_in=D_IN, d_model=D_MODEL, hidden_dim=8, activation="silu")
    except ValueError as exc:
        assert "activation" in str(exc)
    else:
        raise AssertionError("MLPAdapter should reject unknown activation.")


def test_lora_rank_zero_equals_w0() -> None:
    """rank=0 -> the adapter's forward output is exactly W0(x), bit-exact."""
    torch.manual_seed(42)
    adapter = LoRAAdapter(d_in=D_IN, d_model=D_MODEL, rank=0, alpha=1.0)
    x = torch.randn(4, D_IN)
    y_adapter = adapter(x)
    y_w0 = adapter.W0(x)
    assert torch.equal(y_adapter, y_w0)


def test_lora_starts_as_w0_when_rank_positive() -> None:
    """At init, B=0 -> output equals W0(x)."""
    torch.manual_seed(123)
    adapter = LoRAAdapter(d_in=D_IN, d_model=D_MODEL, rank=4, alpha=1.0)
    # Sanity: B is zeros at init.
    assert adapter.B is not None
    assert torch.allclose(adapter.B.weight, torch.zeros_like(adapter.B.weight))
    x = torch.randn(3, D_IN)
    y_adapter = adapter(x)
    y_w0 = adapter.W0(x)
    assert torch.allclose(y_adapter, y_w0, atol=0.0)


def test_lora_w0_frozen_a_b_trainable() -> None:
    adapter = LoRAAdapter(d_in=D_IN, d_model=D_MODEL, rank=4, alpha=1.0)
    for p in adapter.W0.parameters():
        assert p.requires_grad is False
    assert adapter.A is not None and adapter.A.weight.requires_grad is True
    assert adapter.B is not None and adapter.B.weight.requires_grad is True


def test_lora_param_count_grows_with_rank() -> None:
    counts = {}
    for rank in (0, 4, 16, 64):
        torch.manual_seed(0)
        ad = LoRAAdapter(d_in=D_IN, d_model=D_MODEL, rank=rank, alpha=1.0)
        n_train = sum(p.numel() for p in ad.parameters() if p.requires_grad)
        counts[rank] = n_train
    assert counts[0] == 0
    assert counts[4] < counts[16] < counts[64]


def test_lora_rank_zero_no_trainable_params() -> None:
    adapter = LoRAAdapter(d_in=D_IN, d_model=D_MODEL, rank=0, alpha=1.0)
    n_train = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert n_train == 0


def test_adapter_shape_contract_3d_input() -> None:
    """All adapters preserve leading dims, only the trailing dim changes."""
    torch.manual_seed(0)
    adapters = [
        LinearAdapter(d_in=D_IN, d_model=D_MODEL),
        MLPAdapter(d_in=D_IN, d_model=D_MODEL, hidden_dim=32),
        LoRAAdapter(d_in=D_IN, d_model=D_MODEL, rank=4, alpha=1.0),
    ]
    x = torch.randn(2, 7, D_IN)
    for adapter in adapters:
        y = adapter(x)
        assert y.shape == (2, 7, D_MODEL), f"{adapter.name} produced {tuple(y.shape)}"
