"""Tests for state-stack encoders.

Exercise both the transformer and MLP variants on a single shape, and
verify that swapping ``stack_size`` changes the input contract but not
the output contract.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from embpy.world_model.models.encoders.state_stack_encoder import (
    MLPStateStackEncoder,
    TransformerStateStackEncoder,
    build_state_stack_encoder,
)

B, T, K, G, D = 2, 3, 4, 64, 32


def test_transformer_encoder_shape() -> None:
    enc = TransformerStateStackEncoder(n_genes=G, d_model=D, stack_size=K, n_layers=2, n_heads=4)
    x = torch.randn(B, T, K, G)
    s = enc(x)
    assert s.shape == (B, T, D)


def test_mlp_encoder_shape() -> None:
    enc = MLPStateStackEncoder(n_genes=G, d_model=D, stack_size=K)
    x = torch.randn(B, T, K, G)
    s = enc(x)
    assert s.shape == (B, T, D)


def test_factory_dispatch() -> None:
    enc = build_state_stack_encoder(kind="transformer", n_genes=G, d_model=D, stack_size=K)
    assert isinstance(enc, TransformerStateStackEncoder)
    enc = build_state_stack_encoder(kind="mlp", n_genes=G, d_model=D, stack_size=K)
    assert isinstance(enc, MLPStateStackEncoder)
    with pytest.raises(ValueError):
        build_state_stack_encoder(kind="resnet", n_genes=G, d_model=D, stack_size=K)


def test_encoder_rejects_wrong_shape() -> None:
    enc = TransformerStateStackEncoder(n_genes=G, d_model=D, stack_size=K)
    with pytest.raises(ValueError):
        enc(torch.randn(B, T, K + 1, G))
    with pytest.raises(ValueError):
        enc(torch.randn(B, T, K, G + 1))
