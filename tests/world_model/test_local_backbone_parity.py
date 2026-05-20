"""Regression guard: ``state_backbone.kind='local'`` must be bit-equivalent
to the pre-Phase-5 ``build_world_model`` path.

Why this matters
----------------
Every existing experiment YAML was written before Phase 5. The Phase-5
refactor introduces a provider abstraction over the state encoder; if
that refactor leaks any change into the local path (extra projection,
different init RNG order, an Identity head being inserted instead of
nothing), the numbers in every previous run would drift.

We therefore pin: with identical seeds and identical configs, the
world-model forward output matches the legacy one byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import torch

from world_model.configs import StateBackboneConfig
from world_model.models.world_model import build_world_model


def _make_batch(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    B, T, K, G = 2, 4, 3, 16
    obs = torch.randn(B, T, K, G, generator=g)
    next_obs = torch.randn(B, T, K, G, generator=g)
    actions = torch.randint(0, 5, (B, T, 2), generator=g)
    next_expr = torch.randn(B, T, G, generator=g)
    return {
        "obs_stack": obs,
        "next_obs_stack": next_obs,
        "action_indices": actions,
        "next_expression": next_expr,
    }


def _build(state_backbone_cfg, *, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    gene_table = torch.randn(5, 24, generator=g)
    return build_world_model(
        n_genes=16,
        gene_embedding_table=gene_table,
        encoder_kind="transformer",
        d_model=32,
        stack_size=3,
        encoder_layers=1,
        encoder_heads=2,
        dynamics_layers=1,
        dynamics_heads=2,
        dropout=0.0,
        max_sequence_length=8,
        use_action_token=True,
        decoder_hidden_dims=(32,),
        enable_decoder=True,
        action_adapter_cfg=None,
        state_backbone_cfg=state_backbone_cfg,
    )


def test_local_backbone_parity_with_legacy_default() -> None:
    legacy = _build(None, seed=1234)
    cfg = StateBackboneConfig(kind="local", freeze=False)
    new = _build(cfg, seed=1234)

    batch = _make_batch(seed=7)
    legacy.eval()
    new.eval()
    with torch.no_grad():
        out_legacy = legacy(batch)
        out_new = new(batch)

    np.testing.assert_allclose(
        out_legacy.s_hat.numpy(), out_new.s_hat.numpy(), rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        out_legacy.state_tokens.numpy(), out_new.state_tokens.numpy(), rtol=0, atol=0,
    )
    if out_legacy.x_hat is not None:
        np.testing.assert_allclose(
            out_legacy.x_hat.numpy(), out_new.x_hat.numpy(), rtol=0, atol=0,
        )


def test_local_backbone_freeze_disables_grad() -> None:
    cfg = StateBackboneConfig(kind="local", freeze=True)
    model = _build(cfg, seed=0)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    other_grads = [
        p.requires_grad
        for n, p in model.named_parameters()
        if not n.startswith("encoder.")
    ]
    assert any(other_grads), "non-encoder params should remain trainable"
