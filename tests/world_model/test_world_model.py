"""Integration tests for the composed :class:`WorldModel`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from world_model.models.world_model import build_world_model

B, T, K, G, D = 2, 4, 3, 32, 16
N_GENES_PERT = 6


def _make_batch() -> dict:
    return {
        "obs_stack": torch.randn(B, T, K, G),
        "next_obs_stack": torch.randn(B, T, K, G),
        "action_indices": torch.randint(0, N_GENES_PERT + 1, size=(B, T, 2)),
        "next_expression": torch.randn(B, T, G),
    }


def _make_model(decoder: bool = True):
    return build_world_model(
        n_genes=G,
        gene_embedding_table=torch.randn(N_GENES_PERT + 1, 24),
        encoder_kind="transformer",
        d_model=D,
        stack_size=K,
        encoder_layers=1,
        encoder_heads=2,
        dynamics_layers=2,
        dynamics_heads=2,
        max_sequence_length=4 * T,
        enable_decoder=decoder,
    )


def test_forward_shapes() -> None:
    model = _make_model(decoder=True)
    out = model(_make_batch())
    assert out.s_hat.shape == (B, T, D)
    assert out.s_target.shape == (B, T, D)
    assert out.x_hat is not None
    assert out.x_hat.shape == (B, T, G)


def test_loss_runs_and_backprops() -> None:
    model = _make_model(decoder=True)
    loss, components = model.loss(_make_batch(), info_nce_weight=0.1)
    assert torch.isfinite(loss)
    assert "latent_mse" in components
    assert "decoder_mse" in components
    loss.backward()


def test_rollout_shapes() -> None:
    model = _make_model(decoder=True).eval()
    init_obs = torch.randn(B, K, G)
    actions = torch.randint(0, N_GENES_PERT + 1, size=(B, T, 2))
    out = model.rollout(init_obs, actions)
    assert out["s_hat"].shape == (B, T, D)
    assert out["x_hat"].shape == (B, T, G)


def test_mismatched_d_model_raises() -> None:
    from world_model.models import (
        ExpressionDecoder,
        GeneEmbeddingAction,
        GPTAutoregressiveDynamics,
        TransformerStateStackEncoder,
        WorldModel,
    )

    enc = TransformerStateStackEncoder(n_genes=G, d_model=D, stack_size=K)
    act = GeneEmbeddingAction(torch.randn(N_GENES_PERT + 1, 24), d_model=D)
    dyn = GPTAutoregressiveDynamics(d_model=D + 8)  # mismatched
    dec = ExpressionDecoder(d_model=D, n_genes=G)
    with pytest.raises(ValueError):
        WorldModel(encoder=enc, action_encoder=act, dynamics=dyn, decoder=dec)
