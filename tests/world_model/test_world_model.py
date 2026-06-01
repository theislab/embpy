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


def _make_incontext_batch() -> dict:
    m = 3
    return {
        "support_obs": torch.randn(B, m, K, G),
        "support_act": torch.randint(0, N_GENES_PERT + 1, size=(B, m, 2)),
        "support_next": torch.randn(B, m, K, G),
        "query_obs": torch.randn(B, K, G),
        "query_act": torch.randint(0, N_GENES_PERT + 1, size=(B, 2)),
        "query_next": torch.randn(B, K, G),
        "query_next_expression": torch.randn(B, G),
        "obs_stack": torch.randn(B, K, G),
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


def _make_incontext_model(decoder: bool = True):
    return build_world_model(
        n_genes=G,
        gene_embedding_table=torch.randn(N_GENES_PERT + 1, 24),
        query_gene_embedding_table=torch.randn(N_GENES_PERT + 1, 10),
        encoder_kind="transformer",
        d_model=D,
        stack_size=K,
        encoder_layers=1,
        encoder_heads=2,
        dynamics_layers=1,
        dynamics_heads=2,
        dynamics_kind="incontext_set",
        incontext_support_size=3,
        max_sequence_length=8,
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


def test_incontext_support_query_action_tables_have_independent_dims() -> None:
    model = _make_incontext_model(decoder=True)
    batch = _make_incontext_batch()
    out = model.predict(batch)
    assert out["s_hat"].shape == (B, D)
    assert out["query_s"].shape == (B, D)
    assert out["x_hat"].shape == (B, G)
    loss, components = model.loss(
        batch,
        decoder_mse_weight=0.1,
        info_nce_weight=0.1,
        action_counterfactual_weight=0.1,
    )
    assert torch.isfinite(loss)
    assert "latent_mse" in components
    assert "info_nce" in components
    assert "pos_minus_neg" in components
    assert "delta_dim_var" in components
    assert "action_counterfactual" in components
    loss.backward()
