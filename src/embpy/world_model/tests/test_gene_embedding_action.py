"""Tests for the gene-embedding action encoder.

Covers shape, padding-aware aggregation, the project-then-aggregate
ordering, and byte-equivalence with the legacy aggregate-then-project
path under the conditions where the two orderings are provably the same
(``Linear`` adapter + ``mean`` pool, including the all-padding control
edge case).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from embpy.world_model.configs import ActionAdapterConfig
from embpy.world_model.models.action.gene_embedding_action import GeneEmbeddingAction

N_GENES = 10
EMB_DIM = 8
D_MODEL = 16


def _make_table() -> torch.Tensor:
    table = torch.randn(N_GENES + 1, EMB_DIM)
    table[0].zero_()  # control row reserved as zeros
    return table


def _legacy_forward(enc: GeneEmbeddingAction, gene_indices: torch.Tensor) -> torch.Tensor:
    """Reproduce the pre-Fix-2 aggregate-then-project ordering for parity tests."""
    emb = enc.embed(gene_indices)
    valid = (gene_indices != 0).float().unsqueeze(-1)
    if enc.pool == "mean":
        denom = valid.sum(dim=-2).clamp(min=1.0)
        pooled = (emb * valid).sum(dim=-2) / denom
    else:
        pooled = (emb * valid).sum(dim=-2)
    return enc.proj(pooled)


def test_action_shape_single_gene() -> None:
    enc = GeneEmbeddingAction(_make_table(), d_model=D_MODEL)
    indices = torch.tensor([[[1], [2], [0]], [[3], [0], [4]]])  # (2, 3, 1)
    out = enc(indices)
    assert out.shape == (2, 3, D_MODEL)


def test_action_pool_mean_handles_padding() -> None:
    enc = GeneEmbeddingAction(_make_table(), d_model=D_MODEL, pool="mean")
    # (1, 1, 3) -- two real genes (1, 2) + one padding (0).
    indices = torch.tensor([[[1, 2, 0]]])
    out = enc(indices)
    assert out.shape == (1, 1, D_MODEL)
    assert torch.isfinite(out).all()


def test_action_rejects_bad_indices() -> None:
    enc = GeneEmbeddingAction(_make_table(), d_model=D_MODEL)
    with pytest.raises(ValueError):
        enc(torch.tensor([[[N_GENES + 5]]]))
    with pytest.raises(ValueError):
        enc(torch.tensor([[[-1]]]))


def test_freeze_embeddings() -> None:
    enc = GeneEmbeddingAction(_make_table(), d_model=D_MODEL, freeze_embeddings=True)
    assert enc.embed.weight.requires_grad is False
    enc = GeneEmbeddingAction(_make_table(), d_model=D_MODEL, freeze_embeddings=False)
    assert enc.embed.weight.requires_grad is True


# ---------------------------------------------------------------------
# Project-then-aggregate ordering: byte-equivalence and behavioral diffs
# ---------------------------------------------------------------------


def test_linear_mean_parity_single_gene() -> None:
    """Linear-mean is byte-equivalent to the legacy ordering for n_pert=1."""
    torch.manual_seed(0)
    enc = GeneEmbeddingAction(
        _make_table(), d_model=D_MODEL, pool="mean",
        adapter_cfg=ActionAdapterConfig(kind="linear"),
    )
    indices = torch.tensor([[[1], [2], [0]], [[3], [0], [4]]])
    out_new = enc(indices)
    out_legacy = _legacy_forward(enc, indices)
    assert torch.allclose(out_new, out_legacy, atol=1e-6)


def test_linear_mean_parity_multi_gene_with_padding() -> None:
    """Linear-mean with n_pert>1 and partial padding is byte-equivalent (linearity)."""
    torch.manual_seed(1)
    enc = GeneEmbeddingAction(
        _make_table(), d_model=D_MODEL, pool="mean",
        adapter_cfg=ActionAdapterConfig(kind="linear"),
    )
    # Mix of: full set (3 real), partial (2 real + 1 pad), single (1 real + 2 pad).
    indices = torch.tensor([[[1, 2, 3], [4, 5, 0], [6, 0, 0]]])
    out_new = enc(indices)
    out_legacy = _legacy_forward(enc, indices)
    assert torch.allclose(out_new, out_legacy, atol=1e-6)


def test_linear_mean_parity_all_padding_control() -> None:
    """All-padding (control) timestep returns ``proj(table[0])`` in both orderings."""
    torch.manual_seed(2)
    enc = GeneEmbeddingAction(
        _make_table(), d_model=D_MODEL, pool="mean",
        adapter_cfg=ActionAdapterConfig(kind="linear"),
    )
    indices = torch.zeros(2, 4, 3, dtype=torch.long)  # all control
    out_new = enc(indices)
    out_legacy = _legacy_forward(enc, indices)
    assert torch.allclose(out_new, out_legacy, atol=1e-6)


def test_mlp_multi_gene_diverges_from_legacy() -> None:
    """With a non-linear adapter, project-then-aggregate differs from aggregate-then-project.

    This is the *desired* behavior change: each perturbed gene gets its
    own non-linear transformation before the contributions are combined,
    which is strictly more expressive than averaging in E_g space first.
    For n_pert=1 the two orderings still agree (single-slot average is
    identity); the difference only shows up for n_pert>=2.
    """
    torch.manual_seed(3)
    enc = GeneEmbeddingAction(
        _make_table(), d_model=D_MODEL, pool="mean",
        adapter_cfg=ActionAdapterConfig(kind="mlp", hidden_dim=12, dropout=0.0, activation="gelu"),
    )
    enc.eval()  # dropout=0 already, but be explicit
    indices = torch.tensor([[[1, 2, 3]]])
    out_new = enc(indices)
    out_legacy = _legacy_forward(enc, indices)
    assert not torch.allclose(out_new, out_legacy, atol=1e-6)


def test_mlp_single_gene_still_matches_legacy() -> None:
    """For n_pert=1 (the Replogle/Nadig regime), MLP outputs are unchanged."""
    torch.manual_seed(4)
    enc = GeneEmbeddingAction(
        _make_table(), d_model=D_MODEL, pool="mean",
        adapter_cfg=ActionAdapterConfig(kind="mlp", hidden_dim=12, dropout=0.0, activation="gelu"),
    )
    enc.eval()
    indices = torch.tensor([[[1], [2], [3], [0]]])  # all n_pert=1
    out_new = enc(indices)
    out_legacy = _legacy_forward(enc, indices)
    assert torch.allclose(out_new, out_legacy, atol=1e-6)


def test_padding_slot_does_not_leak_into_aggregate() -> None:
    """Adding a padding slot to a single-gene call must not change the output."""
    torch.manual_seed(5)
    enc = GeneEmbeddingAction(
        _make_table(), d_model=D_MODEL, pool="mean",
        adapter_cfg=ActionAdapterConfig(kind="mlp", hidden_dim=12, dropout=0.0, activation="gelu"),
    )
    enc.eval()
    single = torch.tensor([[[5]]])
    padded = torch.tensor([[[5, 0, 0]]])
    out_single = enc(single)
    out_padded = enc(padded)
    assert torch.allclose(out_single, out_padded, atol=1e-6)


def test_control_token_consistent_across_pool_modes() -> None:
    """All-padding timesteps yield ``proj(table[0])`` for both mean and sum pool."""
    torch.manual_seed(6)
    enc_mean = GeneEmbeddingAction(_make_table(), d_model=D_MODEL, pool="mean")
    enc_sum = GeneEmbeddingAction(_make_table(), d_model=D_MODEL, pool="sum")
    # Copy proj weights so the comparison only depends on the pooling rule.
    enc_sum.proj.load_state_dict(enc_mean.proj.state_dict())
    enc_sum.embed.load_state_dict(enc_mean.embed.state_dict())

    indices = torch.zeros(1, 1, 3, dtype=torch.long)
    expected = enc_mean.proj(enc_mean.embed(torch.zeros(1, dtype=torch.long)))
    out_mean = enc_mean(indices).reshape(-1, D_MODEL)
    out_sum = enc_sum(indices).reshape(-1, D_MODEL)
    assert torch.allclose(out_mean, expected, atol=1e-6)
    assert torch.allclose(out_sum, expected, atol=1e-6)


def test_no_grad_leak_into_padding_row_when_trainable() -> None:
    """With freeze_embeddings=False, row 0 must still receive zero gradient (padding_idx)."""
    table = _make_table()
    enc = GeneEmbeddingAction(table, d_model=D_MODEL, pool="mean", freeze_embeddings=False)
    indices = torch.tensor([[[1, 0, 0], [0, 0, 0]]])  # mix of valid + control rows
    out = enc(indices)
    out.sum().backward()
    assert enc.embed.weight.grad is not None
    assert torch.equal(enc.embed.weight.grad[0], torch.zeros(EMB_DIM))
