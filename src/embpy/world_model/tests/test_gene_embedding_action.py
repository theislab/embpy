"""Tests for the gene-embedding action encoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from embpy.world_model.models.action.gene_embedding_action import GeneEmbeddingAction

N_GENES = 10
EMB_DIM = 8
D_MODEL = 16


def _make_table() -> torch.Tensor:
    table = torch.randn(N_GENES + 1, EMB_DIM)
    table[0].zero_()  # control row reserved as zeros
    return table


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
