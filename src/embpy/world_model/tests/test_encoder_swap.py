"""Tests for :func:`embpy.world_model.training.transfer.apply_encoder_swap`.

We use synthetic providers + tables to avoid touching BioEmbedder. Each
strategy is exercised on dim-match and dim-mismatch inputs so the
trainable-parameter contract is tested explicitly.
"""

from __future__ import annotations

import numpy as np
import torch

from embpy.world_model.configs import ActionAdapterConfig
from embpy.world_model.data.datasets.base import GeneIndexer
from embpy.world_model.training.transfer import apply_encoder_swap


D_MODEL = 8


class _DummyProvider:
    """Minimal stand-in for ActionEmbeddingProvider; only exposes name + dim."""

    def __init__(self, name: str, dim: int) -> None:
        self._name = name
        self._dim = dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._dim


def _make_indexer(symbols: list[str]) -> GeneIndexer:
    return GeneIndexer.from_symbols(symbols)


def _make_table(indexer: GeneIndexer, dim: int, *, base: float = 0.0) -> np.ndarray:
    """Deterministic toy embeddings: row i = base + i along the first axis."""
    n = len(indexer)
    table = np.zeros((n, dim), dtype=np.float32)
    for i in range(1, n):
        table[i, :] = float(base + i)
    return table


def _fresh_model(symbols: list[str], dim: int, *, adapter_cfg: ActionAdapterConfig | None = None):
    """Build a tiny WorldModel just for the swap helper to mutate."""
    from embpy.world_model.models.world_model import build_world_model

    indexer = _make_indexer(symbols)
    table = _make_table(indexer, dim)
    model = build_world_model(
        n_genes=4,
        gene_embedding_table=torch.from_numpy(table),
        d_model=D_MODEL,
        stack_size=2,
        encoder_layers=1,
        encoder_heads=1,
        dynamics_layers=1,
        dynamics_heads=1,
        max_sequence_length=8,
        decoder_hidden_dims=(8,),
        action_adapter_cfg=adapter_cfg or ActionAdapterConfig(kind="linear"),
    )
    return model, table, indexer


def test_swap_none_dim_mismatch_raises() -> None:
    model, table, indexer = _fresh_model(["A", "B"], dim=D_MODEL)
    pre = _DummyProvider("X", D_MODEL)
    ft = _DummyProvider("Y", D_MODEL + 1)
    ft_table = np.zeros((3, D_MODEL + 1), dtype=np.float32)
    ft_indexer = _make_indexer(["A", "B"])
    try:
        apply_encoder_swap(
            model,
            pretrain_provider=pre,
            finetune_provider=ft,
            pretrain_table=table,
            finetune_table=ft_table,
            pretrain_indexer=indexer,
            finetune_indexer=ft_indexer,
            strategy="none",
            adapter_cfg=ActionAdapterConfig(kind="linear"),
            d_model=D_MODEL,
        )
    except ValueError as exc:
        assert "embedding dims" in str(exc)
    else:
        raise AssertionError("strategy='none' should reject mismatched dims.")


def test_swap_none_dim_match_replaces_table() -> None:
    model, _, _ = _fresh_model(["A", "B"], dim=D_MODEL)
    pre = _DummyProvider("X", D_MODEL)
    ft = _DummyProvider("X", D_MODEL)
    new_indexer = _make_indexer(["A", "B", "C"])
    new_table = _make_table(new_indexer, D_MODEL, base=10.0)
    apply_encoder_swap(
        model,
        pretrain_provider=pre,
        finetune_provider=ft,
        pretrain_table=None,
        finetune_table=new_table,
        pretrain_indexer=None,
        finetune_indexer=new_indexer,
        strategy="none",
        adapter_cfg=ActionAdapterConfig(kind="linear"),
        d_model=D_MODEL,
    )
    assert model.action_encoder.embed.num_embeddings == new_table.shape[0]
    assert torch.allclose(
        model.action_encoder.embed.weight.detach(),
        torch.from_numpy(new_table),
    )


def test_swap_reset_adapter_handles_dim_mismatch() -> None:
    model, _, _ = _fresh_model(["A", "B"], dim=D_MODEL)
    pre = _DummyProvider("X", D_MODEL)
    ft_dim = D_MODEL * 2
    ft = _DummyProvider("Y", ft_dim)
    ft_indexer = _make_indexer(["A", "B"])
    ft_table = _make_table(ft_indexer, ft_dim)

    apply_encoder_swap(
        model,
        pretrain_provider=pre,
        finetune_provider=ft,
        pretrain_table=None,
        finetune_table=ft_table,
        pretrain_indexer=None,
        finetune_indexer=ft_indexer,
        strategy="reset_adapter",
        adapter_cfg=ActionAdapterConfig(kind="linear"),
        d_model=D_MODEL,
    )
    assert model.action_encoder.embedding_dim == ft_dim
    # The embed table is the new FT one.
    assert model.action_encoder.embed.weight.shape == (ft_table.shape[0], ft_dim)
    # And the new adapter accepts ft_dim as input.
    fake_pooled = torch.zeros(1, 1, ft_dim)
    out = model.action_encoder.proj(fake_pooled)
    assert out.shape == (1, 1, D_MODEL)


def test_swap_reset_all_action_replaces_module() -> None:
    model, _, _ = _fresh_model(["A", "B"], dim=D_MODEL)
    pre = _DummyProvider("X", D_MODEL)
    ft_dim = 12
    ft = _DummyProvider("Y", ft_dim)
    ft_indexer = _make_indexer(["A", "B"])
    ft_table = _make_table(ft_indexer, ft_dim)
    old_action = model.action_encoder

    apply_encoder_swap(
        model,
        pretrain_provider=pre,
        finetune_provider=ft,
        pretrain_table=None,
        finetune_table=ft_table,
        pretrain_indexer=None,
        finetune_indexer=ft_indexer,
        strategy="reset_all_action",
        adapter_cfg=ActionAdapterConfig(kind="mlp", hidden_dim=32),
        d_model=D_MODEL,
    )
    # Whole module replaced -- different identity.
    assert model.action_encoder is not old_action
    assert model.action_encoder.embedding_dim == ft_dim


def test_swap_learn_alignment_freezes_adapter_and_trains_alignment() -> None:
    pre_indexer = _make_indexer(["A", "B", "C"])
    pre_table = _make_table(pre_indexer, D_MODEL, base=0.5)
    model, _, _ = _fresh_model(["A", "B", "C"], dim=D_MODEL)

    pre = _DummyProvider("X", D_MODEL)
    ft_dim = D_MODEL * 2
    ft = _DummyProvider("Y", ft_dim)
    ft_indexer = _make_indexer(["A", "B", "C"])
    ft_table = _make_table(ft_indexer, ft_dim, base=2.0)

    apply_encoder_swap(
        model,
        pretrain_provider=pre,
        finetune_provider=ft,
        pretrain_table=pre_table,
        finetune_table=ft_table,
        pretrain_indexer=pre_indexer,
        finetune_indexer=ft_indexer,
        strategy="learn_alignment",
        adapter_cfg=ActionAdapterConfig(kind="linear"),
        d_model=D_MODEL,
        alignment_epochs=2,
        alignment_lr=1e-2,
    )
    proj = model.action_encoder.proj
    # The proj is now Sequential(alignment, old_proj). Old proj is frozen.
    assert isinstance(proj, torch.nn.Sequential) and len(proj) == 2
    align, old_proj = proj[0], proj[1]
    for p in old_proj.parameters():
        assert p.requires_grad is False
    for p in align.parameters():
        assert p.requires_grad is True


def test_swap_learn_alignment_requires_shared_symbols() -> None:
    pre_indexer = _make_indexer(["A", "B"])
    pre_table = _make_table(pre_indexer, D_MODEL)
    model, _, _ = _fresh_model(["A", "B"], dim=D_MODEL)

    pre = _DummyProvider("X", D_MODEL)
    ft = _DummyProvider("Y", D_MODEL)
    ft_indexer = _make_indexer(["C", "D"])  # no shared symbols
    ft_table = _make_table(ft_indexer, D_MODEL)

    try:
        apply_encoder_swap(
            model,
            pretrain_provider=pre,
            finetune_provider=ft,
            pretrain_table=pre_table,
            finetune_table=ft_table,
            pretrain_indexer=pre_indexer,
            finetune_indexer=ft_indexer,
            strategy="learn_alignment",
            adapter_cfg=ActionAdapterConfig(kind="linear"),
            d_model=D_MODEL,
            alignment_epochs=1,
        )
    except RuntimeError as exc:
        assert "shared perturbation symbols" in str(exc)
    else:
        raise AssertionError("learn_alignment must fail with no shared symbols.")
