"""Cross-encoder transfer helpers.

The pretrain phase fixes the action-encoder to one provider X. When we
move to fine-tune with a different provider Y, three things can change:

* The embedding *dimension* (BioEmbedder backends differ -- Borzoi 1920,
  ESM2 1280, MiniLM 384, ...).
* The embedding *identity* (different foundation models occupy different
  geometric spaces even at matched dimension).
* The *gene set* (Nadig vs Replogle have only partially overlapping
  perturbations).

:func:`apply_encoder_swap` is the single entry-point that handles all
three. It is called from :func:`scripts.train._run_transfer` after the
pretrain phase has finished, with the model still holding pretrained
weights, the fine-tune table already materialised, and an explicit
``swap_strategy`` from :class:`TransferConfig`.

Strategies (see the README for a fuller treatment):

* ``"none"``               -- assert dim + identity match, fail loudly.
* ``"reset_adapter"``      -- keep dynamics + state encoder + decoder;
                              re-instantiate just the adapter head.
* ``"learn_alignment"``    -- freeze everything; learn a small
                              ``Linear(d_finetune, d_pretrain)`` against
                              shared perturbation symbols, then prepend
                              it to the existing (frozen) adapter.
* ``"reset_all_action"``   -- rebuild the entire action encoder.

Each branch logs which params are trainable so the trainer's optimizer
sees exactly the parameters we expect.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ..models.action.adapters import build_action_adapter
from ..models.action.gene_embedding_action import GeneEmbeddingAction

if TYPE_CHECKING:
    from ..configs import ActionAdapterConfig
    from ..data.datasets.base import GeneIndexer
    from ..data.embeddings.provider import ActionEmbeddingProvider
    from ..models.world_model import WorldModel

logger = logging.getLogger(__name__)


def _count_trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _log_trainable_breakdown(model: "WorldModel", tag: str) -> None:
    """Log a per-submodule trainable-param count so transfer mistakes surface fast."""
    parts = {
        "encoder": model.encoder,
        "action_encoder": model.action_encoder,
        "dynamics": model.dynamics,
    }
    if getattr(model, "decoder", None) is not None:
        parts["decoder"] = model.decoder  # type: ignore[assignment]
    breakdown = {k: _count_trainable(v) for k, v in parts.items()}
    logger.info("[%s] trainable params: total=%d %s", tag, sum(breakdown.values()), breakdown)


def _replace_embed_table(action_encoder: GeneEmbeddingAction, new_table: torch.Tensor) -> None:
    """Swap the lookup table while keeping the adapter intact."""
    if new_table.ndim != 2:
        raise ValueError(f"new_table must be 2D, got {tuple(new_table.shape)}")
    n_rows, embedding_dim = new_table.shape
    if int(embedding_dim) != action_encoder.embedding_dim:
        raise ValueError(
            f"_replace_embed_table requires matching embedding_dim "
            f"(have {action_encoder.embedding_dim}, got {embedding_dim})."
        )
    new_embed = nn.Embedding(num_embeddings=n_rows, embedding_dim=int(embedding_dim), padding_idx=0)
    with torch.no_grad():
        new_embed.weight.copy_(new_table)
    new_embed.weight.requires_grad_(False)
    action_encoder.embed = new_embed
    action_encoder.n_rows = int(n_rows)


def _shared_symbol_pairs(
    pretrain_indexer: "GeneIndexer",
    finetune_indexer: "GeneIndexer",
) -> list[tuple[int, int]]:
    """Return ``[(pre_row, ft_row), ...]`` for symbols common to both indexers (excluding control)."""
    pre_map = pretrain_indexer.symbol_to_index
    ft_map = finetune_indexer.symbol_to_index
    pairs: list[tuple[int, int]] = []
    for sym, pre_row in pre_map.items():
        if sym == "<control>":
            continue
        ft_row = ft_map.get(sym)
        if ft_row is None or ft_row == 0:
            continue
        pairs.append((int(pre_row), int(ft_row)))
    return pairs


def _train_alignment(
    pretrain_table: np.ndarray,
    finetune_table: np.ndarray,
    pretrain_indexer: "GeneIndexer",
    finetune_indexer: "GeneIndexer",
    *,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Linear:
    """Fit an ``nn.Linear(d_finetune, d_pretrain)`` on shared perturbation embeddings.

    Loss: MSE between ``alignment(ft_row)`` and ``pretrain_row`` over
    every symbol present in both indexers. Returns the trained module on
    CPU; the caller is responsible for moving it onto the training
    device.
    """
    pairs = _shared_symbol_pairs(pretrain_indexer, finetune_indexer)
    if len(pairs) < 2:
        raise RuntimeError(
            f"learn_alignment needs at least 2 shared perturbation symbols between "
            f"pretrain and fine-tune providers, found {len(pairs)}. Either pick a "
            f"smaller / overlapping perturbation set or switch swap_strategy to "
            f"reset_adapter."
        )
    pre_rows = torch.tensor([p for p, _ in pairs], dtype=torch.long)
    ft_rows = torch.tensor([f for _, f in pairs], dtype=torch.long)
    pre_X = torch.from_numpy(np.asarray(pretrain_table, dtype=np.float32))[pre_rows]   # (N, d_pre)
    ft_X = torch.from_numpy(np.asarray(finetune_table, dtype=np.float32))[ft_rows]     # (N, d_ft)

    d_pre = int(pre_X.shape[1])
    d_ft = int(ft_X.shape[1])
    alignment = nn.Linear(d_ft, d_pre)
    alignment.to(device)
    pre_X = pre_X.to(device)
    ft_X = ft_X.to(device)

    opt = torch.optim.Adam(alignment.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    last_loss = float("nan")
    for epoch in range(int(epochs)):
        opt.zero_grad()
        pred = alignment(ft_X)
        loss = loss_fn(pred, pre_X)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        logger.info("[alignment] epoch %d/%d loss=%.6f", epoch + 1, int(epochs), last_loss)
    logger.info(
        "[alignment] trained on %d shared symbols; final MSE=%.6f", len(pairs), last_loss
    )
    return alignment.cpu()


def apply_encoder_swap(
    model: "WorldModel",
    *,
    pretrain_provider: "ActionEmbeddingProvider",
    finetune_provider: "ActionEmbeddingProvider",
    pretrain_table: np.ndarray | None,
    finetune_table: np.ndarray,
    pretrain_indexer: "GeneIndexer | None",
    finetune_indexer: "GeneIndexer",
    strategy: str,
    adapter_cfg: "ActionAdapterConfig",
    d_model: int,
    alignment_epochs: int = 5,
    alignment_lr: float = 1e-3,
    device: str = "cpu",
) -> "WorldModel":
    """Adapt ``model`` from the pretrain provider to the fine-tune provider in-place.

    Returns the (mutated) model so the caller can chain.
    """
    pre_dim = int(pretrain_provider.embedding_dim)
    ft_dim = int(finetune_provider.embedding_dim)
    pre_name = pretrain_provider.name
    ft_name = finetune_provider.name

    logger.info(
        "Encoder swap: strategy=%s pretrain=%s(d=%d) finetune=%s(d=%d)",
        strategy, pre_name, pre_dim, ft_name, ft_dim,
    )

    ft_table_t = torch.from_numpy(np.asarray(finetune_table, dtype=np.float32))

    if strategy == "none":
        if pre_dim != ft_dim:
            raise ValueError(
                f"swap_strategy='none' requires identical embedding dims, "
                f"got pretrain={pre_dim} vs finetune={ft_dim}. Use "
                f"swap_strategy='reset_adapter' (rebuild adapter only), "
                f"'learn_alignment' (small Linear bridge), or "
                f"'reset_all_action' (full rebuild)."
            )
        if pre_name != ft_name:
            logger.warning(
                "swap_strategy='none' but provider identities differ "
                "(%s vs %s). Pretrained adapter will be reused as-is.",
                pre_name, ft_name,
            )
        _replace_embed_table(model.action_encoder, ft_table_t)
        _log_trainable_breakdown(model, tag="swap=none")
        return model

    if strategy == "reset_adapter":
        _replace_embed_table_force(model.action_encoder, ft_table_t)
        new_adapter = build_action_adapter(d_in=ft_dim, d_model=d_model, cfg=adapter_cfg)
        model.action_encoder.proj = new_adapter
        model.action_encoder.embedding_dim = ft_dim
        _log_trainable_breakdown(model, tag="swap=reset_adapter")
        return model

    if strategy == "reset_all_action":
        new_action = GeneEmbeddingAction(
            gene_embedding_table=ft_table_t,
            d_model=d_model,
            pool="mean",
            freeze_embeddings=True,
            adapter_cfg=adapter_cfg,
        )
        model.action_encoder = new_action
        _log_trainable_breakdown(model, tag="swap=reset_all_action")
        return model

    if strategy == "learn_alignment":
        if pretrain_table is None or pretrain_indexer is None:
            raise ValueError(
                "swap_strategy='learn_alignment' requires both pretrain_table "
                "and pretrain_indexer (the runner must keep the pretrain "
                "DataArtifacts around)."
            )
        alignment = _train_alignment(
            pretrain_table=pretrain_table,
            finetune_table=finetune_table,
            pretrain_indexer=pretrain_indexer,
            finetune_indexer=finetune_indexer,
            epochs=alignment_epochs,
            lr=alignment_lr,
            device=device,
        )
        _replace_embed_table_force(model.action_encoder, ft_table_t)
        old_proj = model.action_encoder.proj
        for p in old_proj.parameters():
            p.requires_grad_(False)
        # Path: pooled (d_ft) -> alignment -> (d_pre) -> old frozen adapter -> (d_model).
        model.action_encoder.proj = nn.Sequential(alignment, old_proj)
        model.action_encoder.embedding_dim = ft_dim
        _log_trainable_breakdown(model, tag="swap=learn_alignment")
        return model

    raise ValueError(
        f"Unknown swap_strategy={strategy!r}. Supported: "
        f"'none', 'reset_adapter', 'learn_alignment', 'reset_all_action'."
    )


def _replace_embed_table_force(action_encoder: GeneEmbeddingAction, new_table: torch.Tensor) -> None:
    """Like :func:`_replace_embed_table` but allows the embedding_dim to change.

    Used by reset_adapter / learn_alignment after the adapter is being
    rebuilt or wrapped, so the old dim no longer matters.
    """
    if new_table.ndim != 2:
        raise ValueError(f"new_table must be 2D, got {tuple(new_table.shape)}")
    n_rows, embedding_dim = new_table.shape
    new_embed = nn.Embedding(num_embeddings=int(n_rows), embedding_dim=int(embedding_dim), padding_idx=0)
    with torch.no_grad():
        new_embed.weight.copy_(new_table)
    new_embed.weight.requires_grad_(False)
    action_encoder.embed = new_embed
    action_encoder.n_rows = int(n_rows)
    action_encoder.embedding_dim = int(embedding_dim)


__all__ = ["apply_encoder_swap"]
