"""Gene-embedding action encoder.

The world model treats a *genetic perturbation* as the action. Each
action is the embedding of the perturbed gene in some pretrained gene
embedding space (GenePT, gene2vec, BioEmbedder-backed foundation
models, ...).

For multi-gene perturbations (e.g. dual-CRISPR) we pool the per-gene
embeddings -- mean by default, sum optionally -- before projecting to
the dynamics token width through a swappable :class:`ActionAdapter`
(see :mod:`models.action.adapters`).

Inputs are *long* indices into the embedding table, not raw float
embeddings, which lets autograd ignore the embedding table when it is
frozen and avoids materialising the full lookup tensor for every batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .adapters import ActionAdapter, build_action_adapter

if TYPE_CHECKING:
    from ...configs import ActionAdapterConfig


def _default_adapter_cfg(project_hidden: int | None) -> "ActionAdapterConfig":
    """Pick an :class:`ActionAdapterConfig` matching the legacy ``project_hidden`` knob.

    * ``project_hidden is None`` -> ``kind="linear"``, byte-equivalent to
      the pre-Phase-3 single ``nn.Linear`` projection.
    * ``project_hidden is int``  -> ``kind="mlp"`` with that hidden dim
      and the legacy ``GELU`` activation.
    """
    from ...configs import ActionAdapterConfig  # noqa: PLC0415

    if project_hidden is None:
        return ActionAdapterConfig(kind="linear")
    return ActionAdapterConfig(
        kind="mlp",
        hidden_dim=int(project_hidden),
        dropout=0.0,
        activation="gelu",
    )


class GeneEmbeddingAction(nn.Module):
    """Embed a (possibly multi-gene) perturbation into a token of width ``d_model``.

    Parameters
    ----------
    gene_embedding_table
        Tensor of shape ``(n_genes + 1, embedding_dim)``. The 0-th row
        is reserved for the control / non-targeting action and the
        padding token used for shorter perturbation cardinalities.
    d_model
        Output token width (matches the dynamics ``d_model``).
    pool
        Pooling over multi-gene perturbations: ``"mean"`` or ``"sum"``.
    freeze_embeddings
        If True, the embedding table's weights stay non-trainable.
    project_hidden
        Legacy knob: ``None`` -> single linear projection, ``int`` -> MLP
        with the given hidden width. Ignored when ``adapter_cfg`` is
        explicitly passed.
    adapter_cfg
        Optional :class:`ActionAdapterConfig` selecting the adapter kind.
        ``None`` (the default) falls back to :func:`_default_adapter_cfg`,
        preserving Phase-1/2 behaviour.
    """

    def __init__(
        self,
        gene_embedding_table: torch.Tensor,
        d_model: int,
        pool: str = "mean",
        freeze_embeddings: bool = True,
        project_hidden: int | None = None,
        adapter_cfg: "ActionAdapterConfig | None" = None,
    ) -> None:
        super().__init__()
        if gene_embedding_table.ndim != 2:
            raise ValueError(
                f"gene_embedding_table must be 2D, got shape {tuple(gene_embedding_table.shape)}"
            )
        if pool not in {"mean", "sum"}:
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")

        n_rows, embedding_dim = gene_embedding_table.shape
        self.n_rows = int(n_rows)
        self.embedding_dim = int(embedding_dim)
        self.d_model = int(d_model)
        self.pool = pool

        # nn.Embedding with padding_idx=0 keeps the control row's
        # gradient at zero even when freeze_embeddings is False.
        self.embed = nn.Embedding(num_embeddings=n_rows, embedding_dim=embedding_dim, padding_idx=0)
        with torch.no_grad():
            self.embed.weight.copy_(gene_embedding_table)
        if freeze_embeddings:
            self.embed.weight.requires_grad_(False)

        if adapter_cfg is None:
            adapter_cfg = _default_adapter_cfg(project_hidden)
        self.proj: ActionAdapter = build_action_adapter(embedding_dim, d_model, adapter_cfg)

    def forward(self, gene_indices: torch.Tensor) -> torch.Tensor:
        """Embed a batch of perturbations.

        Parameters
        ----------
        gene_indices
            ``(B, T, n_pert)`` long tensor.

        Returns
        -------
        torch.Tensor
            Action tokens of shape ``(B, T, d_model)``.
        """
        if gene_indices.ndim != 3:
            raise ValueError(
                f"Expected (B, T, n_pert) indices, got shape {tuple(gene_indices.shape)}"
            )
        if gene_indices.dtype not in (torch.long, torch.int64, torch.int32):
            raise TypeError(f"gene_indices must be int/long, got {gene_indices.dtype}")
        if (gene_indices < 0).any() or (gene_indices >= self.n_rows).any():
            raise ValueError(
                f"gene_indices out of range [0, {self.n_rows - 1}]: "
                f"min={int(gene_indices.min())}, max={int(gene_indices.max())}"
            )

        emb = self.embed(gene_indices)  # (B, T, n_pert, embedding_dim)
        valid = (gene_indices != 0).float().unsqueeze(-1)  # (B, T, n_pert, 1)

        if self.pool == "mean":
            denom = valid.sum(dim=-2).clamp(min=1.0)
            pooled = (emb * valid).sum(dim=-2) / denom
        else:
            pooled = (emb * valid).sum(dim=-2)

        return self.proj(pooled)


__all__ = ["GeneEmbeddingAction"]
