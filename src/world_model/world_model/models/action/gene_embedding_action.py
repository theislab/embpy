"""Gene-embedding action encoder.

The world model treats a *genetic perturbation* as the action. Each
action is the embedding of the perturbed gene in some pretrained gene
embedding space (GenePT, gene2vec, BioEmbedder-backed foundation
models, ...).

What the ``n_pert`` axis represents
-----------------------------------
``n_pert`` is the **maximum number of genes co-perturbed at a single
timestep** in the dataset (e.g. ``n_pert = 1`` for single-CRISPR
screens like Replogle / Nadig, ``n_pert = 2`` for double knockouts,
etc.). For each ``(b, t)`` step, ``gene_indices[b, t, :]`` lists the
indices of the genes perturbed at that step, right-padded with the
reserved row ``0`` (control / non-targeting / padding token) when the
local cardinality is smaller than ``n_pert``.

So for a double knockout of genes ``A`` and ``B`` at step ``t`` and a
dataset with ``n_pert = 2``, ``gene_indices[b, t, :] = [idx_A, idx_B]``.
For a single-gene knockout of ``A`` with the same global ``n_pert = 2``,
``gene_indices[b, t, :] = [idx_A, 0]``. For an all-control step,
``gene_indices[b, t, :] = [0, 0]``.

Data flow (multi-gene perturbations)
------------------------------------
::

    lookup    : (B, T, n_pert)        -> (B, T, n_pert, E_g)
    project   : ActionAdapter applied PER GENE over the trailing dim
                (Linear / MLP / LoRA), broadcasting cleanly over leading
                dims     ->  (B, T, n_pert, d_model)
    aggregate : mean / sum across the n_pert axis -- i.e. across the
                co-perturbed genes at each (b, t) step -- with padded
                slots (index 0) masked out
              -> (B, T, d_model)

Worked example. Take a double knockout of (A, B) at step t, with
``n_pert = 2``. The encoder:

1. Looks up two gene vectors ``e_A, e_B in R^{E_g}`` from the
   pretrained (frozen-by-default) embedding table.
2. Applies the trainable ``ActionAdapter`` *independently to each*:
   ``z_A = ActionAdapter(e_A)`` and ``z_B = ActionAdapter(e_B)``,
   both in ``R^{d_model}``.
3. Averages them (mean pool over the ``n_pert`` axis; padding ignored
   if any slot equals 0):
   ``a_t = (z_A + z_B) / 2  in R^{d_model}``.

For a single-gene knockout (``n_pert == 1`` or all-but-one slot is
padding) the average is a no-op and ``a_t = z_A``. For an all-control
step, the encoder falls back to ``proj(table[0])`` (see the
"control / all-padding edge case" block in ``forward``).

The aggregator is permutation-invariant on purpose: biologically
perturbing ``{A, B}`` is the same event as perturbing ``{B, A}``.

Why project then aggregate (rather than the reverse)?
-----------------------------------------------------

1. For the ``Linear`` adapter and ``pool="mean"`` the two orderings
   are **byte-equivalent** (linearity of ``W x + b`` commutes with
   masked-mean when at least one slot is valid). The all-padding edge
   case (control timesteps) is preserved explicitly by falling back to
   ``proj(table[0])`` instead of zero.
2. For the non-linear adapters (``MLP``, ``LoRA`` with dropout) it
   lets each co-perturbed gene undergo its own non-linear projection
   before the contributions are combined, which is strictly more
   expressive than averaging in the raw ``E_g`` space and then
   applying one non-linearity.

Implementation notes
--------------------
Inputs are *long* indices into the embedding table, not raw float
embeddings, which lets autograd ignore the embedding table when it is
frozen and avoids materialising the full lookup tensor for every batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from world_model.models.action.adapters import ActionAdapter, build_action_adapter

if TYPE_CHECKING:
    from world_model.configs import ActionAdapterConfig


def _default_adapter_cfg(project_hidden: int | None) -> ActionAdapterConfig:
    """Pick an :class:`ActionAdapterConfig` matching the legacy ``project_hidden`` knob.

    * ``project_hidden is None`` -> ``kind="linear"``, byte-equivalent to
      the pre-Phase-3 single ``nn.Linear`` projection.
    * ``project_hidden is int``  -> ``kind="mlp"`` with that hidden dim
      and the legacy ``GELU`` activation.
    """
    from world_model.configs import ActionAdapterConfig

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
        Aggregator across the ``n_pert`` axis -- i.e. across the genes
        that are co-perturbed at a single ``(b, t)`` step (e.g. the two
        genes of a double knockout) -- applied *after* the per-gene
        ``ActionAdapter`` projection: ``"mean"`` (default, padded slots
        masked) or ``"sum"``. For single-gene perturbations
        (``n_pert == 1`` or only one non-padded slot) both reduce to the
        identity.
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
        adapter_cfg: ActionAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        if gene_embedding_table.ndim != 2:
            raise ValueError(f"gene_embedding_table must be 2D, got shape {tuple(gene_embedding_table.shape)}")
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
        self._frozen_embedding_table_cpu = gene_embedding_table.detach().cpu().clone() if freeze_embeddings else None

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

        Notes
        -----
        The projection is applied **per gene** before aggregation. See the
        module docstring for the exact data flow and the byte-equivalence
        argument with the legacy aggregate-then-project ordering.
        """
        self._validate_gene_indices(gene_indices)
        emb = self._lookup_gene_embeddings(gene_indices)  # (B, T, n_pert, embedding_dim)
        # Project per gene. nn.Linear (and the Sequential / LoRA stacks built
        # on it) broadcast over arbitrary leading dimensions, so this single
        # forward pass produces one d_model vector per (B, T, n_pert) slot.
        projected = self.proj(emb)  # (B, T, n_pert, d_model)

        valid = (gene_indices != 0).to(device=projected.device, dtype=projected.dtype).unsqueeze(-1)
        n_valid = valid.sum(dim=-2)  # (B, T, 1); 0 for all-padding (control) steps
        masked_sum = (projected * valid).sum(dim=-2)  # (B, T, d_model)

        if self.pool == "mean":
            denom = n_valid.clamp(min=1.0)
            pooled = masked_sum / denom
        else:
            pooled = masked_sum

        # Control / all-padding edge case. With the legacy aggregate-then-
        # project ordering, an all-zero gene_indices row produced
        # ``proj(mean of zeros) == proj(0) == bias``, i.e. the projection of
        # the padding row. We preserve that exactly by adding the projected
        # padding embedding back where every slot was masked out. For mean
        # pooling with at least one valid slot this is a no-op; for sum
        # pooling with at least one valid slot it is also a no-op. This
        # keeps the Linear-mean output bit-exactly equal to the legacy code
        # for every input the dataloader can produce.
        control_mask = (n_valid == 0).to(pooled.dtype)  # (B, T, 1)
        if bool(control_mask.any()):
            # Go through nn.Embedding (not a raw .weight slice) so the
            # padding_idx=0 gradient-zeroing rule keeps applying when the
            # embedding table is trainable (freeze_embeddings=False).
            pad_idx = torch.zeros(1, dtype=torch.long, device=gene_indices.device)
            pad_token = self.proj(self._lookup_gene_embeddings(pad_idx))  # (1, d_model)
            pooled = pooled + control_mask * pad_token.view(1, 1, -1)

        return pooled

    def _validate_gene_indices(self, gene_indices: torch.Tensor) -> None:
        """Validate integer action indices before table lookup."""
        if gene_indices.ndim != 3:
            raise ValueError(f"Expected (B, T, n_pert) indices, got shape {tuple(gene_indices.shape)}")
        if gene_indices.dtype not in (torch.long, torch.int64, torch.int32):
            raise TypeError(f"gene_indices must be int/long, got {gene_indices.dtype}")
        if (gene_indices < 0).any() or (gene_indices >= self.n_rows).any():
            raise ValueError(
                f"gene_indices out of range [0, {self.n_rows - 1}]: "
                f"min={int(gene_indices.min())}, max={int(gene_indices.max())}"
            )

    def _lookup_gene_embeddings(self, gene_indices: torch.Tensor) -> torch.Tensor:
        """Lookup frozen action rows, using a CPU-index path for Apple MPS."""
        weight_device = self.embed.weight.device
        if weight_device.type == "mps" and gene_indices.device.type == "cpu":
            if self._frozen_embedding_table_cpu is None:
                raise RuntimeError("MPS CPU-index action lookup requires frozen action embeddings.")
            emb = nn.functional.embedding(
                gene_indices.to(torch.long),
                self._frozen_embedding_table_cpu,
                padding_idx=0,
            )
            return emb.to(weight_device)
        return self.embed(gene_indices)


__all__ = ["GeneEmbeddingAction"]
