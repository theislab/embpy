"""Explicit 3-token-per-triplet, permutation-invariant set dynamics.

The literal implementation of the "context-SSL over triplets" picture:
each triplet ``(s, a, s')`` becomes THREE tokens instead of one fused
token, so the bidirectional transformer can attend over individual
states / actions / results across the whole context (e.g. the query
can look up the support triplet whose *action* is closest to its own
and read that triplet's ``s'``).

Directionality ``s, a -> s'`` is taught WITHOUT a causal mask:

* a local-position / **type** embedding tags each token's role inside
  its triplet (``state`` / ``action`` / ``next_state``);
* the **objective** only ever predicts the query's masked ``s'`` from
  its ``s, a`` (never the reverse);
* a per-triplet **group** embedding (a random permutation each forward)
  binds the 3 tokens of one triplet together while carrying no absolute
  order -- so the context stays a *set* (permutation invariant across
  triplets), with no positional embeddings.

Same ``forward`` signature as :class:`InContextSetDynamics`, so it is a
drop-in alternative inside :class:`InContextWorldModel`.

The training-time ``forward_autoregressive`` method implements the
teacher-forced in-context SSL objective: predict support triplet 1 with
no previous triplets, predict support triplet 2 with triplet 1 revealed,
..., then predict the held-out query with all support triplets revealed.
"""

from __future__ import annotations

import torch
from torch import nn

from ..blocks import TransformerBlock


class InContextTokensDynamics(nn.Module):
    use_action_token: bool = True

    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_set_size: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_set_size = int(max_set_size)

        self.type_embed = nn.Embedding(3, d_model)   # 0=state, 1=action, 2=next
        self.role_embed = nn.Embedding(2, d_model)   # 0=support, 1=query
        # Per-triplet binding tag. Assigned by a random permutation each
        # forward so it groups a triplet's 3 tokens without encoding any
        # absolute position -> permutation invariant across triplets.
        self.group_embed = nn.Embedding(max_set_size, d_model)
        # Learned placeholder for the query's unknown s'.
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                causal=False,                       # bidirectional over the set
                max_sequence_length=3 * max_set_size,
            )
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    def forward(
        self,
        support_s: torch.Tensor,   # (B, M, d)
        support_a: torch.Tensor,   # (B, M, d)
        support_sp: torch.Tensor,  # (B, M, d)
        query_s: torch.Tensor,     # (B, d)
        query_a: torch.Tensor,     # (B, d)
    ) -> torch.Tensor:
        if support_s.ndim != 3:
            raise ValueError(
                f"support_s must be (B, M, d), got {tuple(support_s.shape)}"
            )
        b, m, d = support_s.shape
        n = m + 1  # triplets incl. query
        if n > self.max_set_size:
            raise ValueError(f"set size {n} exceeds max_set_size {self.max_set_size}")
        dev = support_s.device

        mask = self.mask_token.expand(b, d).unsqueeze(1)          # (B,1,d)
        s = torch.cat([support_s, query_s.unsqueeze(1)], dim=1)   # (B,n,d)
        a = torch.cat([support_a, query_a.unsqueeze(1)], dim=1)   # (B,n,d)
        sp = torch.cat([support_sp, mask], dim=1)                 # (B,n,d) query=mask

        # Interleave to [s0,a0,sp0, s1,a1,sp1, ...]  -> (B, 3n, d)
        tokens = torch.stack([s, a, sp], dim=2).reshape(b, 3 * n, d)

        # type: state/action/next repeated per triplet.
        type_ids = torch.tensor([0, 1, 2], device=dev).repeat(n)          # (3n,)
        # role: query triplet (last) flagged 1.
        role_ids = torch.zeros(3 * n, dtype=torch.long, device=dev)
        role_ids[3 * m:] = 1
        # group: random permutation so binding carries no absolute order.
        perm = torch.randperm(n, device=dev)                              # (n,)
        group_ids = perm.repeat_interleave(3)                            # (3n,)

        h = (
            tokens
            + self.type_embed(type_ids)[None]
            + self.role_embed(role_ids)[None]
            + self.group_embed(group_ids)[None]
        )
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        # Query's s' (MASK) token is the last position by construction.
        return self.head(h[:, -1])

    def forward_autoregressive(
        self,
        support_s: torch.Tensor,   # (B, M, d)
        support_a: torch.Tensor,   # (B, M, d)
        support_sp: torch.Tensor,  # (B, M, d)
        query_s: torch.Tensor,     # (B, d)
        query_a: torch.Tensor,     # (B, d)
        query_sp: torch.Tensor,    # (B, d)
    ) -> torch.Tensor:
        """Predict every triplet's next-state with teacher-forced context.

        Returns one prediction per support triplet plus the final query,
        shape ``(B, M + 1, d)``. For target triplet ``j`` the transformer
        input contains all earlier triplets fully revealed, then the
        current triplet as ``[s_j, a_j, MASK]``. Future triplets are
        absent via an attention mask. This matches the in-context SSL
        objective:

        ``context_{j-1} + [s_j, a_j, MASK] -> s'_j``.
        """
        if support_s.ndim != 3:
            raise ValueError(f"support_s must be (B, M, d), got {tuple(support_s.shape)}")
        b, m, d = support_s.shape
        n = m + 1
        if n > self.max_set_size:
            raise ValueError(f"set size {n} exceeds max_set_size {self.max_set_size}")
        dev = support_s.device

        s = torch.cat([support_s, query_s.unsqueeze(1)], dim=1)       # (B,n,d)
        a = torch.cat([support_a, query_a.unsqueeze(1)], dim=1)       # (B,n,d)
        sp = torch.cat([support_sp, query_sp.unsqueeze(1)], dim=1)    # (B,n,d)

        max_len = 3 * n
        mask = self.mask_token.to(dev).expand(b, d)
        tokens = torch.zeros(b, n, max_len, d, dtype=s.dtype, device=dev)
        attn_mask = torch.zeros(b, n, max_len, dtype=torch.bool, device=dev)
        role_ids = torch.zeros(n, max_len, dtype=torch.long, device=dev)
        group_ids = torch.zeros(n, max_len, dtype=torch.long, device=dev)
        target_positions = torch.empty(n, dtype=torch.long, device=dev)

        for j in range(n):
            if j > 0:
                prefix = torch.stack([s[:, :j], a[:, :j], sp[:, :j]], dim=2).reshape(b, 3 * j, d)
                tokens[:, j, : 3 * j] = prefix
                attn_mask[:, j, : 3 * j] = True

            cur = 3 * j
            tokens[:, j, cur] = s[:, j]
            tokens[:, j, cur + 1] = a[:, j]
            tokens[:, j, cur + 2] = mask
            attn_mask[:, j, cur : cur + 3] = True
            role_ids[j, cur : cur + 3] = 1
            target_positions[j] = cur + 2

            # Randomized group tags bind the local triplet tokens without
            # introducing a stable absolute support position.
            perm = torch.randperm(n, device=dev)
            for i in range(j + 1):
                group_ids[j, 3 * i : 3 * i + 3] = perm[i]

        flat_tokens = tokens.reshape(b * n, max_len, d)
        flat_attn = attn_mask.reshape(b * n, max_len)
        type_ids = torch.tensor([0, 1, 2], device=dev).repeat(n)
        flat_role_ids = role_ids.repeat(b, 1)
        flat_group_ids = group_ids.repeat(b, 1)

        h = (
            flat_tokens
            + self.type_embed(type_ids)[None]
            + self.role_embed(flat_role_ids)
            + self.group_embed(flat_group_ids)
        )
        for layer in self.layers:
            h = layer(h, attn_mask=flat_attn)
        h = self.norm(h)

        flat_pos = target_positions.repeat(b)
        rows = torch.arange(b * n, device=dev)
        pred = self.head(h[rows, flat_pos])
        return pred.view(b, n, d)


__all__ = ["InContextTokensDynamics"]
