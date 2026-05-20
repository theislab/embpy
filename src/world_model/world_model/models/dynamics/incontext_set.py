"""In-context, permutation-invariant set dynamics.

This is the alternative to :class:`GPTAutoregressiveDynamics` for the
``dynamics.kind = "incontext_set"`` path. Instead of a causal rollout
over a fabricated ``control -> A -> B`` trajectory, one *task* is:

    support = { (s_control, a_i, s_pert_i) }_{i=1..M}      (visible)
    query   =   (s_control, a_q,    ?    )                 (predict s')

Each triplet is fused into a single token. The support tokens carry
their real perturbed state ``s'``; the query token carries a learned
``[MASK]`` in the ``s'`` slot. A *bidirectional* (non-causal)
transformer attends over the whole set -- order does not matter, so no
positional embeddings are used -- and the query token's output is the
predicted query latent ``s'_hat``.

Why bidirectional is correct here (unlike the causal model): the
target ``s'`` of the query is NOT a token in the input (it is masked),
so full self-attention over the set leaks no label -- it is exactly
the BERT-style masked-prediction-over-a-set setup the user asked for.
"""

from __future__ import annotations

import torch
from torch import nn

from ..blocks import TransformerBlock


class InContextSetDynamics(nn.Module):
    """Bidirectional set transformer over fused ``(s, a, s')`` triplets.

    Parameters
    ----------
    d_model
        Width of the state / action tokens (encoder & action-encoder
        output dim).
    n_layers, n_heads, dropout
        Transformer stack hyper-parameters.
    max_set_size
        Upper bound on ``M + 1`` (support + query) tokens; sizes the
        attention buffers. The set has no order so there is no
        positional embedding -- this is only a safety cap.
    """

    # Kept for interface-compatibility with the GPT dynamics: callers
    # (WorldModel.forward / loss) branch on ``use_action_token``. The
    # in-context model always consumes the action (it is part of every
    # triplet), so this is always True.
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

        # Fuse a (state, action, next-state) triplet -> one token.
        self.triplet_fuse = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Learned placeholder for the query's unknown s'.
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        # Role tag: 0 = support triplet, 1 = query triplet. Permutation
        # invariant across the support set (all share role 0); only
        # tells the stack which token must be answered.
        self.role_embed = nn.Embedding(2, d_model)

        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                causal=False,  # set -> bidirectional, no leakage (s' masked)
                max_sequence_length=max_set_size,
            )
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    def _fuse(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sp: torch.Tensor,
    ) -> torch.Tensor:
        """``s,a,sp``: ``(..., d)`` -> fused token ``(..., d)``."""
        return self.triplet_fuse(torch.cat([s, a, sp], dim=-1))

    def forward(
        self,
        support_s: torch.Tensor,   # (B, M, d)
        support_a: torch.Tensor,   # (B, M, d)
        support_sp: torch.Tensor,  # (B, M, d)
        query_s: torch.Tensor,     # (B, d)
        query_a: torch.Tensor,     # (B, d)
    ) -> torch.Tensor:
        """Predict the query's perturbed latent ``s'_hat`` ``(B, d)``."""
        if support_s.ndim != 3:
            raise ValueError(
                f"support_s must be (B, M, d), got {tuple(support_s.shape)}"
            )
        b, m, d = support_s.shape
        if m + 1 > self.max_set_size:
            raise ValueError(
                f"set size {m + 1} exceeds max_set_size {self.max_set_size}"
            )

        support_tok = self._fuse(support_s, support_a, support_sp)  # (B,M,d)
        mask = self.mask_token.expand(b, d)
        query_tok = self._fuse(query_s, query_a, mask).unsqueeze(1)  # (B,1,d)

        tokens = torch.cat([support_tok, query_tok], dim=1)  # (B, M+1, d)

        roles = torch.zeros(m + 1, dtype=torch.long, device=tokens.device)
        roles[-1] = 1  # last token is the query
        tokens = tokens + self.role_embed(roles)[None]  # no positional embed

        h = tokens
        for layer in self.layers:
            h = layer(h)  # bidirectional (causal=False)
        h = self.norm(h)
        # The query token sits at the last position by construction.
        return self.head(h[:, -1])  # (B, d)


__all__ = ["InContextSetDynamics"]
