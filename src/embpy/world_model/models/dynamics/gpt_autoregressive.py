"""GPT-style autoregressive dynamics over (state, action) sequences.

Token layout (Decision-Transformer style):

    s_0 a_0 s_1 a_1 s_2 a_2 ... s_{T-1} a_{T-1}

The next-state prediction at timestep ``t`` is read off the action-token
output position ``2t + 1``: the causal mask ensures it has only seen
``s_{<=t}`` and ``a_{<=t}``, which is exactly the information needed to
predict ``s_{t+1}``.

This trades the paper's image-based ResNet+GRU dynamics for a clean
autoregressive transformer: the same machinery that makes Decision
Transformer work on Atari translates straightforwardly to perturbation
sequences, where the "trajectory" is a chain of perturbations applied
to a cell.
"""

from __future__ import annotations

import torch
from torch import nn

from ..blocks import TransformerBlock


class GPTAutoregressiveDynamics(nn.Module):
    """Causal transformer over interleaved state/action tokens.

    Parameters
    ----------
    d_model
        Token width.
    n_layers
        Number of transformer layers.
    n_heads
        Number of attention heads per layer.
    dropout
        Dropout probability.
    max_sequence_length
        Maximum number of (state+action) tokens in a single sequence
        (i.e. ``2 * T_max``). Sized once and used to allocate the
        positional embedding table and causal mask.
    use_action_token
        If False, the model becomes a state-only causal transformer.
        Useful as an ablation that quantifies how much the action
        carries the predictive signal.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_sequence_length: int = 64,
        use_action_token: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_action_token = use_action_token
        self.max_sequence_length = max_sequence_length

        self.pos_embed = nn.Parameter(torch.zeros(1, max_sequence_length, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # A learned token-type embedding lets the model distinguish
        # state slots from action slots without relying purely on
        # position parity.
        self.token_type_embed = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=4,
                    dropout=dropout,
                    causal=True,
                    max_sequence_length=max_sequence_length,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.next_state_head = nn.Linear(d_model, d_model)

    @staticmethod
    def _interleave(state_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        # state_tokens, action_tokens: (B, T, D)
        b, t, d = state_tokens.shape
        out = torch.empty(b, 2 * t, d, dtype=state_tokens.dtype, device=state_tokens.device)
        out[:, 0::2] = state_tokens
        out[:, 1::2] = action_tokens
        return out

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next-state tokens for every timestep.

        Parameters
        ----------
        state_tokens
            ``(B, T, d_model)`` state tokens.
        action_tokens
            ``(B, T, d_model)`` action tokens. May be ``None`` only when
            ``use_action_token=False`` was set at construction time.

        Returns
        -------
        torch.Tensor
            Predicted next-state tokens ``s_hat`` of shape
            ``(B, T, d_model)``. The ``t``-th slice is the model's
            prediction for ``s_{t+1}`` conditioned on
            ``s_{<=t}, a_{<=t}``.
        """
        if state_tokens.ndim != 3:
            raise ValueError(f"state_tokens must be (B, T, D), got {tuple(state_tokens.shape)}")
        b, t, d = state_tokens.shape
        if d != self.d_model:
            raise ValueError(f"d_model mismatch: got {d}, expected {self.d_model}")

        if self.use_action_token:
            if action_tokens is None:
                raise ValueError("action_tokens is required when use_action_token=True")
            if action_tokens.shape != state_tokens.shape:
                raise ValueError(
                    f"action_tokens shape {tuple(action_tokens.shape)} must match "
                    f"state_tokens shape {tuple(state_tokens.shape)}"
                )
            tokens = self._interleave(state_tokens, action_tokens)  # (B, 2T, D)
            type_ids = torch.tensor(
                [0, 1], device=state_tokens.device, dtype=torch.long
            ).repeat(t)  # (2T,)
        else:
            tokens = state_tokens
            type_ids = torch.zeros(t, device=state_tokens.device, dtype=torch.long)

        seq_len = tokens.shape[1]
        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}"
            )

        h = tokens + self.pos_embed[:, :seq_len] + self.token_type_embed(type_ids)[None]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        h = self.next_state_head(h)

        if self.use_action_token:
            # Read predictions off action positions: position 2t+1 has
            # attended to s_{<=t}, a_{<=t} -- exactly the information
            # required for s_{t+1}.
            s_hat = h[:, 1::2]
        else:
            # State-only sequences: position t has attended to s_{<=t},
            # so it predicts s_{t+1}.
            s_hat = h
        return s_hat


__all__ = ["GPTAutoregressiveDynamics"]
