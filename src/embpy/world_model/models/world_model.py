"""Composed perturbation world model.

Brings together the four moving parts:

* state encoder    -- (B, T, K, G) -> (B, T, d)
* action encoder   -- (B, T, n_pert) [long] -> (B, T, d)
* dynamics         -- (B, T, d) + (B, T, d) -> (B, T, d) (next-state tokens)
* decoder          -- (B, T, d) -> (B, T, G) (expression reconstruction)

All four are passed in at construction time so that swapping any one of
them is a one-line change in the training script. The compositional
class exposes the four operations the caller actually cares about:
:meth:`encode`, :meth:`predict_next`, :meth:`rollout` and :meth:`loss`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..training.losses import info_nce, latent_mse
from .action.gene_embedding_action import GeneEmbeddingAction
from .decoders.expression_decoder import ExpressionDecoder
from .dynamics.gpt_autoregressive import GPTAutoregressiveDynamics
from .encoders.backbones import (
    ForeignBackboneHead,
    LocalStackBackbone,
    StateBackboneProvider,
    build_backbone,
)
from .encoders.state_stack_encoder import StateStackEncoder


@dataclass
class WorldModelOutput:
    """Container for the per-step outputs produced by :meth:`WorldModel.forward`."""

    s_hat: torch.Tensor
    """Predicted next-state tokens, shape ``(B, T, d)``."""

    s_target: torch.Tensor
    """Encoded ground-truth next-state tokens, shape ``(B, T, d)``."""

    x_hat: torch.Tensor | None
    """Decoded gene-expression predictions, shape ``(B, T, G)`` or ``None``."""

    state_tokens: torch.Tensor
    """Encoded current state tokens used as dynamics input, shape ``(B, T, d)``."""

    action_tokens: torch.Tensor | None
    """Encoded action tokens used as dynamics input, shape ``(B, T, d)`` or ``None``."""


class WorldModel(nn.Module):
    """Composition of encoder, action encoder, dynamics and decoder.

    Parameters
    ----------
    encoder
        Implementation of :class:`StateStackEncoder`.
    action_encoder
        Maps ``(B, T, n_pert)`` long indices to ``(B, T, d_model)``.
    dynamics
        Autoregressive next-state predictor.
    decoder
        Optional decoder back to gene space. Pass ``None`` for a
        latent-only world model.

    Notes
    -----
    The ``forward`` method consumes a *batch dict* (the structure
    produced by :class:`world_model.data.PerturbationSequenceDataset`)
    so it slots cleanly into the trainer without an adapter layer.
    """

    def __init__(
        self,
        encoder: StateStackEncoder | ForeignBackboneHead,
        action_encoder: GeneEmbeddingAction,
        dynamics: GPTAutoregressiveDynamics,
        decoder: ExpressionDecoder | None = None,
        backbone: StateBackboneProvider | None = None,
    ) -> None:
        super().__init__()
        if encoder.d_model != dynamics.d_model:
            raise ValueError(
                f"encoder d_model ({encoder.d_model}) must match dynamics d_model ({dynamics.d_model})"
            )
        if action_encoder.d_model != dynamics.d_model:
            raise ValueError(
                f"action_encoder d_model ({action_encoder.d_model}) must match "
                f"dynamics d_model ({dynamics.d_model})"
            )
        if decoder is not None and decoder.d_model != dynamics.d_model:
            raise ValueError(
                f"decoder d_model ({decoder.d_model}) must match dynamics d_model ({dynamics.d_model})"
            )
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.d_model = dynamics.d_model
        # The backbone reference is held *outside* the autograd graph for
        # foreign backbones (STATE/STACK pre-encode in the dataloader), so
        # we attach it via object.__setattr__ rather than as a child module.
        # The local backbone's encoder IS self.encoder, so backbone here
        # is purely informational on that path.
        object.__setattr__(self, "backbone", backbone)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observation stacks into state tokens.

        Shape contract depends on the configured backbone:

        * ``kind='local'`` (default, pre-Phase-5 path):
          input  ``obs``  : ``(B, T, K, G)`` raw expression stack
          output state    : ``(B, T, d_model)``
          The encoder is :class:`StateStackEncoder` and consumes the
          raw expression matrix slice produced by the dataset.

        * ``kind='state'`` or ``kind='stack'``:
          input  ``obs``  : ``(B, T, K, embedding_dim)`` pre-encoded
                            cell embeddings produced by the foundation
                            backbone (the dataloader replaced the
                            in-memory expression matrix with the
                            cached embeddings; see
                            :func:`world_model.data.build_dataloaders`).
          output state    : ``(B, T, d_model)``
          The encoder is :class:`ForeignBackboneHead`, a tiny mean-pool
          + projection layer. Foundation backbones operate on AnnData,
          not torch tensors, so the foundation forward never lives on
          the per-batch hot path -- it runs once at dataloader build
          time and is cached on disk.

        Returns
        -------
        torch.Tensor
            State tokens ``(B, T, d_model)``.
        """
        return self.encoder(obs)

    def encode_action(self, gene_indices: torch.Tensor) -> torch.Tensor:
        """Encode a batch of action indices into ``(B, T, d_model)`` tokens."""
        return self.action_encoder(gene_indices)

    def predict_next(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the dynamics; return ``(B, T, d_model)`` next-state predictions."""
        return self.dynamics(state_tokens, action_tokens)

    def decode(self, s: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode state tokens back to gene-expression space.

        Raises :class:`RuntimeError` if no decoder was attached.
        """
        if self.decoder is None:
            raise RuntimeError("WorldModel was built without a decoder; cannot decode.")
        return self.decoder(s)

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> WorldModelOutput:
        """Run the full encode -> dynamics -> decode pipeline.

        Expected batch keys:

        * ``obs_stack``         -- ``(B, T, K, G)``
        * ``next_obs_stack``    -- ``(B, T, K, G)``
        * ``action_indices``    -- ``(B, T, n_pert)`` long
        * ``next_expression``   -- ``(B, T, G)`` (optional, decoder target)
        """
        obs = batch["obs_stack"]
        next_obs = batch["next_obs_stack"]
        actions = batch["action_indices"]

        state_tokens = self.encode(obs)
        next_state_tokens = self.encode(next_obs)
        action_tokens = self.encode_action(actions) if self.dynamics.use_action_token else None
        s_hat = self.predict_next(state_tokens, action_tokens)

        x_hat: torch.Tensor | None = None
        if self.decoder is not None:
            decoded = self.decode(s_hat)
            x_hat = decoded[0] if isinstance(decoded, tuple) else decoded

        return WorldModelOutput(
            s_hat=s_hat,
            s_target=next_state_tokens,
            x_hat=x_hat,
            state_tokens=state_tokens,
            action_tokens=action_tokens,
        )

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        latent_mse_weight: float = 1.0,
        decoder_mse_weight: float = 0.5,
        info_nce_weight: float = 0.0,
        info_nce_temperature: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine the latent / decoder / contrastive objectives.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor for ``loss.backward()``.
        components : dict[str, torch.Tensor]
            Per-term scalar tensors for logging / debugging.
        """
        out = self.forward(batch)
        components: dict[str, torch.Tensor] = {}

        # Targets do not need gradient -- detach to keep the encoder
        # honest on the dynamics path. (Equivalent to BYOL-style stop-grad
        # on the prediction target.)
        s_target = out.s_target.detach()
        loss_lat = latent_mse(out.s_hat, s_target)
        components["latent_mse"] = loss_lat
        total = latent_mse_weight * loss_lat

        if self.decoder is not None and "next_expression" in batch and decoder_mse_weight > 0.0:
            target = batch["next_expression"]
            if out.x_hat is None:
                raise RuntimeError("Decoder enabled but x_hat was not produced.")
            loss_dec = nn.functional.mse_loss(out.x_hat, target)
            components["decoder_mse"] = loss_dec
            total = total + decoder_mse_weight * loss_dec

        if info_nce_weight > 0.0:
            # Flatten time so each (B*T) position is its own query/key.
            b, t, d = out.s_hat.shape
            loss_nce = info_nce(
                out.s_hat.reshape(b * t, d),
                s_target.reshape(b * t, d),
                temperature=info_nce_temperature,
            )
            components["info_nce"] = loss_nce
            total = total + info_nce_weight * loss_nce

        components["total"] = total.detach()
        return total, components

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        init_obs_stack: torch.Tensor,
        action_indices_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Multi-step open-loop rollout.

        Starting from a single observation stack, applies a sequence of
        ``T`` actions and returns predicted next-state tokens (and
        decoded expression vectors when a decoder is attached). Each
        step feeds its own predicted token back into the dynamics; this
        is the autoregressive / "imagined" rollout used at evaluation
        time.

        Parameters
        ----------
        init_obs_stack
            ``(B, K, G)`` initial stack of observations (a single
            timestep, before any action).
        action_indices_seq
            ``(B, T, n_pert)`` long tensor of perturbation indices to
            apply, one perturbation per imagined step.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"s_hat": (B, T, d_model), "x_hat": (B, T, G) or None}``.
        """
        if init_obs_stack.ndim != 3:
            raise ValueError(
                f"init_obs_stack must be (B, K, G), got {tuple(init_obs_stack.shape)}"
            )
        if action_indices_seq.ndim != 3:
            raise ValueError(
                f"action_indices_seq must be (B, T, n_pert), got {tuple(action_indices_seq.shape)}"
            )

        b, k, g = init_obs_stack.shape
        t = action_indices_seq.shape[1]

        # We feed the first state token from the encoder, then keep
        # appending predicted state tokens. Action tokens come straight
        # from the table at every step.
        s0 = self.encode(init_obs_stack.unsqueeze(1))  # (B, 1, d)
        action_tokens_full = (
            self.encode_action(action_indices_seq) if self.dynamics.use_action_token else None
        )

        s_hist = s0
        s_hat_list: list[torch.Tensor] = []
        for step in range(t):
            if self.dynamics.use_action_token:
                a_hist = action_tokens_full[:, : step + 1]  # (B, step+1, d)
            else:
                a_hist = None
            s_hat_seq = self.predict_next(s_hist, a_hist)  # (B, step+1, d)
            s_next = s_hat_seq[:, -1:]  # last step's prediction
            s_hat_list.append(s_next)
            s_hist = torch.cat([s_hist, s_next], dim=1)

        s_hat = torch.cat(s_hat_list, dim=1)  # (B, T, d)

        x_hat: torch.Tensor | None = None
        if self.decoder is not None:
            decoded = self.decode(s_hat)
            x_hat = decoded[0] if isinstance(decoded, tuple) else decoded

        return {"s_hat": s_hat, "x_hat": x_hat}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, has_decoder={self.decoder is not None}"


def build_world_model(
    *,
    n_genes: int,
    gene_embedding_table: torch.Tensor,
    encoder_kind: str = "transformer",
    d_model: int = 256,
    stack_size: int = 4,
    encoder_layers: int = 2,
    encoder_heads: int = 4,
    dynamics_layers: int = 6,
    dynamics_heads: int = 8,
    dropout: float = 0.1,
    max_sequence_length: int = 64,
    use_action_token: bool = True,
    decoder_hidden_dims: tuple[int, ...] = (512, 1024),
    enable_decoder: bool = True,
    action_adapter_cfg: Any = None,
    state_backbone_cfg: Any = None,
    state_backbone_provider: StateBackboneProvider | None = None,
    state_backbone_embedding_dim: int | None = None,
    **_unused: Any,
) -> WorldModel:
    """One-call factory for the default architecture.

    Parameters
    ----------
    state_backbone_cfg
        ``StateBackboneConfig`` from :mod:`configs`. ``None`` (the
        legacy default) behaves exactly like the pre-Phase-5 path: a
        local :class:`StateStackEncoder` with the same hyperparameters.
        When set to ``kind='state' | 'stack'``, the dataloader is
        expected to have pre-encoded the cells through the foundation
        model (see :func:`world_model.data.build_dataloaders`); in that
        case ``state_backbone_provider`` and ``state_backbone_embedding_dim``
        should be passed in by the caller (the dataloader builds the
        provider once on rank 0).
    state_backbone_provider
        Provider instance owned by the caller (typically the
        dataloader). Only used when ``state_backbone_cfg.kind`` is
        ``'state'`` or ``'stack'`` to register a reference on the
        model and to source ``embedding_dim``.
    state_backbone_embedding_dim
        Override for the foreign backbone's embedding dim. Useful in
        tests so we don't have to actually load STATE / STACK weights.
    """
    from .encoders.state_stack_encoder import build_state_stack_encoder  # noqa: PLC0415

    kind = getattr(state_backbone_cfg, "kind", "local") if state_backbone_cfg is not None else "local"
    backbone: StateBackboneProvider | None = None
    encoder: StateStackEncoder | ForeignBackboneHead

    if kind == "local":
        encoder = build_state_stack_encoder(
            kind=encoder_kind,
            n_genes=n_genes,
            d_model=d_model,
            stack_size=stack_size,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            dropout=dropout,
        )
        if state_backbone_cfg is not None:
            backbone = LocalStackBackbone(encoder=encoder)
            if state_backbone_cfg.freeze:
                # The local backbone's encoder participates in the world
                # model's parameter list. Freezing it here mirrors the
                # foreign-backbone semantics (freeze means: out of the
                # optimizer) while leaving the dataloader code path
                # untouched.
                backbone.freeze()
    else:
        if state_backbone_provider is None:
            backbone = build_backbone(
                state_backbone_cfg,
                n_genes=n_genes,
                d_model=d_model,
                stack_size=stack_size,
                encoder_kind=encoder_kind,
                encoder_layers=encoder_layers,
                encoder_heads=encoder_heads,
                dropout=dropout,
            )
        else:
            backbone = state_backbone_provider
        emb_dim = state_backbone_embedding_dim
        if emb_dim is None:
            emb_dim = backbone.embedding_dim
        identity_when_dims_match = bool(
            state_backbone_cfg is not None and state_backbone_cfg.freeze
        )
        encoder = ForeignBackboneHead(
            embedding_dim=int(emb_dim),
            d_model=d_model,
            stack_size=stack_size,
            identity_when_dims_match=identity_when_dims_match,
        )

    action_encoder = GeneEmbeddingAction(
        gene_embedding_table=gene_embedding_table,
        d_model=d_model,
        pool="mean",
        freeze_embeddings=True,
        adapter_cfg=action_adapter_cfg,
    )
    dynamics = GPTAutoregressiveDynamics(
        d_model=d_model,
        n_layers=dynamics_layers,
        n_heads=dynamics_heads,
        dropout=dropout,
        max_sequence_length=max_sequence_length,
        use_action_token=use_action_token,
    )
    decoder = (
        ExpressionDecoder(
            d_model=d_model,
            n_genes=n_genes,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
        )
        if enable_decoder
        else None
    )
    return WorldModel(
        encoder=encoder,
        action_encoder=action_encoder,
        dynamics=dynamics,
        decoder=decoder,
        backbone=backbone,
    )


__all__ = ["WorldModel", "WorldModelOutput", "build_world_model"]
