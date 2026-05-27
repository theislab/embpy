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

from world_model.models.action.gene_embedding_action import GeneEmbeddingAction
from world_model.models.decoders.expression_decoder import ExpressionDecoder
from world_model.models.dynamics.gpt_autoregressive import GPTAutoregressiveDynamics
from world_model.models.encoders.backbones import (
    ForeignBackboneHead,
    LocalBackbone,
    StateBackboneProvider,
    build_backbone,
)
from world_model.models.encoders.state_stack_encoder import StateStackEncoder
from world_model.training.losses import info_nce, latent_mse


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
            raise ValueError(f"encoder d_model ({encoder.d_model}) must match dynamics d_model ({dynamics.d_model})")
        if action_encoder.d_model != dynamics.d_model:
            raise ValueError(
                f"action_encoder d_model ({action_encoder.d_model}) must match dynamics d_model ({dynamics.d_model})"
            )
        if decoder is not None and decoder.d_model != dynamics.d_model:
            raise ValueError(f"decoder d_model ({decoder.d_model}) must match dynamics d_model ({dynamics.d_model})")
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

        # Consume integer action indices before the float-heavy state path.
        # This is equivalent mathematically and avoids flaky Apple MPS
        # corruption of long index tensors after unrelated kernels run.
        action_tokens = self.encode_action(actions) if self.dynamics.use_action_token else None
        state_tokens = self.encode(obs)
        next_state_tokens = self.encode(next_obs)
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
        info_nce_mask_same_pert: bool = True,
        action_counterfactual_weight: float = 0.0,
        action_counterfactual_temperature: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine the latent / decoder / contrastive objectives.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor for ``loss.backward()``.
        components : dict[str, torch.Tensor]
            Per-term scalar tensors for logging / debugging.
        """
        valid_neg_mask: torch.Tensor | None = None
        cf_action_tokens: torch.Tensor | None = None
        if "action_indices" in batch:
            actions = batch["action_indices"]
            # Keep small integer action metadata on CPU for permutation and
            # same-perturbation bookkeeping. Apple MPS has flaky long-tensor
            # advanced-index behavior here; move back only for embedding.
            actions_cpu = actions.detach().cpu()
            if info_nce_weight > 0.0 and info_nce_mask_same_pert:
                b_actions, t_actions = actions_cpu.shape[:2]
                actions_flat = actions_cpu.reshape(b_actions * t_actions, -1).contiguous()
                sorted_actions, _ = actions_flat.sort(dim=-1)
                same_pert = (sorted_actions.unsqueeze(0) == sorted_actions.unsqueeze(1)).all(dim=-1)
                n = same_pert.shape[0]
                eye = torch.eye(n, dtype=torch.bool, device=same_pert.device)
                valid_neg_mask = ((~same_pert) | eye).to(actions.device)
            if action_counterfactual_weight > 0.0:
                b_orig = actions_cpu.shape[0]
                perm = torch.randperm(b_orig)
                cf_actions = actions_cpu[perm]
                cf_action_tokens = self.encode_action(cf_actions)

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
            pred_flat = out.s_hat.reshape(b * t, d)
            target_flat = s_target.reshape(b * t, d)

            # Option 3 -- hard-negative mining via same-perturbation mask.
            # Two items share a perturbation when their action_indices
            # vectors are element-wise equal up to ordering. We compare
            # sorted indices so e.g. (TP53, MYC) and (MYC, TP53) collapse.
            # The mask is True where the negative is *kept*; same-pert
            # off-diagonals are False (excluded from the softmax). The
            # diagonal stays True so positives are preserved.
            if valid_neg_mask is not None:
                # Fraction of off-diagonal entries kept as negatives.
                # 1.0 means no masking applied; < 1.0 means same-pert
                # negatives were filtered out.
                n = valid_neg_mask.shape[0]
                n_off = n * (n - 1)
                kept_off = valid_neg_mask.sum() - n
                components["info_nce_neg_kept_frac"] = kept_off.float() / max(float(n_off), 1.0)

            loss_nce = info_nce(
                pred_flat,
                target_flat,
                temperature=info_nce_temperature,
                valid_negative_mask=valid_neg_mask.to(pred_flat.device) if valid_neg_mask is not None else None,
            )
            components["info_nce"] = loss_nce
            total = total + info_nce_weight * loss_nce

            # Diagnostics for collapse / no-signal failure modes. Computed
            # under no_grad so they don't perturb training. Logged via the
            # standard components dict so existing log/csv hooks pick them
            # up. Reading guide:
            #   pos_sim ~ 1.0 AND neg_sim ~ 1.0  -> embedding collapse
            #   pos_sim ~ neg_sim                -> no signal (stuck at ln(N))
            #   pos_sim - neg_sim large          -> InfoNCE actually working
            #   s_hat_dim_var ~ 0                -> latents are constant -> collapse
            #   s_hat_norm wildly varying        -> scale instability
            with torch.no_grad():
                pred_n = nn.functional.normalize(pred_flat, dim=-1)
                target_n = nn.functional.normalize(target_flat, dim=-1)
                sim = pred_n @ target_n.T
                n = sim.shape[0]
                diag_mask = torch.eye(n, dtype=torch.bool, device=sim.device)
                pos_sim = sim[diag_mask].mean()
                neg_sim = sim[~diag_mask].mean()
                self_sim = pred_n @ pred_n.T
                self_off = self_sim[~diag_mask].mean()
                components["pos_sim"] = pos_sim
                components["neg_sim"] = neg_sim
                components["pos_minus_neg"] = pos_sim - neg_sim
                components["s_hat_self_offdiag"] = self_off
                components["s_hat_norm"] = pred_flat.norm(dim=-1).mean()
                # Per-dim variance averaged across dims; ~0 means latents
                # are nearly constant across the batch (collapse).
                components["s_hat_dim_var"] = pred_flat.var(dim=0).mean()

        if action_counterfactual_weight > 0.0 and cf_action_tokens is not None:
            # Option 1 -- counterfactual-action contrast. Re-run dynamics
            # with action indices permuted across the batch dim. The
            # real-action prediction should be more similar to s_target
            # than the counterfactual-action prediction is; if it isn't,
            # the dynamics module is ignoring the action token.
            s_hat_cf = self.predict_next(out.state_tokens, cf_action_tokens)

            b_, t_, d_ = s_hat_cf.shape
            pred_flat = out.s_hat.reshape(b_ * t_, d_)
            cf_flat = s_hat_cf.reshape(b_ * t_, d_)
            tgt_flat = s_target.reshape(b_ * t_, d_)
            pred_n = nn.functional.normalize(pred_flat, dim=-1)
            cf_n = nn.functional.normalize(cf_flat, dim=-1)
            tgt_n = nn.functional.normalize(tgt_flat, dim=-1)
            tau = max(action_counterfactual_temperature, 1e-8)
            sim_real = (pred_n * tgt_n).sum(dim=-1) / tau
            sim_cf = (cf_n * tgt_n).sum(dim=-1) / tau
            # Binary contrast over (real, counterfactual); class 0 = real.
            cf_logits = torch.stack([sim_real, sim_cf], dim=-1)
            cf_labels = torch.zeros(cf_logits.size(0), dtype=torch.long, device=cf_logits.device)
            loss_cf = nn.functional.cross_entropy(cf_logits, cf_labels)
            components["action_counterfactual"] = loss_cf
            with torch.no_grad():
                # Pre-temperature similarities for human-readable logs.
                components["cf_sim_real"] = sim_real.mean() * tau
                components["cf_sim_counter"] = sim_cf.mean() * tau
                components["cf_real_minus_counter"] = (sim_real - sim_cf).mean() * tau
            total = total + action_counterfactual_weight * loss_cf

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
            raise ValueError(f"init_obs_stack must be (B, K, G), got {tuple(init_obs_stack.shape)}")
        if action_indices_seq.ndim != 3:
            raise ValueError(f"action_indices_seq must be (B, T, n_pert), got {tuple(action_indices_seq.shape)}")

        b, k, g = init_obs_stack.shape
        t = action_indices_seq.shape[1]

        # We feed the first state token from the encoder, then keep
        # appending predicted state tokens. Action tokens come straight
        # from the table at every step.
        s0 = self.encode(init_obs_stack.unsqueeze(1))  # (B, 1, d)
        action_tokens_full = self.encode_action(action_indices_seq) if self.dynamics.use_action_token else None

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
        """Return the number of model parameters."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        """Return the compact ``nn.Module`` representation string."""
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
    dynamics_kind: str = "gpt",
    incontext_support_size: int = 16,
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
        ``StateBackboneConfig`` from :mod:`configs`. Training tensors
        are always pre-attached AnnData ``.obsm`` rows. ``kind='local'``
        uses the normal stack encoder over those rows; ``kind='state'``
        or ``'stack'`` uses a small foreign-embedding head and requires
        ``state_backbone_embedding_dim`` unless a provider is explicitly
        passed for backwards-compatible tests.
    state_backbone_provider
        Provider instance owned by the caller (typically the
        dataloader). Only used when ``state_backbone_cfg.kind`` is
        ``'state'`` or ``'stack'`` to register a reference on the
        model and to source ``embedding_dim``.
    state_backbone_embedding_dim
        Override for the foreign backbone's embedding dim. Useful in
        tests so we don't have to actually load STATE / STACK weights.
    """
    from .encoders.state_stack_encoder import build_state_stack_encoder

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
            backbone = LocalBackbone(encoder=encoder)
            if state_backbone_cfg.freeze:
                # The local backbone's encoder participates in the world
                # model's parameter list. Freezing it here mirrors the
                # foreign-backbone semantics (freeze means: out of the
                # optimizer) while leaving the dataloader code path
                # untouched.
                backbone.freeze()
    else:
        if state_backbone_provider is None and state_backbone_embedding_dim is None:
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
        elif state_backbone_provider is not None:
            backbone = state_backbone_provider
        emb_dim = state_backbone_embedding_dim
        if emb_dim is None and backbone is not None:
            emb_dim = backbone.embedding_dim
        if emb_dim is None:
            raise ValueError(
                "state_backbone.kind in {'state', 'stack'} requires "
                "state_backbone_embedding_dim when consuming pre-attached "
                "adata.obsm state embeddings."
            )
        identity_when_dims_match = bool(state_backbone_cfg is not None and state_backbone_cfg.freeze)
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

    if dynamics_kind in ("incontext_set", "incontext_tokens"):
        # Bidirectional in-context models: predict a held-out query
        # perturbation from a SET of (control, action, perturbed)
        # support triplets. Non-causal, permutation-invariant.
        #   * incontext_set    -> one fused token per triplet
        #   * incontext_tokens -> explicit 3 tokens (s, a, s') per triplet
        from .incontext_world_model import InContextWorldModel

        max_set = max(int(incontext_support_size) + 1, max_sequence_length)
        if dynamics_kind == "incontext_tokens":
            from .dynamics.incontext_tokens import InContextTokensDynamics

            dynamics = InContextTokensDynamics(
                d_model=d_model,
                n_layers=dynamics_layers,
                n_heads=dynamics_heads,
                dropout=dropout,
                max_set_size=max_set,
            )
        else:
            from .dynamics.incontext_set import InContextSetDynamics

            dynamics = InContextSetDynamics(
                d_model=d_model,
                n_layers=dynamics_layers,
                n_heads=dynamics_heads,
                dropout=dropout,
                max_set_size=max_set,
            )
        return InContextWorldModel(
            encoder=encoder,
            action_encoder=action_encoder,
            dynamics=dynamics,
            decoder=decoder,
            d_model=d_model,
            backbone=backbone,
            default_support_size=int(incontext_support_size),
        )

    dynamics = GPTAutoregressiveDynamics(
        d_model=d_model,
        n_layers=dynamics_layers,
        n_heads=dynamics_heads,
        dropout=dropout,
        max_sequence_length=max_sequence_length,
        use_action_token=use_action_token,
    )
    return WorldModel(
        encoder=encoder,
        action_encoder=action_encoder,
        dynamics=dynamics,
        decoder=decoder,
        backbone=backbone,
    )


__all__ = ["WorldModel", "WorldModelOutput", "build_world_model"]
