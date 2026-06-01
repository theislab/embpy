"""In-context world model: predict a held-out perturbation's state from
a SET of (control, action, perturbed) support triplets.

Drop-in sibling of :class:`world_model.models.world_model.WorldModel`
that the trainer can use unchanged: it exposes the same
``loss(batch, **weights) -> (scalar, components)`` contract and
``num_parameters()``. The dataset must run in
``data.context_mode = "incontext_set"`` so batches carry the
``support_* / query_*`` keys.

Reuses the existing building blocks:

* ``encoder``        -- (B, n, K, G) cell stacks -> (B, n, d) state tokens
* ``action_encoder`` -- (B, n, n_pert) gene ids   -> (B, n, d) action tokens
* ``dynamics``       -- :class:`InContextSetDynamics` (bidirectional)
* ``decoder``        -- (B, d) latent -> (B, G) expression
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from world_model.training.losses import info_nce

from .dynamics.incontext_set import InContextSetDynamics


class InContextWorldModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        action_encoder: nn.Module,
        query_action_encoder: nn.Module | None = None,
        dynamics: InContextSetDynamics,
        decoder: nn.Module | None,
        d_model: int,
        backbone: Any = None,
        default_support_size: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.query_action_encoder = query_action_encoder or action_encoder
        for name, module in (
            ("action_encoder", self.action_encoder),
            ("query_action_encoder", self.query_action_encoder),
        ):
            module_d_model = getattr(module, "d_model", d_model)
            if int(module_d_model) != int(d_model):
                raise ValueError(f"{name} d_model ({module_d_model}) must match dynamics d_model ({d_model}).")
        self.dynamics = dynamics
        self.decoder = decoder
        self.d_model = int(d_model)
        # Support-set size the eval adapter uses when building tasks for
        # held-out query perturbations (mirrors data.incontext_support_size).
        self.default_support_size = int(default_support_size)
        # Mirror WorldModel: keep a non-parameter ref to the backbone.
        object.__setattr__(self, "backbone", backbone)

    # ------------------------------------------------------------------
    # Encoding helpers (same contracts as WorldModel.encode*)
    # ------------------------------------------------------------------

    def _encode_stacks(self, obs: torch.Tensor) -> torch.Tensor:
        """``(B, n, K, G) -> (B, n, d)``."""
        return self.encoder(obs)

    def _encode_one_stack(self, obs: torch.Tensor) -> torch.Tensor:
        """``(B, K, G) -> (B, d)`` (single stack per item)."""
        return self.encoder(obs.unsqueeze(1)).squeeze(1)

    def _encode_actions(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, n, n_pert) -> (B, n, d)``."""
        return self.action_encoder(idx)

    def _encode_one_action(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, n_pert) -> (B, d)``."""
        return self.action_encoder(idx.unsqueeze(1)).squeeze(1)

    def _encode_one_query_action(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, n_pert) -> (B, d)`` using the query action table."""
        return self.query_action_encoder(idx.unsqueeze(1)).squeeze(1)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("InContextWorldModel built without a decoder.")
        out = self.decoder(s)
        return out[0] if isinstance(out, tuple) else out

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode the task and predict the query's perturbed latent.

        Returns ``{"s_hat", "s_target", "query_s", "x_hat"}``
        (x_hat None if no decoder). ``s_target`` is the encoded real
        query perturbed state (used as the latent regression target).
        ``query_s`` is returned so contrastive losses can compare
        perturbation effects relative to the same pre-perturbation
        state, rather than contrasting absolute state embeddings.
        """
        support_s = self._encode_stacks(batch["support_obs"])  # (B,M,d)
        support_sp = self._encode_stacks(batch["support_next"])  # (B,M,d)
        support_a = self._encode_actions(batch["support_act"])  # (B,M,d)
        query_s = self._encode_one_stack(batch["query_obs"])  # (B,d)
        query_a = self._encode_one_query_action(batch["query_act"])  # (B,d)

        s_hat = self.dynamics(
            support_s,
            support_a,
            support_sp,
            query_s,
            query_a,
        )  # (B,d)
        # The target is only present at train/val time. At inference
        # ``query_next`` is deliberately absent (the answer is what we
        # predict) -- encode it only when given, so .predict() works for
        # both loss computation and held-out inference.
        s_target = self._encode_one_stack(batch["query_next"]) if "query_next" in batch else None

        x_hat = self.decode(s_hat) if self.decoder is not None else None
        return {
            "s_hat": s_hat,
            "s_target": s_target,
            "query_s": query_s,
            "support_s": support_s,
            "support_sp": support_sp,
            "support_a": support_a,
            "x_hat": x_hat,
        }

    # ------------------------------------------------------------------
    # Loss (same call signature the trainer uses for WorldModel)
    # ------------------------------------------------------------------

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
        out = self.predict(batch)
        components: dict[str, torch.Tensor] = {}

        # BYOL-style stop-grad on the target (same as WorldModel).
        s_hat = out["s_hat"]
        s_target = out["s_target"].detach()
        query_s_ref = out["query_s"].detach()
        loss_lat = nn.functional.mse_loss(s_hat, s_target)
        components["latent_mse"] = loss_lat
        total = latent_mse_weight * loss_lat

        if self.decoder is not None and "query_next_expression" in batch and decoder_mse_weight > 0.0:
            tgt = batch["query_next_expression"]
            loss_dec = nn.functional.mse_loss(out["x_hat"], tgt)
            components["decoder_mse"] = loss_dec
            total = total + decoder_mse_weight * loss_dec

        if info_nce_weight > 0.0:
            # Contrast perturbation effects, not absolute states. In
            # these cell-state spaces absolute latents can all have very
            # high cosine similarity; subtracting the query/control
            # state asks InfoNCE to match "what changed under this
            # perturbation", preserving the in-context prediction
            # principle for both within- and cross-modality runs.
            pred_delta = s_hat - query_s_ref
            target_delta = s_target - query_s_ref
            loss_nce = info_nce(pred_delta, target_delta, temperature=info_nce_temperature)
            components["info_nce"] = loss_nce
            total = total + info_nce_weight * loss_nce
            with torch.no_grad():
                p = nn.functional.normalize(pred_delta, dim=-1)
                t = nn.functional.normalize(target_delta, dim=-1)
                sim = p @ t.T
                eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                pos_sim = sim[eye].mean()
                neg_sim = sim[~eye].mean()
                pred_self = p @ p.T
                target_self = t @ t.T
                components["pos_sim"] = pos_sim
                components["neg_sim"] = neg_sim
                components["pos_minus_neg"] = pos_sim - neg_sim
                components["pred_delta_self_offdiag"] = pred_self[~eye].mean()
                components["target_delta_self_offdiag"] = target_self[~eye].mean()
                components["delta_norm"] = pred_delta.norm(dim=-1).mean()
                components["delta_dim_var"] = pred_delta.var(dim=0).mean()
                components["s_hat_dim_var"] = s_hat.var(dim=0).mean()

        if action_counterfactual_weight > 0.0:
            # Re-predict with the query action permuted across the batch
            # (swap which perturbation each task is asked about). The
            # real-action prediction must be closer to the target than
            # the counterfactual one -- forces action conditioning.
            perm = torch.randperm(s_hat.size(0), device=s_hat.device)
            cf_a = self._encode_one_query_action(batch["query_act"][perm])
            s_hat_cf = self.dynamics(
                out["support_s"],
                out["support_a"],
                out["support_sp"],
                out["query_s"],
                cf_a,
            )
            tau = max(action_counterfactual_temperature, 1e-8)
            real_delta = s_hat - query_s_ref
            cf_delta = s_hat_cf - query_s_ref
            target_delta = s_target - query_s_ref
            p = nn.functional.normalize(real_delta, dim=-1)
            cf = nn.functional.normalize(cf_delta, dim=-1)
            tn = nn.functional.normalize(target_delta, dim=-1)
            sim_real = (p * tn).sum(-1) / tau
            sim_cf = (cf * tn).sum(-1) / tau
            cf_logits = torch.stack([sim_real, sim_cf], dim=-1)
            cf_labels = torch.zeros(cf_logits.size(0), dtype=torch.long, device=cf_logits.device)
            loss_cf = nn.functional.cross_entropy(cf_logits, cf_labels)
            components["action_counterfactual"] = loss_cf
            total = total + action_counterfactual_weight * loss_cf

        return total, components

    # ------------------------------------------------------------------
    # Misc (parity with WorldModel surface the trainer/scripts touch)
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, has_decoder={self.decoder is not None}"


__all__ = ["InContextWorldModel"]
