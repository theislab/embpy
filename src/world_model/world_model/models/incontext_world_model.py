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
* ``dynamics``       -- in-context bidirectional dynamics
* ``decoder``        -- (B, d) latent -> (B, G) expression
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from world_model.training.losses import info_nce


class InContextWorldModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        action_encoder: nn.Module,
        query_action_encoder: nn.Module | None = None,
        dynamics: nn.Module,
        decoder: nn.Module | None,
        d_model: int,
        backbone: Any = None,
        default_support_size: int = 16,
        latent_normalization: str = "none",
        prediction_mode: str = "absolute",
    ) -> None:
        super().__init__()
        if latent_normalization not in {"none", "layer_norm", "l2"}:
            raise ValueError(
                "latent_normalization must be one of {'none', 'layer_norm', 'l2'}, "
                f"got {latent_normalization!r}."
            )
        if prediction_mode not in {"absolute", "residual_delta"}:
            raise ValueError(
                "prediction_mode must be one of {'absolute', 'residual_delta'}, "
                f"got {prediction_mode!r}."
            )
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
        self.latent_normalization = latent_normalization
        self.prediction_mode = prediction_mode
        self.latent_norm = (
            nn.LayerNorm(d_model, elementwise_affine=False)
            if latent_normalization == "layer_norm"
            else None
        )
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

    def _encode_context_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        """Encode an in-context batch into support/query latent tokens."""
        support_s = self._normalize_latent(self._encode_stacks(batch["support_obs"]))  # (B,M,d)
        support_sp = self._normalize_latent(self._encode_stacks(batch["support_next"]))  # (B,M,d)
        support_a = self._encode_actions(batch["support_act"])  # (B,M,d)
        query_s = self._normalize_latent(self._encode_one_stack(batch["query_obs"]))  # (B,d)
        query_a = self._encode_one_query_action(batch["query_act"])  # (B,d)
        s_target = (
            self._normalize_latent(self._encode_one_stack(batch["query_next"]))
            if "query_next" in batch
            else None
        )
        return {
            "support_s": support_s,
            "support_sp": support_sp,
            "support_a": support_a,
            "query_s": query_s,
            "query_a": query_a,
            "s_target": s_target,
        }

    def _normalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_normalization == "none":
            return x
        if self.latent_normalization == "layer_norm":
            if self.latent_norm is None:  # pragma: no cover - defensive
                raise RuntimeError("latent_norm module was not initialized.")
            return self.latent_norm(x)
        if self.latent_normalization == "l2":
            return F.normalize(x, dim=-1)
        raise RuntimeError(f"Unexpected latent_normalization={self.latent_normalization!r}")

    def _predict_from_encoded(
        self,
        support_s: torch.Tensor,
        support_a: torch.Tensor,
        support_sp: torch.Tensor,
        query_s: torch.Tensor,
        query_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.dynamics(
            support_s,
            support_a,
            support_sp,
            query_s,
            query_a,
        )
        if self.prediction_mode == "residual_delta":
            delta_hat = raw
            s_hat = query_s + delta_hat
        else:
            s_hat = raw
            delta_hat = s_hat - query_s
        return s_hat, delta_hat

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
        encoded = self._encode_context_batch(batch)
        support_s = encoded["support_s"]
        support_sp = encoded["support_sp"]
        support_a = encoded["support_a"]
        query_s = encoded["query_s"]
        query_a = encoded["query_a"]
        if (
            not isinstance(support_s, torch.Tensor)
            or not isinstance(support_sp, torch.Tensor)
            or not isinstance(support_a, torch.Tensor)
            or not isinstance(query_s, torch.Tensor)
            or not isinstance(query_a, torch.Tensor)
        ):  # pragma: no cover - defensive; helper always returns tensors here
            raise RuntimeError("Encoded in-context batch is missing required tensors.")

        s_hat, delta_hat = self._predict_from_encoded(
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
        s_target = encoded["s_target"]

        x_hat = self.decode(s_hat) if self.decoder is not None else None
        return {
            "s_hat": s_hat,
            "delta_hat": delta_hat,
            "s_target": s_target,
            "query_s": query_s,
            "query_a": query_a,
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
        out = self._encode_context_batch(batch)
        components: dict[str, torch.Tensor] = {}

        # BYOL-style stop-grad on the target (same as WorldModel).
        if out["s_target"] is None:
            raise ValueError("batch must include query_next when computing in-context loss.")
        support_s = out["support_s"]
        support_a = out["support_a"]
        support_sp = out["support_sp"]
        query_s = out["query_s"]
        query_a = out["query_a"]
        s_target_raw = out["s_target"]
        if (
            not isinstance(support_s, torch.Tensor)
            or not isinstance(support_a, torch.Tensor)
            or not isinstance(support_sp, torch.Tensor)
            or not isinstance(query_s, torch.Tensor)
            or not isinstance(query_a, torch.Tensor)
            or not isinstance(s_target_raw, torch.Tensor)
        ):  # pragma: no cover - defensive; validated above
            raise RuntimeError("Encoded in-context batch is missing required tensors.")

        query_s_ref = query_s.detach()
        s_target = s_target_raw.detach()

        contrast_pred_delta: torch.Tensor
        contrast_target_delta: torch.Tensor
        if hasattr(self.dynamics, "forward_autoregressive"):
            raw_all = self.dynamics.forward_autoregressive(
                support_s,
                support_a,
                support_sp,
                query_s,
                query_a,
                s_target_raw,
            )
            states_all = torch.cat([support_s, query_s.unsqueeze(1)], dim=1)
            targets_all_raw = torch.cat([support_sp, s_target_raw.unsqueeze(1)], dim=1)
            states_all_ref = states_all.detach()
            targets_all = targets_all_raw.detach()
            if self.prediction_mode == "residual_delta":
                delta_hat_all = raw_all
                s_hat_all = states_all + delta_hat_all
            else:
                s_hat_all = raw_all
                delta_hat_all = s_hat_all - states_all_ref
            target_delta_all = targets_all - states_all_ref

            s_hat = s_hat_all[:, -1]
            delta_hat = delta_hat_all[:, -1]
            loss_lat = F.mse_loss(s_hat_all, targets_all)
            loss_delta = F.mse_loss(delta_hat_all, target_delta_all)
            pred_delta = delta_hat if self.prediction_mode == "residual_delta" else s_hat - query_s_ref
            target_delta = s_target - query_s_ref
            contrast_pred_delta = delta_hat_all.reshape(-1, delta_hat_all.size(-1))
            contrast_target_delta = target_delta_all.reshape(-1, target_delta_all.size(-1))
            components["autoregressive_targets"] = torch.as_tensor(
                raw_all.shape[1],
                dtype=s_hat.dtype,
                device=s_hat.device,
            )
            x_hat = self.decode(s_hat) if self.decoder is not None else None
        else:
            s_hat, delta_hat = self._predict_from_encoded(
                support_s,
                support_a,
                support_sp,
                query_s,
                query_a,
            )
            pred_delta = delta_hat if self.prediction_mode == "residual_delta" else s_hat - query_s_ref
            target_delta = s_target - query_s_ref
            loss_lat = F.mse_loss(s_hat, s_target)
            loss_delta = F.mse_loss(delta_hat, target_delta)
            contrast_pred_delta = pred_delta
            contrast_target_delta = target_delta
            x_hat = self.decode(s_hat) if self.decoder is not None else None

        latent_objective = loss_delta if self.prediction_mode == "residual_delta" else loss_lat
        components["latent_mse"] = loss_lat
        components["delta_mse"] = loss_delta
        components["latent_objective"] = latent_objective
        total = latent_mse_weight * latent_objective

        with torch.no_grad():
            pred_norm = contrast_pred_delta.norm(dim=-1).mean()
            target_norm = contrast_target_delta.norm(dim=-1).mean()
            components["delta_norm"] = pred_norm
            components["target_delta_norm"] = target_norm
            components["delta_norm_ratio"] = pred_norm / target_norm.clamp_min(1e-8)
            components["delta_dim_var"] = contrast_pred_delta.var(dim=0, unbiased=False).mean()
            components["target_delta_dim_var"] = contrast_target_delta.var(dim=0, unbiased=False).mean()
            components["s_hat_dim_var"] = s_hat.var(dim=0, unbiased=False).mean()
            components["target_s_dim_var"] = s_target.var(dim=0, unbiased=False).mean()
            components["query_s_dim_var"] = query_s_ref.var(dim=0, unbiased=False).mean()
            components["s_hat_norm"] = s_hat.norm(dim=-1).mean()
            components["target_s_norm"] = s_target.norm(dim=-1).mean()

        if self.decoder is not None and "query_next_expression" in batch and decoder_mse_weight > 0.0:
            tgt = batch["query_next_expression"]
            if x_hat is None:
                raise RuntimeError("Decoder enabled but x_hat was not produced.")
            loss_dec = F.mse_loss(x_hat, tgt)
            components["decoder_mse"] = loss_dec
            total = total + decoder_mse_weight * loss_dec

        if info_nce_weight > 0.0:
            # Contrast perturbation effects, not absolute states. In
            # these cell-state spaces absolute latents can all have very
            # high cosine similarity; subtracting the query/control
            # state asks InfoNCE to match "what changed under this
            # perturbation", preserving the in-context prediction
            # principle for both within- and cross-modality runs.
            loss_nce = info_nce(contrast_pred_delta, contrast_target_delta, temperature=info_nce_temperature)
            components["info_nce"] = loss_nce
            total = total + info_nce_weight * loss_nce
            with torch.no_grad():
                p = F.normalize(contrast_pred_delta, dim=-1)
                t = F.normalize(contrast_target_delta, dim=-1)
                sim = p @ t.T
                eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                pos_sim = sim[eye].mean()
                neg_sim = sim[~eye].mean() if bool((~eye).any()) else torch.zeros((), device=sim.device)
                pred_self = p @ p.T
                target_self = t @ t.T
                components["pos_sim"] = pos_sim
                components["neg_sim"] = neg_sim
                components["pos_minus_neg"] = pos_sim - neg_sim
                components["pred_delta_self_offdiag"] = (
                    pred_self[~eye].mean() if bool((~eye).any()) else torch.zeros((), device=sim.device)
                )
                components["target_delta_self_offdiag"] = (
                    target_self[~eye].mean() if bool((~eye).any()) else torch.zeros((), device=sim.device)
                )

        if action_counterfactual_weight > 0.0:
            # Re-predict with the query action permuted across the batch
            # (swap which perturbation each task is asked about). The
            # real-action prediction must be closer to the target than
            # the counterfactual one -- forces action conditioning.
            perm = torch.randperm(s_hat.size(0), device=s_hat.device)
            cf_a = self._encode_one_query_action(batch["query_act"][perm])
            s_hat_cf, cf_delta_hat = self._predict_from_encoded(
                support_s,
                support_a,
                support_sp,
                query_s,
                cf_a,
            )
            tau = max(action_counterfactual_temperature, 1e-8)
            real_delta = pred_delta
            cf_delta = cf_delta_hat if self.prediction_mode == "residual_delta" else s_hat_cf - query_s_ref
            p = F.normalize(real_delta, dim=-1)
            cf = F.normalize(cf_delta, dim=-1)
            tn = F.normalize(target_delta, dim=-1)
            sim_real = (p * tn).sum(-1) / tau
            sim_cf = (cf * tn).sum(-1) / tau
            cf_logits = torch.stack([sim_real, sim_cf], dim=-1)
            cf_labels = torch.zeros(cf_logits.size(0), dtype=torch.long, device=cf_logits.device)
            loss_cf = F.cross_entropy(cf_logits, cf_labels)
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
