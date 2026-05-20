"""Rollout utilities for evaluating the world model end-to-end.

Wraps :meth:`WorldModel.rollout` with the metrics from
:mod:`world_model.evaluation.metrics` so callers can run a full
evaluation in a single function.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..models.world_model import WorldModel
from .metrics import cosine_similarity, delta_pearson, expression_r2, latent_l2_error

logger = logging.getLogger(__name__)


@torch.no_grad()
def imagined_rollout(
    model: WorldModel,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run open-loop rollouts on every batch of ``loader`` and aggregate metrics.

    For each batch:

    1. Take ``obs_stack[:, 0]`` as the initial state stack.
    2. Apply the full ``action_indices`` sequence in autoregressive mode.
    3. Compare predicted state tokens (and decoded expression) against
       the encoder-generated targets.

    Returns
    -------
    dict[str, float]
        Dictionary of aggregated metric names to scalar values.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval().to(device)

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def _accumulate(name: str, value: torch.Tensor, n: int) -> None:
        sums[name] = sums.get(name, 0.0) + float(value) * n
        counts[name] = counts.get(name, 0) + n

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        init_obs = batch["obs_stack"][:, 0]  # (B, K, G)
        action_seq = batch["action_indices"]  # (B, T, n_pert)

        rollout_out = model.rollout(init_obs, action_seq)
        s_hat = rollout_out["s_hat"]  # (B, T, d)
        x_hat = rollout_out["x_hat"]  # (B, T, G) or None

        s_target = model.encode(batch["next_obs_stack"])
        bsz = batch["obs_stack"].size(0)

        _accumulate("latent_l2", latent_l2_error(s_hat, s_target), bsz)
        _accumulate("latent_cos", cosine_similarity(s_hat, s_target), bsz)

        if x_hat is not None and "next_expression" in batch:
            x_target = batch["next_expression"]
            x_basal = batch["obs_stack"].mean(dim=2)  # (B, T, G)
            _accumulate("expression_r2", expression_r2(x_hat, x_target), bsz)
            _accumulate("delta_pearson", delta_pearson(x_basal, x_hat, x_target), bsz)

    metrics = {k: sums[k] / max(counts[k], 1) for k in sums}
    logger.info("Rollout metrics: %s", metrics)
    return metrics


__all__ = ["imagined_rollout"]
