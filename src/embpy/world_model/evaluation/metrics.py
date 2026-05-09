"""Numerical evaluation metrics for the world model.

All metrics operate on torch tensors and return scalars (or per-sample
vectors when ``reduction="none"``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_l2_error(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Mean Euclidean distance between predicted and target latents.

    Standard "how far is the predicted next-state from the true one"
    metric, in the same units as the latent space.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    err = (pred - target).pow(2).sum(dim=-1).sqrt()
    if reduction == "mean":
        return err.mean()
    if reduction == "sum":
        return err.sum()
    if reduction == "none":
        return err
    raise ValueError(f"Invalid reduction {reduction!r}")


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Cosine similarity between predicted and target tokens."""
    cs = F.cosine_similarity(pred, target, dim=-1)
    if reduction == "mean":
        return cs.mean()
    if reduction == "sum":
        return cs.sum()
    if reduction == "none":
        return cs
    raise ValueError(f"Invalid reduction {reduction!r}")


def expression_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination (R^2) between gene-expression vectors.

    Computed across the gene axis (last dim), then averaged over all
    other axes. Negative values are possible if the model is worse than
    the per-gene mean.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    target_mean = target.mean(dim=-1, keepdim=True)
    ss_res = (target - pred).pow(2).sum(dim=-1)
    ss_tot = (target - target_mean).pow(2).sum(dim=-1).clamp(min=1e-8)
    return (1.0 - ss_res / ss_tot).mean()


def delta_pearson(
    basal: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Pearson correlation of perturbation effect vectors.

    Compares ``pred - basal`` against ``target - basal``. Pearson is the
    standard headline metric in the perturbation-prediction literature
    (e.g. Lotfollahi 2019, Roohani 2023, Bunne 2023) because it is
    invariant to per-gene scale.

    Returns
    -------
    torch.Tensor
        Mean Pearson r over the leading batch dimensions.
    """
    if not (basal.shape == pred.shape == target.shape):
        raise ValueError(
            "Expected matching shapes; got "
            f"basal={tuple(basal.shape)}, pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    delta_pred = pred - basal
    delta_true = target - basal
    delta_pred = delta_pred - delta_pred.mean(dim=-1, keepdim=True)
    delta_true = delta_true - delta_true.mean(dim=-1, keepdim=True)
    num = (delta_pred * delta_true).sum(dim=-1)
    denom = (
        delta_pred.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()
        * delta_true.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()
    )
    return (num / denom).mean()


__all__ = ["cosine_similarity", "delta_pearson", "expression_r2", "latent_l2_error"]
