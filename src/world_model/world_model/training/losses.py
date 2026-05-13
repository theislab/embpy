"""Loss functions used by the world model.

Kept as plain functions so they can be unit-tested without instantiating
any model. Reduce-modes follow the PyTorch convention
(``"mean"`` / ``"sum"`` / ``"none"``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_mse(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """MSE between predicted and target latent tokens."""
    return F.mse_loss(pred, target, reduction=reduction)


def delta_mse(
    z_basal: torch.Tensor,
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE on the perturbation-induced shift ``z' - z_basal``.

    Equivalent to :func:`latent_mse` when both predictions and targets
    are unconstrained; encodes the residual inductive bias when the
    model is parameterised as ``z_pred = z_basal + delta(...)``.
    """
    return F.mse_loss(z_pred - z_basal, z_target - z_basal, reduction=reduction)


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
    log_var_clamp: tuple[float, float] = (-10.0, 10.0),
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log likelihood.

    Parameters
    ----------
    target
        Ground-truth tensor.
    mu, log_var
        Predicted mean and per-element log-variance.
    reduction
        ``"mean"``, ``"sum"`` or ``"none"``.
    log_var_clamp
        Numerical safety net on the predicted log-variance.
    """
    log_var = log_var.clamp(min=log_var_clamp[0], max=log_var_clamp[1])
    nll = 0.5 * (log_var + (target - mu).pow(2) * torch.exp(-log_var))
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"Invalid reduction '{reduction}'. Use 'mean', 'sum' or 'none'.")


def info_nce(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.1,
    *,
    valid_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE between two batches of tokens.

    The diagonal of the cosine-similarity matrix is treated as the
    positive set; off-diagonal entries are negatives. Symmetrised over
    the two directions.

    Parameters
    ----------
    pred, target
        ``(N, d)`` matched tensors. Row ``i`` of ``pred`` is contrasted
        against row ``i`` of ``target`` (positive) versus every other
        row of ``target`` (negative).
    temperature
        Softmax temperature applied to the cosine-similarity logits.
    valid_negative_mask
        Optional ``(N, N)`` bool tensor. ``True`` entries are kept in
        the softmax denominator; ``False`` entries are masked out
        (set to ``-inf`` before cross-entropy). The diagonal MUST be
        ``True`` -- it carries the positives. Used by the world-model
        caller to mask out same-perturbation pairs from the negative
        pool (hard-negative mining), which otherwise punish the
        encoder for embedding biologically-similar items nearby.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"info_nce expects matching shapes, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = pred_n @ target_n.T / max(temperature, 1e-8)
    if valid_negative_mask is not None:
        if valid_negative_mask.shape != logits.shape:
            raise ValueError(
                f"valid_negative_mask shape {tuple(valid_negative_mask.shape)} "
                f"does not match logits {tuple(logits.shape)}"
            )
        logits = logits.masked_fill(~valid_negative_mask, float("-inf"))
    labels = torch.arange(pred.size(0), device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


__all__ = ["delta_mse", "gaussian_nll", "info_nce", "latent_mse"]
