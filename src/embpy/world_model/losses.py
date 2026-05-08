"""Loss functions for cellular world models.

Covers the common cases used when training latent-dynamics models:

* :func:`latent_mse` -- standard reconstruction objective in latent space.
* :func:`delta_mse` -- regress the perturbation-induced shift
  ``z' - z`` rather than ``z'`` directly. Often more stable when the
  perturbation effect is small relative to ``||z||``.
* :func:`gaussian_nll` -- negative log-likelihood for stochastic models
  predicting ``(mu, log_var)``.
* :func:`info_nce` -- contrastive loss for aligning predicted and true
  perturbed latents within a batch (useful when paired
  basal/perturbed cells are sparse).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean-squared error between predicted and target latents."""
    return F.mse_loss(pred, target, reduction=reduction)


def delta_mse(
    z_basal: torch.Tensor,
    z_pred: torch.Tensor,
    z_perturbed: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE on the perturbation-induced shift.

    Computes ``MSE(z_pred - z_basal, z_perturbed - z_basal)``. Identical
    to :func:`latent_mse` when ``z_pred`` is unconstrained, but enforces a
    useful inductive bias when the model is parameterised as
    ``z_pred = z_basal + delta_theta(z_basal, c)``.
    """
    return F.mse_loss(z_pred - z_basal, z_perturbed - z_basal, reduction=reduction)


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood.

    Parameters
    ----------
    target
        Ground-truth latent, shape ``(batch, latent_dim)``.
    mu
        Predicted mean, same shape as ``target``.
    log_var
        Predicted log-variance, same shape as ``target``. Clamping is
        applied internally to avoid numerical instabilities at extreme
        values.
    reduction
        ``"mean"``, ``"sum"`` or ``"none"``.
    """
    log_var = log_var.clamp(min=-10.0, max=10.0)
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
) -> torch.Tensor:
    """Symmetric InfoNCE loss between two batches of latents.

    Treats the diagonal of the (cosine) similarity matrix as positives
    and all off-diagonal entries as negatives. Symmetrised over the two
    directions so the loss is order-invariant.

    Parameters
    ----------
    pred
        Predicted latents, shape ``(batch, latent_dim)``.
    target
        Ground-truth latents, shape ``(batch, latent_dim)``.
    temperature
        Softmax temperature; lower values sharpen the distribution.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"info_nce expects matching shapes, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = pred_n @ target_n.T / max(temperature, 1e-8)
    labels = torch.arange(pred.size(0), device=pred.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


__all__ = ["delta_mse", "gaussian_nll", "info_nce", "latent_mse"]
