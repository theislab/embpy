"""Numerical evaluation metrics for the world model.

Two flavours coexist:

* The torch-based latent metrics (:func:`latent_l2_error`,
  :func:`cosine_similarity`, :func:`expression_r2`,
  :func:`delta_pearson`) used by the training-time rollout summary.
* The numpy-based gene-expression metrics
  (:func:`mse`, :func:`mae`, :func:`r2_score`, :func:`pearson_corr`,
  :func:`spearman_corr`, :func:`deg_overlap_top_k`) consumed by the
  baseline / cell-eval comparison pipeline. These mirror the headline
  metrics emitted by the STATE / cell-eval suite so internal results
  and external cell-eval results are directly comparable.
"""

from __future__ import annotations

import numpy as np
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


# ----------------------------------------------------------------------
# Gene-expression-space metrics (numpy, used by the comparison pipeline)
# ----------------------------------------------------------------------


def mse(real: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error between two arrays (any matching shape)."""
    real = np.asarray(real, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if real.shape != pred.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {pred.shape}")
    return float(np.mean((real - pred) ** 2))


def mae(real: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error."""
    real = np.asarray(real, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if real.shape != pred.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {pred.shape}")
    return float(np.mean(np.abs(real - pred)))


def r2_score(real: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination, ``1 - SS_res / SS_tot``.

    Computed over flattened arrays. Returns ``nan`` when the target has
    zero variance (so we never divide by zero silently).
    """
    real = np.asarray(real, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    ss_res = float(np.sum((real - pred) ** 2))
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pearson_corr(real: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation between flattened arrays.

    Falls back to a numpy implementation when scipy is missing so the
    helper has no hard scipy dependency.
    """
    real = np.asarray(real, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if real.size < 2:
        return float("nan")
    try:
        from scipy.stats import pearsonr  # noqa: PLC0415

        r = float(pearsonr(real, pred)[0])
    except ImportError:
        r_mat = np.corrcoef(real, pred)
        r = float(r_mat[0, 1])
    return r


def spearman_corr(real: np.ndarray, pred: np.ndarray) -> float:
    """Spearman rank correlation. Requires scipy; returns ``nan`` if missing."""
    real = np.asarray(real, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if real.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr  # noqa: PLC0415

        return float(spearmanr(real, pred)[0])
    except ImportError:
        r_real = np.argsort(np.argsort(real))
        r_pred = np.argsort(np.argsort(pred))
        return float(np.corrcoef(r_real, r_pred)[0, 1])


def deg_overlap_top_k(
    real_delta: np.ndarray,
    pred_delta: np.ndarray,
    k: int = 50,
) -> float:
    """Top-K differentially expressed gene overlap.

    Both inputs are perturbation-effect vectors of shape ``(n_genes,)``.
    Genes are ranked by absolute effect magnitude; the score is the
    Jaccard-style overlap of the top-K sets divided by ``K``.
    """
    real_delta = np.asarray(real_delta, dtype=np.float32).reshape(-1)
    pred_delta = np.asarray(pred_delta, dtype=np.float32).reshape(-1)
    if real_delta.shape != pred_delta.shape:
        raise ValueError(f"shape mismatch: {real_delta.shape} vs {pred_delta.shape}")
    k = int(min(k, real_delta.size))
    if k <= 0:
        return float("nan")
    top_real = np.argsort(-np.abs(real_delta))[:k]
    top_pred = np.argsort(-np.abs(pred_delta))[:k]
    return float(len(set(top_real.tolist()) & set(top_pred.tolist())) / k)


__all__ = [
    "cosine_similarity",
    "deg_overlap_top_k",
    "delta_pearson",
    "expression_r2",
    "latent_l2_error",
    "mae",
    "mse",
    "pearson_corr",
    "r2_score",
    "spearman_corr",
]
