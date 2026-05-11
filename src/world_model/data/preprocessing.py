"""Preprocessing helpers for transcriptomics counts.

Operates on dense numpy arrays / scipy sparse matrices so the dataset
classes can keep their adapter responsibilities separate from the
numerical preprocessing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def log_normalize_counts(
    x: np.ndarray,
    target_sum: float = 1e4,
    eps: float = 1e-8,
) -> np.ndarray:
    """Library-size normalise then log1p transform.

    Parameters
    ----------
    x
        ``(n_cells, n_genes)`` non-negative count matrix (dense).
    target_sum
        Target library size after normalisation.
    eps
        Numerical floor for the per-cell library size.

    Returns
    -------
    np.ndarray
        ``log1p(x / sum(x, axis=1) * target_sum)``.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")
    sizes = np.asarray(x.sum(axis=1)).reshape(-1)
    sizes = np.maximum(sizes, eps)
    scale = (target_sum / sizes).astype(np.float32)
    out = (x.astype(np.float32) * scale[:, None])
    return np.log1p(out, dtype=np.float32)


def select_highly_variable_genes(
    x: np.ndarray,
    n_top: int,
    min_cells: int = 3,
) -> np.ndarray:
    """Return indices of the top-N genes by normalised variance.

    A simple, dependency-free fallback for environments without scanpy.
    Mimics the scanpy "seurat_v3"-flavoured ranking closely enough for
    benchmark training, but should not be relied on for publication
    results -- prefer ``sc.pp.highly_variable_genes`` upstream.

    Parameters
    ----------
    x
        ``(n_cells, n_genes)`` raw counts.
    n_top
        Number of genes to keep. If 0 or negative, returns all gene indices.
    min_cells
        Minimum number of cells expressing a gene for it to be eligible.

    Returns
    -------
    np.ndarray
        Sorted indices into the gene axis.
    """
    n_genes = x.shape[1]
    if n_top <= 0 or n_top >= n_genes:
        return np.arange(n_genes, dtype=np.int64)

    presence = (x > 0).sum(axis=0)
    eligible = presence >= min_cells

    mu = x.mean(axis=0)
    var = x.var(axis=0)
    # Coefficient-of-variation-like score; clamp small means.
    score = var / np.maximum(mu, 1e-6)
    score = np.where(eligible, score, -np.inf)

    order = np.argsort(-score)
    keep = order[:n_top]
    keep.sort()
    logger.info("Selected %d HVGs out of %d (eligible=%d).", keep.size, n_genes, int(eligible.sum()))
    return keep.astype(np.int64)


def sequence_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of dataset samples into a batch dict.

    The dataset classes already produce torch tensors, so this is just a
    concatenation along a new leading dim. Pulled out into its own
    function so the dataloader builder can pass it explicitly.
    """
    import torch  # noqa: PLC0415

    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor):
            batch[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            batch[k] = [s[k] for s in samples]
    return batch


__all__ = [
    "log_normalize_counts",
    "select_highly_variable_genes",
    "sequence_collate_fn",
]
