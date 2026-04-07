"""Phenotypic activity scoring via chunked mean average precision.

Measures whether replicates of a perturbation cluster together in
embedding space better than chance.  Uses **chunked matrix
multiplication** for the similarity computation so that memory stays
bounded regardless of dataset size.

Public API
----------
- :func:`phenotypic_activity` -- per-perturbation mAP from an AnnData
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

def _ap_from_ranked_relevance(relevance: np.ndarray) -> float:
    """Average precision from a binary relevance vector (descending order).

    Parameters
    ----------
    relevance
        1-D boolean / 0-1 array where ``True`` / 1 marks a positive
        (same perturbation) and the array is already sorted by
        descending similarity.

    Returns
    -------
    Average precision in [0, 1].  Returns 0 when there are no positives.
    """
    n_pos = relevance.sum()
    if n_pos == 0:
        return 0.0
    tp_cumsum = np.cumsum(relevance)
    precision_at_k = tp_cumsum / np.arange(1, len(relevance) + 1)
    return float((precision_at_k * relevance).sum() / n_pos)


def _chunked_cosine_ap_cpu(
    X_norm: np.ndarray,
    labels: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Compute per-well AP using chunked cosine similarity (CPU).

    Parameters
    ----------
    X_norm
        L2-normalised feature matrix, shape ``(n, d)``, float32.
    labels
        Integer perturbation labels, shape ``(n,)``.
    chunk_size
        Number of query rows per chunk.

    Returns
    -------
    Per-well average precision, shape ``(n,)``.
    """
    n = X_norm.shape[0]
    ap = np.empty(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = X_norm[start:end] @ X_norm.T  # (chunk, n)

        for local_j, global_j in enumerate(range(start, end)):
            sims = sim_chunk[local_j]
            pos_mask = labels == labels[global_j]

            keep = np.ones(n, dtype=bool)
            keep[global_j] = False
            sims_no_self = sims[keep]
            pos_no_self = pos_mask[keep]

            order = np.argsort(-sims_no_self)
            ap[global_j] = _ap_from_ranked_relevance(pos_no_self[order])

    return ap


def _chunked_cosine_ap_gpu(
    X_norm: np.ndarray,
    labels: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Compute per-well AP using chunked cosine similarity (GPU via PyTorch).

    Falls back to CPU implementation if CUDA is not available.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available; falling back to CPU.")
        return _chunked_cosine_ap_cpu(X_norm, labels, chunk_size)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        return _chunked_cosine_ap_cpu(X_norm, labels, chunk_size)

    device = torch.device("cuda")
    n = X_norm.shape[0]
    ap = np.empty(n, dtype=np.float64)

    X_t = torch.from_numpy(X_norm).to(device)
    labels_t = torch.from_numpy(labels).to(device)

    keep_base = torch.ones(n, dtype=torch.bool, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = X_t[start:end] @ X_t.T  # (chunk, n)

        chunk_labels = labels_t[start:end]
        for local_j in range(end - start):
            global_j = start + local_j
            pos_mask = (labels_t == chunk_labels[local_j])

            keep = keep_base.clone()
            keep[global_j] = False
            sims_no_self = sim_chunk[local_j][keep]
            pos_no_self = pos_mask[keep]

            order = torch.argsort(sims_no_self, descending=True)
            ranked_rel = pos_no_self[order].cpu().numpy()
            ap[global_j] = _ap_from_ranked_relevance(ranked_rel)

    return ap


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def phenotypic_activity(
    adata: AnnData,
    obsm_key: str,
    perturbation_col: str,
    control_col: str | None = None,
    control_ids: set[str] | None = None,
    metric: str = "cosine",
    chunk_size: int = 2000,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Compute phenotypic activity (mean average precision) per perturbation.

    Measures how well replicates of each perturbation cluster together
    in embedding space relative to all other wells.  Uses **chunked
    matrix multiplication** to keep memory bounded regardless of
    dataset size.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]`` and
        perturbation metadata in ``obs[perturbation_col]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    perturbation_col
        Column in ``.obs`` with perturbation identifiers (e.g.
        ``"Metadata_JCP2022"`` or ``"gene"``).
    control_col
        Optional column used to identify control wells.  When
        provided together with *control_ids*, control wells are
        excluded before computing activity.
    control_ids
        Set of values in *control_col* that mark negative controls.
    metric
        Similarity metric.  Currently only ``"cosine"`` is supported.
    chunk_size
        Number of query rows processed per chunk.  Controls peak
        memory: ``chunk_size * n_wells * 4`` bytes.  Default 2000
        uses ~400 MB for 51K wells.
    use_gpu
        If ``True``, use PyTorch CUDA for the matrix multiplication
        and argsort steps.  Falls back to CPU when CUDA is absent.

    Returns
    -------
    :class:`~pandas.DataFrame` with columns:

    - ``perturbation`` -- perturbation identifier
    - ``mean_ap`` -- mean of per-well AP across replicates
    - ``normalized_mean_ap`` -- bias-corrected mAP (accounts for
      replicate count)
    - ``n_wells`` -- number of wells for that perturbation

    Notes
    -----
    Memory is *O(chunk_size * n)* instead of *O(n^2)*.  For 51K
    wells with the default ``chunk_size=2000``, peak allocation is
    ~400 MB vs ~21 GB for a dense similarity matrix.

    The normalised AP is ``(AP - E[AP]) / (1 - E[AP])`` where
    ``E[AP] = n_pos / (n - 1)`` is the expected AP under a random
    ranking.  This corrects for perturbations with many replicates
    having inflated raw AP.

    Examples
    --------
    >>> import embpy.tl as tl
    >>> activity = tl.phenotypic_activity(
    ...     adata,
    ...     obsm_key="X_jump",
    ...     perturbation_col="Metadata_JCP2022",
    ...     control_col="is_control",
    ...     control_ids={True},
    ... )
    """
    from anndata import AnnData as _AnnData  # deferred for type-check perf

    if not isinstance(adata, _AnnData):
        raise TypeError(f"Expected AnnData, got {type(adata).__name__}.")
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if perturbation_col not in adata.obs.columns:
        raise KeyError(
            f"'{perturbation_col}' not in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if metric != "cosine":
        raise ValueError(
            f"Unsupported metric '{metric}'. Currently only 'cosine' is supported."
        )

    # ---- filter controls ------------------------------------------------
    mask = np.ones(adata.n_obs, dtype=bool)
    if control_col is not None and control_ids is not None:
        if control_col not in adata.obs.columns:
            raise KeyError(f"control_col '{control_col}' not in adata.obs.")
        mask = ~adata.obs[control_col].isin(control_ids).values

    X = np.asarray(adata.obsm[obsm_key][mask], dtype=np.float32)
    pert_labels_raw = adata.obs[perturbation_col].values[mask]
    n = X.shape[0]

    logger.info(
        "Computing phenotypic activity for %d wells (%d perturbations), "
        "chunk_size=%d, gpu=%s",
        n,
        len(np.unique(pert_labels_raw)),
        chunk_size,
        use_gpu,
    )

    # ---- encode labels as integers for fast comparison ------------------
    unique_labels, int_labels = np.unique(pert_labels_raw, return_inverse=True)

    # ---- L2-normalise for cosine similarity via dot product -------------
    X_norm = normalize(X, norm="l2", axis=1).astype(np.float32)

    # ---- compute per-well AP --------------------------------------------
    if use_gpu:
        well_ap = _chunked_cosine_ap_gpu(X_norm, int_labels, chunk_size)
    else:
        well_ap = _chunked_cosine_ap_cpu(X_norm, int_labels, chunk_size)

    # ---- aggregate per perturbation -------------------------------------
    rows = []
    for label_idx, label in enumerate(unique_labels):
        well_mask = int_labels == label_idx
        n_wells = int(well_mask.sum())
        mean_ap = float(well_ap[well_mask].mean())

        n_pos = n_wells - 1  # replicates excluding self
        expected_ap = n_pos / (n - 1) if n > 1 else 0.0
        if expected_ap < 1.0:
            nap = (mean_ap - expected_ap) / (1.0 - expected_ap)
        else:
            nap = 0.0

        rows.append({
            "perturbation": label,
            "mean_ap": mean_ap,
            "normalized_mean_ap": nap,
            "n_wells": n_wells,
        })

    result = pd.DataFrame(rows).sort_values("mean_ap", ascending=False)
    result = result.reset_index(drop=True)

    logger.info(
        "Done. Overall mean AP = %.4f, normalised = %.4f",
        result["mean_ap"].mean(),
        result["normalized_mean_ap"].mean(),
    )

    return result
