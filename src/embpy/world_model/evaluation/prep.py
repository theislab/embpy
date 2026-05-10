"""Build AnnData objects in the format ``cell_eval`` expects.

Both ``real`` and ``pred`` AnnData objects share:

* same gene order (``var_names``),
* an ``obs`` column ``perturbation`` carrying the per-cell label,
* one row per (cell, perturbation) prediction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _require_anndata():  # type: ignore[no-untyped-def]
    try:
        import anndata as ad  # noqa: PLC0415

        return ad
    except ImportError as exc:
        raise ImportError(
            "anndata is required for evaluation prep. Install with: pip install anndata"
        ) from exc


def build_real_anndata(
    expression: np.ndarray,
    perturbation_labels: np.ndarray,
    test_indices: np.ndarray,
    test_perturbations: list[str],
    gene_symbols: list[str],
    *,
    perturbation_key: str = "perturbation",
) -> Any:
    """Subset ``expression`` to ``test_indices`` filtered to ``test_perturbations``."""
    ad = _require_anndata()
    test_mask = np.zeros(expression.shape[0], dtype=bool)
    test_mask[test_indices] = True
    keep_set = set(test_perturbations)
    label_mask = np.array([lbl in keep_set for lbl in perturbation_labels])
    final_idx = np.flatnonzero(test_mask & label_mask)

    if final_idx.size == 0:
        raise ValueError("No cells survive the test_indices x test_perturbations filter.")

    X = expression[final_idx].astype(np.float32)
    obs = {perturbation_key: perturbation_labels[final_idx].astype(str)}
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = list(gene_symbols)
    return adata


def build_pred_anndata(
    predictions: dict[str, np.ndarray],
    real_adata: Any,
    *,
    perturbation_key: str = "perturbation",
) -> Any:
    """Tile per-perturbation predictions to match ``real_adata`` cell ordering.

    Cell-eval-style metrics expect identical (n_cells, n_genes) layouts
    between real and pred. We therefore broadcast each per-perturbation
    prediction to the count of test cells carrying that perturbation.
    """
    ad = _require_anndata()
    if real_adata.n_vars != next(iter(predictions.values())).size:
        raise ValueError(
            "predictions and real_adata.var must share gene count "
            f"({next(iter(predictions.values())).size} vs {real_adata.n_vars})"
        )
    labels = np.asarray(real_adata.obs[perturbation_key].values).astype(str)
    n_genes = real_adata.n_vars
    X = np.empty((labels.size, n_genes), dtype=np.float32)
    n_missing = 0
    for i, lbl in enumerate(labels):
        vec = predictions.get(str(lbl))
        if vec is None:
            X[i] = 0.0
            n_missing += 1
        else:
            X[i] = vec
    if n_missing:
        logger.warning(
            "build_pred_anndata: %d/%d cells had no prediction (filled with zeros)",
            n_missing, labels.size,
        )
    pred = ad.AnnData(X=X, obs=real_adata.obs.copy())
    pred.var_names = list(real_adata.var_names)
    return pred


__all__ = ["build_pred_anndata", "build_real_anndata"]
